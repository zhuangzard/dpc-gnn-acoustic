#!/usr/bin/env python3
"""
train.py — DPC-GNN-Acoustic v2 Training Script.

PyTorch Lightning-style training loop with:
  - Curriculum learning (homogeneous → layered → heterogeneous)
  - Warmup + cosine annealing scheduler
  - Gradient clipping
  - WandB logging (optional)
  - Checkpointing (save top-k)
  - Physics validation during training

Usage:
    python train.py --config configs/default_2d.yaml
    python train.py --config configs/default_2d.yaml --resume checkpoints/last.pt
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.dpc_gnn_acoustic import DPCGNNAcousticV2, create_model
from data.kwave_dataset import KWaveDataset, create_dataloader
from losses.combined_loss import CombinedLoss, create_loss
from utils.metrics import evaluate_sample, print_metrics


class Trainer:
    """Training loop for DPC-GNN-Acoustic v2.
    
    Args:
        config: Configuration dictionary
        device: Training device
        resume_from: Path to checkpoint for resuming
    """
    
    def __init__(
        self,
        config: dict,
        device: str = 'cuda',
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.device = device
        
        train_cfg = config.get('training', {})
        self.max_epochs = train_cfg.get('max_epochs', 500)
        self.lr = train_cfg.get('lr', 1e-4)
        self.weight_decay = train_cfg.get('weight_decay', 1e-6)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 20)
        self.grad_clip = train_cfg.get('grad_clip', 1.0)
        
        log_cfg = config.get('logging', {})
        self.checkpoint_dir = log_cfg.get('checkpoint_dir', 'checkpoints')
        self.log_every = log_cfg.get('log_every_n_steps', 10)
        self.val_every = log_cfg.get('val_every_n_epochs', 5)
        self.save_top_k = log_cfg.get('save_top_k', 3)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # ── Create model ──
        self.model = create_model(config, device)
        
        # ── Create optimizer ──
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # ── Create scheduler ──
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.lr * 0.01,
        )
        
        # ── Create loss ──
        self.criterion = create_loss(config)
        
        # ── Mixed precision training ──
        self.use_amp = train_cfg.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("🔥 Mixed precision training enabled (AMP)")
        
        # ── Create data loaders ──
        data_cfg = config.get('data', {})
        data_dir = data_cfg.get('kwave_data_dir', 'data/kwave_gt')
        
        self.train_loader = create_dataloader(
            data_dir, split='train',
            grid_resolution=config.get('graph', {}).get('grid_resolution', 256),
            frequency=float(config.get('physics', {}).get('frequency', 5e6)),
            n_elements=config.get('probe', {}).get('n_elements', 128),
        )
        self.val_loader = create_dataloader(
            data_dir, split='val',
            grid_resolution=config.get('graph', {}).get('grid_resolution', 256),
            frequency=float(config.get('physics', {}).get('frequency', 5e6)),
            n_elements=config.get('probe', {}).get('n_elements', 128),
        )
        
        # ── Curriculum learning setup ──
        self.curriculum_cfg = train_cfg.get('curriculum', {})
        self.curriculum_enabled = self.curriculum_cfg.get('enabled', False)
        self.curriculum_stages = self.curriculum_cfg.get('stages', [])
        self.current_stage = 0
        self.stage_start_epoch = 0
        
        if self.curriculum_enabled:
            print(f"📚 Curriculum learning enabled with {len(self.curriculum_stages)} stages")
            for i, stage in enumerate(self.curriculum_stages):
                print(f"   Stage {i+1}: {stage.get('epochs', '?')} epochs, medium={stage.get('medium', '?')}")
        
        # ── Training state ──
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_checkpoints = []  # (loss, path) sorted
        self.history = []
        
        # ── WandB (optional) ──
        self.wandb_run = None
        try:
            import wandb
            wandb_project = log_cfg.get('wandb_project', 'dpc-gnn-acoustic-v2')
            self.wandb_run = wandb.init(
                project=wandb_project,
                config=config,
                name=f"v2_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            print(f"✅ WandB initialized: {wandb_project}")
        except (ImportError, Exception) as e:
            print(f"⚠️ WandB not available: {e}")
        
        # ── Resume ──
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.max_epochs} epochs")
        print(f"{'='*60}\n")
        
        t_start = time.time()
        
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            
            # ── Warmup ──
            if epoch < self.warmup_epochs:
                lr_scale = (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.lr * lr_scale
            
            # ── Train epoch ──
            train_metrics = self._train_epoch()
            
            # ── Curriculum learning: update stage ──
            if self.curriculum_enabled:
                self._update_curriculum_stage(epoch)
            
            # ── Scheduler step ──
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # ── Validation ──
            if (epoch + 1) % self.val_every == 0 or epoch == self.max_epochs - 1:
                val_metrics = self._validate()
                
                # Checkpointing
                val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                
                # Log validation
                if self.wandb_run:
                    import wandb
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=self.global_step)
            
            # ── Periodic checkpoint ──
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch, train_metrics['train_loss'])
            
            # ── Logging ──
            self.history.append({
                'epoch': epoch,
                **train_metrics,
            })
            
            if self.wandb_run:
                import wandb
                wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=self.global_step)
            
            # Print progress
            if epoch == 0 or (epoch + 1) % self.log_every == 0 or epoch == self.max_epochs - 1:
                lr_current = self.optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch+1:4d}/{self.max_epochs} | "
                    f"Loss: {train_metrics['train_loss']:.6e} | "
                    f"L1: {train_metrics.get('l1', 0):.4e} | "
                    f"SSIM: {train_metrics.get('ssim', 0):.4f} | "
                    f"LR: {lr_current:.2e}"
                )
        
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  Best val loss: {self.best_val_loss:.6e}")
        print(f"{'='*60}\n")
        
        # Save final
        self._save_checkpoint(self.max_epochs - 1, self.best_val_loss, is_best=False)
        
        # Save history
        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _update_curriculum_stage(self, epoch: int):
        """Update curriculum learning stage based on current epoch."""
        if not self.curriculum_enabled or not self.curriculum_stages:
            return
        
        # Calculate cumulative epochs for each stage
        cumulative_epochs = 0
        new_stage = 0
        
        for i, stage in enumerate(self.curriculum_stages):
            stage_epochs = stage.get('epochs', 100)
            cumulative_epochs += stage_epochs
            
            if epoch < cumulative_epochs:
                new_stage = i
                break
        else:
            new_stage = len(self.curriculum_stages) - 1
        
        # Check if stage changed
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            self.stage_start_epoch = epoch
            
            stage = self.curriculum_stages[new_stage]
            print(f"\n{'='*60}")
            print(f"📚 Curriculum: Entering Stage {new_stage + 1}/{len(self.curriculum_stages)}")
            print(f"   Medium: {stage.get('medium', 'default')}")
            print(f"   Physics loss weight: {stage.get('loss_weight_physics', 'unchanged')}")
            print(f"{'='*60}\n")
            
            # Update physics loss weight if specified
            if 'loss_weight_physics' in stage:
                self.criterion.lambda_physics = stage['loss_weight_physics']
    
    def _get_current_stage_info(self) -> dict:
        """Get current curriculum stage information."""
        if not self.curriculum_enabled:
            return {}
        
        if self.current_stage < len(self.curriculum_stages):
            return self.curriculum_stages[self.current_stage]
        return {}
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {}
        
        for batch in self.train_loader:
            self.global_step += 1
            
            # Move to device
            hu = batch['hu'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            bmode_gt = batch['bmode_gt'].to(self.device)
            transducer_idx = batch['transducer_idx'].to(self.device)
            positions = batch['positions'].to(self.device)
            domain_size = batch['domain_size'].to(self.device)
            
            node_props = {k: v.to(self.device) for k, v in batch['node_props'].items()}
            
            # Forward with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        hu, edge_index, edge_attr, node_props,
                        transducer_idx, positions, domain_size,
                    )
                
                # FIX #9: Compute physics loss OUTSIDE autocast (force float32)
                # AMP with float16 + very small dt² values causes overflow/underflow
                # in the wave equation residual computation.
                with torch.cuda.amp.autocast(enabled=False):
                    physics_loss = self.model.compute_physics_loss()
                
                with autocast():
                    total_loss, loss_dict = self.criterion(
                        outputs['bmode'],
                        bmode_gt,
                        physics_loss=physics_loss.float(),  # FIX #9: ensure float32
                        energy_history=outputs.get('energy_history'),
                    )
                
                # Backward with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping (unscale first)
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Step optimizer and update scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training (FP32)
                outputs = self.model(
                    hu, edge_index, edge_attr, node_props,
                    transducer_idx, positions, domain_size,
                )
                
                # Compute losses
                physics_loss = self.model.compute_physics_loss()
                
                total_loss, loss_dict = self.criterion(
                    outputs['bmode'],
                    bmode_gt,
                    physics_loss=physics_loss,
                    energy_history=outputs.get('energy_history'),
                )
                
                # Backward
                total_loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Step optimizer
                self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
            for k, v in loss_dict.items():
                epoch_metrics.setdefault(k, []).append(v)
            
            # Check for NaN
            if math.isnan(total_loss.item()):
                print(f"  ⚠️ NaN loss at step {self.global_step}!")
                break
        
        # Aggregate
        result = {'train_loss': sum(epoch_losses) / max(len(epoch_losses), 1)}
        for k, v in epoch_metrics.items():
            result[k] = sum(v) / len(v)
        
        return result
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        val_losses = []
        val_metrics_all = []
        
        for batch in self.val_loader:
            hu = batch['hu'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            bmode_gt = batch['bmode_gt'].to(self.device)
            transducer_idx = batch['transducer_idx'].to(self.device)
            positions = batch['positions'].to(self.device)
            domain_size = batch['domain_size'].to(self.device)
            node_props = {k: v.to(self.device) for k, v in batch['node_props'].items()}
            
            outputs = self.model(
                hu, edge_index, edge_attr, node_props,
                transducer_idx, positions, domain_size,
            )
            
            total_loss, loss_dict = self.criterion(outputs['bmode'], bmode_gt)
            val_losses.append(total_loss.item())
            
            # Compute evaluation metrics
            metrics = evaluate_sample(
                outputs['bmode'], bmode_gt,
                outputs.get('energy_history'),
            )
            val_metrics_all.append(metrics)
        
        # Aggregate
        result = {'val_loss': sum(val_losses) / max(len(val_losses), 1)}
        
        if val_metrics_all:
            for key in val_metrics_all[0]:
                values = [m[key] for m in val_metrics_all if isinstance(m.get(key), (int, float))]
                if values:
                    result[f'val_{key}'] = sum(values) / len(values)
        
        # Print validation results
        print(f"\n  📊 Validation: loss={result['val_loss']:.6e}", end="")
        if 'val_ssim' in result:
            print(f" | SSIM={result['val_ssim']:.4f}", end="")
        if 'val_psnr' in result:
            print(f" | PSNR={result['val_psnr']:.2f}dB", end="")
        print()
        
        return result
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'global_step': self.global_step,
        }
        
        # Save last
        last_path = os.path.join(self.checkpoint_dir, 'last.pt')
        torch.save(ckpt, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(ckpt, best_path)
            print(f"  💾 Best checkpoint saved (loss={loss:.6e})")
        
        # Top-k management
        epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch:04d}.pt')
        torch.save(ckpt, epoch_path)
        self.best_checkpoints.append((loss, epoch_path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        
        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.save_top_k:
            _, path = self.best_checkpoints.pop()
            if os.path.exists(path) and 'best' not in path and 'last' not in path:
                os.remove(path)
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint for resuming."""
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        
        print(f"  Resumed from epoch {self.epoch}, best loss={self.best_val_loss:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Train DPC-GNN-Acoustic v2")
    parser.add_argument('--config', type=str, default='configs/default_2d.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override from CLI
    if args.epochs:
        config.setdefault('training', {})['max_epochs'] = args.epochs
    if args.lr:
        config.setdefault('training', {})['lr'] = args.lr
    
    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"DPC-GNN-Acoustic v2 — Training")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # Train
    trainer = Trainer(config, device=device, resume_from=args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
