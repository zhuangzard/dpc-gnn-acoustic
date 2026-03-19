#!/usr/bin/env python3
"""
train_v3.py — Training script for DPC-GNN-Acoustic v3.

Uses the physics-correct v3 model with:
  - LeapfrogWavePropagator (correct Taylor init, graph Laplacian, PML)
  - DASBeamformDecoder (delay-and-sum with Hilbert envelope)
  - CombinedLossV2 (L1 + SSIM + dimensionless physics loss)
  - AMP mixed precision (physics loss in float32)

Usage:
    python train_v3.py --config configs/default_2d.yaml
    python train_v3.py --config configs/default_2d.yaml --resume checkpoints_v3/last.pt
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.dpc_gnn_acoustic_v3 import DPCGNNAcousticV3, create_model_v3
from data.kwave_dataset import KWaveDataset, create_dataloader
from data.analytical_dataset import AnalyticalDataset, create_analytical_dataloader
from losses.combined_loss_v2 import CombinedLossV2, create_loss_v2


class TrainerV3:
    """Training loop for DPC-GNN-Acoustic v3.

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
        dataset_type: str = 'kwave',
    ):
        self.config = config
        self.device = device
        self.dataset_type = dataset_type

        train_cfg = config.get('training', {})
        self.max_epochs = train_cfg.get('max_epochs', 500)
        self.lr = train_cfg.get('lr', 1e-4)
        self.weight_decay = train_cfg.get('weight_decay', 1e-6)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 20)
        self.grad_clip = train_cfg.get('grad_clip', 1.0)

        log_cfg = config.get('logging', {})
        self.checkpoint_dir = log_cfg.get('checkpoint_dir', 'checkpoints_v3')
        self.log_every = log_cfg.get('log_every_n_steps', 10)
        self.val_every = log_cfg.get('val_every_n_epochs', 5)
        self.save_top_k = log_cfg.get('save_top_k', 3)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ── Create model ──
        self.model = create_model_v3(config, device)

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
        self.criterion = create_loss_v2(config)

        # ── Mixed precision ──
        self.use_amp = train_cfg.get('use_amp', True) and device != 'cpu'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("🔥 Mixed precision training enabled (AMP)")

        # ── Data loaders ──
        if dataset_type == 'analytical':
            print("📐 Using ANALYTICAL dataset (exact solutions as GT)")
            grid_res = config.get('graph', {}).get('grid_resolution', 256)
            n_elems = config.get('probe', {}).get('n_elements', 128)
            physics_cfg = config.get('physics', {})
            n_steps = int(physics_cfg.get('n_time_steps', 200))
            dt_val = float(physics_cfg.get('dt', 2e-8))

            self.train_loader = create_analytical_dataloader(
                split='train',
                grid_resolution=grid_res,
                n_time_steps=n_steps,
                dt=dt_val,
                n_elements=n_elems,
            )
            self.val_loader = create_analytical_dataloader(
                split='val',
                grid_resolution=grid_res,
                n_time_steps=n_steps,
                dt=dt_val,
                n_elements=n_elems,
            )
        else:
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

        # ── Training state ──
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_checkpoints = []
        self.history = []

        # ── Resume ──
        if resume_from:
            self._load_checkpoint(resume_from)

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"DPC-GNN-Acoustic v3 — Starting training: {self.max_epochs} epochs")
        print(f"  Physics: dt={self.model.dt:.1e}, steps={self.model.n_time_steps}")
        print(f"  CFL ≈ 1540*{self.model.dt:.1e}/2.34e-4 ≈ {1540*self.model.dt/2.34e-4:.3f}")
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

            # ── Scheduler step ──
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # ── Validation ──
            if (epoch + 1) % self.val_every == 0 or epoch == self.max_epochs - 1:
                val_metrics = self._validate()
                val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss, is_best=True)

            # ── Periodic checkpoint ──
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch, train_metrics['train_loss'])

            # ── History ──
            self.history.append({'epoch': epoch, **train_metrics})

            # ── Logging ──
            if epoch == 0 or (epoch + 1) % self.log_every == 0 or epoch == self.max_epochs - 1:
                lr_current = self.optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch+1:4d}/{self.max_epochs} | "
                    f"Loss: {train_metrics['train_loss']:.6e} | "
                    f"L1: {train_metrics.get('l1', 0):.4e} | "
                    f"SSIM: {train_metrics.get('ssim', 0):.4f} | "
                    f"Phys: {train_metrics.get('physics', 0):.4e} | "
                    f"LR: {lr_current:.2e}"
                )

            # ── Detailed metrics every 10 epochs ──
            if (epoch + 1) % 10 == 0:
                self._print_detailed_metrics(epoch, train_metrics)

        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  Best val loss: {self.best_val_loss:.6e}")
        print(f"{'='*60}\n")

        self._save_checkpoint(self.max_epochs - 1, self.best_val_loss, is_best=False)

        with open(os.path.join(self.checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

    def _print_detailed_metrics(self, epoch: int, metrics: Dict):
        """Print detailed metrics every N epochs."""
        print(f"\n  ── Epoch {epoch+1} Detailed ──")
        print(f"    L1 loss:      {metrics.get('l1', 0):.6e}")
        print(f"    SSIM:         {metrics.get('ssim', 0):.6f}")
        print(f"    Physics loss: {metrics.get('physics', 0):.6e}")
        print(f"    Total loss:   {metrics.get('total', metrics['train_loss']):.6e}")

        # Energy conservation from propagator
        energy_hist = getattr(self.model.propagator, 'energy_history', [])
        if len(energy_hist) >= 2:
            e_initial = energy_hist[0]
            e_final = energy_hist[-1]
            if e_initial > 0:
                conservation_ratio = e_final / e_initial
                print(f"    Energy: initial={e_initial:.4e}, final={e_final:.4e}, ratio={conservation_ratio:.4f}")

        # Pressure statistics
        p_hist = getattr(self.model.propagator, 'pressure_history', None)
        if p_hist is not None and len(p_hist) > 0:
            p_last = p_hist[-1].detach()
            print(f"    Pressure (final): max={p_last.max().item():.4e}, min={p_last.min().item():.4e}, rms={p_last.pow(2).mean().sqrt().item():.4e}")
        print()

    def _compute_analytical_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        physics_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for analytical dataset: compare pressure fields directly.
        
        Loss = L1(pred_pressure, gt_pressure) 
             + SSIM(pred_final_field, gt_final_field) 
             + λ_phys * physics_loss
        """
        import torch.nn.functional as F

        # ── Pressure history comparison ──
        # Model outputs: all_pressures is list of (N,1) tensors, length T
        pred_pressures = outputs['all_pressures']  # list of (N, 1) tensors
        gt_pressures = batch['pressure_gt'].to(self.device)  # (T, N)
        
        T_pred = len(pred_pressures)
        T_gt = gt_pressures.shape[0]
        T = min(T_pred, T_gt)
        
        # Stack predicted pressures: (T, N)
        pred_stack = torch.cat([p.squeeze(-1).unsqueeze(0) for p in pred_pressures[:T]], dim=0)
        gt_stack = gt_pressures[:T]
        
        # Normalize both for fair comparison
        pred_norm = pred_stack / (pred_stack.abs().max() + 1e-10)
        gt_norm = gt_stack / (gt_stack.abs().max() + 1e-10)
        
        # L1 loss on full pressure history
        l1_pressure = F.l1_loss(pred_norm, gt_norm)
        
        # ── Final field SSIM ──
        # Reshape to 2D images for SSIM
        nx = ny = self.config.get('graph', {}).get('grid_resolution', 256)
        pred_final_2d = pred_norm[-1].view(nx, ny)
        gt_final_2d = gt_norm[-1].view(nx, ny)
        
        # SSIM on final pressure field
        ssim_val = self.criterion.ssim_loss(
            pred_final_2d.unsqueeze(0).unsqueeze(0),
            gt_final_2d.unsqueeze(0).unsqueeze(0),
        )
        
        # Total loss
        lambda_phys = self.criterion.lambda_physics
        total_loss = l1_pressure + 0.5 * ssim_val + lambda_phys * physics_loss
        
        loss_dict = {
            'l1': l1_pressure.item(),
            'ssim': ssim_val.item(),
            'physics': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        use_analytical = (self.dataset_type == 'analytical')

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

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        hu, edge_index, edge_attr, node_props,
                        transducer_idx, positions, domain_size,
                    )

                # Physics loss MUST be float32
                with torch.cuda.amp.autocast(enabled=False):
                    physics_loss = self.model.compute_physics_loss()

                with autocast():
                    if use_analytical:
                        total_loss, loss_dict = self._compute_analytical_loss(
                            outputs, batch, physics_loss.float(),
                        )
                    else:
                        total_loss, loss_dict = self.criterion(
                            outputs['bmode'], bmode_gt,
                            physics_loss=physics_loss.float(),
                        )

                self.scaler.scale(total_loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    hu, edge_index, edge_attr, node_props,
                    transducer_idx, positions, domain_size,
                )
                physics_loss = self.model.compute_physics_loss()
                if use_analytical:
                    total_loss, loss_dict = self._compute_analytical_loss(
                        outputs, batch, physics_loss,
                    )
                else:
                    total_loss, loss_dict = self.criterion(
                        outputs['bmode'], bmode_gt,
                        physics_loss=physics_loss,
                    )
                total_loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            epoch_losses.append(total_loss.item())
            for k, v in loss_dict.items():
                epoch_metrics.setdefault(k, []).append(v)

            if math.isnan(total_loss.item()):
                print(f"  ⚠️ NaN loss at step {self.global_step}!")
                break

        result = {'train_loss': sum(epoch_losses) / max(len(epoch_losses), 1)}
        for k, v in epoch_metrics.items():
            result[k] = sum(v) / len(v)
        return result

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        val_losses = []
        use_analytical = (self.dataset_type == 'analytical')

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
            if use_analytical:
                physics_loss = self.model.compute_physics_loss()
                total_loss, loss_dict = self._compute_analytical_loss(
                    outputs, batch, physics_loss,
                )
            else:
                total_loss, loss_dict = self.criterion(outputs['bmode'], bmode_gt)
            val_losses.append(total_loss.item())

        result = {'val_loss': sum(val_losses) / max(len(val_losses), 1)}

        # Extra validation diagnostics
        energy_hist = getattr(self.model.propagator, 'energy_history', [])
        if len(energy_hist) >= 2 and energy_hist[0] > 0:
            result['energy_ratio'] = energy_hist[-1] / energy_hist[0]

        p_hist = getattr(self.model.propagator, 'pressure_history', None)
        if p_hist is not None and len(p_hist) > 0:
            p_last = p_hist[-1]
            result['p_max'] = p_last.max().item()
            result['p_min'] = p_last.min().item()

        print(f"\n  📊 Validation: loss={result['val_loss']:.6e}", end="")
        if 'energy_ratio' in result:
            print(f" | E_ratio={result['energy_ratio']:.4f}", end="")
        if 'p_max' in result:
            print(f" | p_max={result['p_max']:.4e} | p_min={result['p_min']:.4e}", end="")
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

        last_path = os.path.join(self.checkpoint_dir, 'last.pt')
        torch.save(ckpt, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(ckpt, best_path)
            print(f"  💾 Best checkpoint saved (loss={loss:.6e})")

        epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch:04d}.pt')
        torch.save(ckpt, epoch_path)
        self.best_checkpoints.append((loss, epoch_path))
        self.best_checkpoints.sort(key=lambda x: x[0])

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
    parser = argparse.ArgumentParser(description="Train DPC-GNN-Acoustic v3")
    parser.add_argument('--config', type=str, default='configs/default_2d.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dataset', type=str, default='kwave',
                        choices=['kwave', 'analytical'],
                        help='Dataset type: kwave (default) or analytical (exact solutions)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config.setdefault('training', {})['max_epochs'] = args.epochs
    if args.lr:
        config.setdefault('training', {})['lr'] = args.lr

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"DPC-GNN-Acoustic v3 — Training")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    trainer = TrainerV3(config, device=device, resume_from=args.resume,
                        dataset_type=args.dataset)
    trainer.train()


if __name__ == '__main__':
    main()
