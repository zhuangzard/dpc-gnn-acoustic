"""
DPC-GNN-Acoustic V4: Training Script

Usage:
    python train_v4.py --config configs/v4_default.yaml
    python train_v4.py --config configs/v4_default.yaml --resume checkpoints_v4/best.pt
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from src.models.dpc_gnn_acoustic_v4 import DPCGNNAcousticV4
from src.losses.combined_loss_v4 import DPCGNNAcousticLossV4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class KWaveDataset(Dataset):
    """
    k-Wave ground truth dataset.
    Each sample: ct_slice.npy (256, 256) + bmode_gt.npy (H, W) or (128, n_samples)
    
    Directory structure:
        data_dir/
            sample_000/
                ct_slice.npy
                bmode_gt.npy
            sample_001/
                ...
    
    If only a single pair exists at data_dir level, that's fine too.
    """

    def __init__(self, data_dir: str, grid_resolution: int = 256):
        self.data_dir = Path(data_dir)
        self.grid_resolution = grid_resolution
        self.samples = []

        # Check for subdirectory structure
        subdirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if subdirs:
            for d in subdirs:
                ct_path = d / 'ct_slice.npy'
                bmode_path = d / 'bmode_gt.npy'
                if ct_path.exists() and bmode_path.exists():
                    self.samples.append((str(ct_path), str(bmode_path)))
        else:
            # Single sample at root level
            ct_path = self.data_dir / 'ct_slice.npy'
            bmode_path = self.data_dir / 'bmode_gt.npy'
            if ct_path.exists() and bmode_path.exists():
                self.samples.append((str(ct_path), str(bmode_path)))

        if len(self.samples) == 0:
            print(f"WARNING: No samples found in {data_dir}. "
                  f"Expected ct_slice.npy + bmode_gt.npy pairs.")

    def __len__(self):
        return max(len(self.samples), 1)  # At least 1 for dummy mode

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            # Dummy data for testing without real data
            ct = torch.rand(1, self.grid_resolution, self.grid_resolution)
            bmode = torch.rand(1, 128, 128)
            return ct, bmode

        ct_path, bmode_path = self.samples[idx % len(self.samples)]

        # Load CT slice
        ct = np.load(ct_path).astype(np.float32)
        if ct.ndim == 2:
            ct = ct[np.newaxis, :, :]  # [1, H, W]

        # Normalise CT to [0, 1] using global range (handles uniform phantoms)
        # CT data contains speed-of-sound values (~20-1700 m/s range)
        # Use fixed global range instead of per-sample min/max
        ct_global_min, ct_global_max = 0.0, 400.0  # covers k-Wave GT data range
        ct = (ct - ct_global_min) / (ct_global_max - ct_global_min)
        ct = np.clip(ct, 0.0, 1.0)

        # Resize if needed
        ct = torch.from_numpy(ct)
        if ct.shape[-2:] != (self.grid_resolution, self.grid_resolution):
            ct = F.interpolate(ct.unsqueeze(0), size=(self.grid_resolution, self.grid_resolution),
                               mode='bilinear', align_corners=False).squeeze(0)

        # Load B-mode GT
        bmode = np.load(bmode_path).astype(np.float32)
        if bmode.ndim == 2:
            bmode = bmode[np.newaxis, :, :]  # [1, H, W]

        # Normalise B-mode to [0, 1]
        bm_min, bm_max = bmode.min(), bmode.max()
        if bm_max - bm_min > 1e-8:
            bmode = (bmode - bm_min) / (bm_max - bm_min)
        else:
            bmode = np.zeros_like(bmode)

        bmode = torch.from_numpy(bmode)
        # Resize to (128, 128)
        if bmode.shape[-2:] != (128, 128):
            bmode = F.interpolate(bmode.unsqueeze(0), size=(128, 128),
                                   mode='bilinear', align_corners=False).squeeze(0)

        return ct, bmode


# ---------------------------------------------------------------------------
# Warmup + CosineAnnealing Scheduler
# ---------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(config: dict, resume_path: str = None):
    # --- Config ---
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    epochs = train_cfg.get('epochs', 200)
    lr = train_cfg.get('lr', 1e-4)
    batch_size = train_cfg.get('batch_size', 1)
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    patience = train_cfg.get('patience', 30)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    use_amp = train_cfg.get('use_amp', False)
    ckpt_dir = Path(train_cfg.get('checkpoint_dir', 'checkpoints_v4'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Data ---
    dataset = KWaveDataset(
        data_dir=data_cfg.get('data_dir', 'data/kwave_gt'),
        grid_resolution=data_cfg.get('grid_resolution', 256),
    )
    print(f"Dataset: {len(dataset)} samples")

    # Split train/val (80/20)
    n_total = len(dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # --- Model ---
    model = DPCGNNAcousticV4(config).to(device)
    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts}")

    # --- Loss ---
    loss_type = config.get('training', {}).get('loss_type', 'l1_ssim')
    loss_fn = DPCGNNAcousticLossV4(loss_type=loss_type).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=train_cfg.get('weight_decay', 1e-5)
    )

    # --- Scheduler ---
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)

    # --- AMP ---
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # --- Training ---
    print(f"\n{'='*80}")
    print(f"DPC-GNN-Acoustic V4 Training")
    print(f"Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"Warmup: {warmup_epochs} epochs, Patience: {patience}")
    print(f"AMP: {use_amp}, Grad clip: {grad_clip}")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        # --- Train ---
        model.train()
        train_metrics_accum = {}
        n_train_batches = 0

        for ct, gt_bmode in train_loader:
            ct = ct.to(device)
            gt_bmode = gt_bmode.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(ct)
                    loss, metrics = loss_fn(outputs['bmode'], gt_bmode, outputs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(ct)
                loss, metrics = loss_fn(outputs['bmode'], gt_bmode, outputs)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                train_metrics_accum[k] = train_metrics_accum.get(k, 0.0) + v
            n_train_batches += 1

        # Average training metrics
        train_metrics = {k: v / max(1, n_train_batches) for k, v in train_metrics_accum.items()}

        # --- Validation ---
        model.eval()
        val_metrics_accum = {}
        n_val_batches = 0

        with torch.no_grad():
            for ct, gt_bmode in val_loader:
                ct = ct.to(device)
                gt_bmode = gt_bmode.to(device)

                outputs = model(ct)
                loss, metrics = loss_fn(outputs['bmode'], gt_bmode, outputs)

                for k, v in metrics.items():
                    val_metrics_accum[k] = val_metrics_accum.get(k, 0.0) + v
                n_val_batches += 1

        val_metrics = {k: v / max(1, n_val_batches) for k, v in val_metrics_accum.items()}

        # --- Scheduler step ---
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()[0]

        # --- Logging ---
        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Loss: {train_metrics.get('total_loss', 0):.4f} | "
            f"L1: {train_metrics.get('l1', 0):.4f} | "
            f"SSIM: {train_metrics.get('ssim', 0):.4f} | "
            f"Val Loss: {val_metrics.get('total_loss', 0):.4f} | "
            f"c_mean: {train_metrics.get('c_mean', 0):.1f} | "
            f"c_std: {train_metrics.get('c_std', 0):.1f} | "
            f"α_mean: {train_metrics.get('alpha_mean', 0):.2f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- Checkpointing ---
        val_loss = val_metrics.get('total_loss', float('inf'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'metrics': val_metrics,
            }, ckpt_dir / 'best.pt')
            print(f"  → Saved best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'metrics': val_metrics,
            }, ckpt_dir / f'epoch_{epoch:03d}.pt')
            print(f"  → Saved checkpoint epoch_{epoch:03d}.pt")

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {ckpt_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='DPC-GNN-Acoustic V4 Training')
    parser.add_argument('--config', type=str, default='configs/v4_default.yaml',
                        help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config: {args.config}")
    print(f"Config contents: {config}")

    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()
