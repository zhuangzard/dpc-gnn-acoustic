#!/usr/bin/env python3
"""
U-Net Baseline (L0): CT → B-mode via pure regression

Architecture:
  - Standard U-Net encoder-decoder with skip connections
  - Input: CT 256×256 → Output: B-mode 128×128
  - Loss: L1 + (1 - SSIM) — same as V4 for fair comparison
  - NO physics, NO GAN, NO graph — pure CNN regression

Usage:
    python scripts/train_unet_baseline.py --data_dir data/kwave_gt
    python scripts/train_unet_baseline.py --data_dir data/kwave_gt --epochs 300 --lr 1e-4
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses.combined_loss_v4 import SSIM


# ---------------------------------------------------------------------------
# Dataset (identical to V4)
# ---------------------------------------------------------------------------
class KWaveDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        subdirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if subdirs:
            for d in subdirs:
                ct = d / 'ct_slice.npy'
                bm = d / 'bmode_gt.npy'
                if ct.exists() and bm.exists():
                    self.samples.append((str(ct), str(bm)))
        else:
            ct = self.data_dir / 'ct_slice.npy'
            bm = self.data_dir / 'bmode_gt.npy'
            if ct.exists() and bm.exists():
                self.samples.append((str(ct), str(bm)))

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if not self.samples:
            return torch.rand(1, 256, 256), torch.rand(1, 128, 128)

        ct_path, bm_path = self.samples[idx % len(self.samples)]

        ct = np.load(ct_path).astype(np.float32)
        if ct.ndim == 2:
            ct = ct[np.newaxis, :, :]
        ct = (ct - 0.0) / 400.0
        ct = np.clip(ct, 0.0, 1.0)
        ct = torch.from_numpy(ct)
        if ct.shape[-2:] != (256, 256):
            ct = F.interpolate(ct.unsqueeze(0), (256, 256), mode='bilinear',
                               align_corners=False).squeeze(0)

        bm = np.load(bm_path).astype(np.float32)
        if bm.ndim == 2:
            bm = bm[np.newaxis, :, :]
        bm_min, bm_max = bm.min(), bm.max()
        if bm_max - bm_min > 1e-8:
            bm = (bm - bm_min) / (bm_max - bm_min)
        else:
            bm = np.zeros_like(bm)
        bm = torch.from_numpy(bm)
        if bm.shape[-2:] != (128, 128):
            bm = F.interpolate(bm.unsqueeze(0), (128, 128), mode='bilinear',
                               align_corners=False).squeeze(0)

        return ct, bm


# ---------------------------------------------------------------------------
# U-Net Model
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBaseline(nn.Module):
    """
    Standard U-Net for CT (256×256) → B-mode (128×128) regression.

    Encoder: 256→128→64→32→16→8
    Decoder: 8→16→32→64→128
    Final output: 128×128×1 (via adaptive pool if needed)
    """

    def __init__(self, init_features: int = 64):
        super().__init__()
        f = init_features

        # Encoder
        self.enc1 = DoubleConv(1, f)       # 256→256
        self.pool1 = nn.MaxPool2d(2)        # 256→128
        self.enc2 = DoubleConv(f, f*2)     # 128→128
        self.pool2 = nn.MaxPool2d(2)        # 128→64
        self.enc3 = DoubleConv(f*2, f*4)   # 64→64
        self.pool3 = nn.MaxPool2d(2)        # 64→32
        self.enc4 = DoubleConv(f*4, f*8)   # 32→32
        self.pool4 = nn.MaxPool2d(2)        # 32→16

        # Bottleneck
        self.bottleneck = DoubleConv(f*8, f*16)  # 16→16

        # Decoder
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)  # 16→32
        self.dec4 = DoubleConv(f*16, f*8)  # skip + up = f*16 → f*8
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)   # 32→64
        self.dec3 = DoubleConv(f*8, f*4)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)   # 64→128
        self.dec2 = DoubleConv(f*4, f*2)

        # Output head: 128×128 with 1 channel
        self.out_conv = nn.Sequential(
            nn.Conv2d(f*2, f, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 256] CT input
        Returns:
            [B, 1, 128, 128] predicted B-mode
        """
        # Encoder
        e1 = self.enc1(x)          # [B, 64, 256, 256]
        e2 = self.enc2(self.pool1(e1))  # [B, 128, 128, 128]
        e3 = self.enc3(self.pool2(e2))  # [B, 256, 64, 64]
        e4 = self.enc4(self.pool3(e3))  # [B, 512, 32, 32]

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # [B, 1024, 16, 16]

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))     # [B, 512, 32, 32]
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))    # [B, 256, 64, 64]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))    # [B, 128, 128, 128]

        # Output (skip enc1 — we want 128×128, not 256×256)
        out = self.out_conv(d2)  # [B, 1, 128, 128]
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='U-Net Baseline (L0)')
    parser.add_argument('--data_dir', type=str, default='data/kwave_gt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_unet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init_features', type=int, default=64,
                        help='Initial feature channels (64=standard, 32=lightweight)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    dataset = KWaveDataset(args.data_dir)
    n_total = len(dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {n_total} samples (train={n_train}, val={n_val})")

    # --- Model ---
    model = UNetBaseline(init_features=args.init_features).to(device)
    n_params = count_parameters(model)
    print(f"U-Net parameters: {n_params:,}")

    # --- Loss (same as V4: L1 + (1 - SSIM)) ---
    l1_fn = nn.L1Loss()
    ssim_fn = SSIM(window_size=11, sigma=1.5)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    # --- Cosine annealing scheduler ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Training ---
    best_val_ssim = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"U-Net Baseline Training (L1 + SSIM loss)")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Params: {n_params:,}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        t_start = time.time()

        # --- Train ---
        model.train()
        train_l1 = 0.0
        train_ssim = 0.0
        train_loss = 0.0
        n_batches = 0

        for ct, gt_bmode in train_loader:
            ct = ct.to(device)
            gt_bmode = gt_bmode.to(device)

            optimizer.zero_grad()
            pred = model(ct)

            l1_loss = l1_fn(pred, gt_bmode)
            ssim_val = ssim_fn(pred, gt_bmode)
            loss = l1_loss + (1.0 - ssim_val)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_l1 += l1_loss.item()
            train_ssim += ssim_val.item()
            train_loss += loss.item()
            n_batches += 1

        train_l1 /= max(1, n_batches)
        train_ssim /= max(1, n_batches)
        train_loss /= max(1, n_batches)

        # --- Validation ---
        model.eval()
        val_l1 = 0.0
        val_ssim = 0.0
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for ct, gt_bmode in val_loader:
                ct = ct.to(device)
                gt_bmode = gt_bmode.to(device)
                pred = model(ct)

                l1_loss = l1_fn(pred, gt_bmode)
                ssim_val = ssim_fn(pred, gt_bmode)
                loss = l1_loss + (1.0 - ssim_val)

                val_l1 += l1_loss.item()
                val_ssim += ssim_val.item()
                val_loss += loss.item()
                n_val_batches += 1

        val_l1 /= max(1, n_val_batches)
        val_ssim /= max(1, n_val_batches)
        val_loss /= max(1, n_val_batches)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t_start
        print(
            f"Ep {epoch:03d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | L1: {train_l1:.4f} | SSIM: {train_ssim:.4f} | "
            f"Val_Loss: {val_loss:.4f} | Val_SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # --- Checkpoint ---
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_ssim': best_val_ssim,
                'val_l1': val_l1,
                'val_loss': val_loss,
                'n_params': n_params,
            }, ckpt_dir / 'best.pt')
            print(f"  → Best model (SSIM={best_val_ssim:.4f})")
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_ssim': val_ssim,
            }, ckpt_dir / f'epoch_{epoch:03d}.pt')

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*70}")
    print(f"U-Net Training Complete")
    print(f"Best Val SSIM: {best_val_ssim:.4f}")
    print(f"Parameters:    {n_params:,}")
    print(f"Checkpoint:    {ckpt_dir / 'best.pt'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
