#!/usr/bin/env python3
"""
Pix2Pix Baseline (L0): CT → B-mode via adversarial training

Architecture:
  - Generator: U-Net (6 down + 6 up, skip connections)
  - Discriminator: PatchGAN (70×70 receptive field)
  - Input: CT 256×256  → Output: B-mode 128×128
  - Loss: cGAN (PatchGAN) + L1

This is the L0 "pure learning" baseline with NO physics inductive bias.
Uses the same k-Wave GT data as V4 for fair comparison.

Usage:
    python scripts/train_pix2pix_baseline.py --data_dir data/kwave_gt
    python scripts/train_pix2pix_baseline.py --data_dir data/kwave_gt --epochs 300 --lr_g 2e-4
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

# Reuse SSIM from V4 loss
from src.losses.combined_loss_v4 import SSIM


# ---------------------------------------------------------------------------
# Dataset (same as V4)
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
# U-Net Generator (Pix2Pix style)
# ---------------------------------------------------------------------------
class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)


class Pix2PixGenerator(nn.Module):
    """
    U-Net Generator: 256×256 (1ch) → 128×128 (1ch)
    Encoder: 256→128→64→32→16→8→4
    Decoder: 4→8→16→32→64→128 (with skip connections)
    Final: adaptive pool to 128×128
    """

    def __init__(self):
        super().__init__()
        # Encoder (downsampling)
        self.down1 = UNetDown(1, 64, normalize=False)   # 256→128
        self.down2 = UNetDown(64, 128)                   # 128→64
        self.down3 = UNetDown(128, 256)                  # 64→32
        self.down4 = UNetDown(256, 512)                  # 32→16
        self.down5 = UNetDown(512, 512)                  # 16→8
        self.down6 = UNetDown(512, 512, normalize=False) # 8→4

        # Decoder (upsampling with skip connections)
        self.up1 = UNetUp(512, 512, dropout=True)        # 4→8, cat→1024
        self.up2 = UNetUp(1024, 512)                     # 8→16, cat→1024
        self.up3 = UNetUp(1024, 256)                     # 16→32, cat→512
        self.up4 = UNetUp(512, 128)                      # 32→64, cat→256
        self.up5 = UNetUp(256, 64)                       # 64→128, cat→128

        # Final: 128→128×128 with 1 output channel
        self.final = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)    # [B,64,128,128]
        d2 = self.down2(d1)   # [B,128,64,64]
        d3 = self.down3(d2)   # [B,256,32,32]
        d4 = self.down4(d3)   # [B,512,16,16]
        d5 = self.down5(d4)   # [B,512,8,8]
        d6 = self.down6(d5)   # [B,512,4,4]

        # Decoder with skip connections
        u1 = self.up1(d6, d5)  # [B,1024,8,8]
        u2 = self.up2(u1, d4)  # [B,1024,16,16]
        u3 = self.up3(u2, d3)  # [B,512,32,32]
        u4 = self.up4(u3, d2)  # [B,256,64,64]
        u5 = self.up5(u4, d1)  # [B,128,128,128]

        out = self.final(u5)   # [B,1,128,128]
        return out


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator: operates on 128×128 patches.
    Input: concatenated (CT_downsampled, B-mode) = 2 channels.
    Output: [B, 1, N, N] patch predictions.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 128→64
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64→32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 32→16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 16→15 (stride=1)
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 15→14
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, ct_down, bmode):
        """
        Args:
            ct_down: [B, 1, 128, 128] downsampled CT
            bmode:   [B, 1, 128, 128] B-mode (real or fake)
        """
        x = torch.cat([ct_down, bmode], dim=1)  # [B, 2, 128, 128]
        return self.model(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Pix2Pix Baseline (L0)')
    parser.add_argument('--data_dir', type=str, default='data/kwave_gt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help='Weight for L1 loss (standard Pix2Pix uses 100)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_pix2pix')
    parser.add_argument('--seed', type=int, default=42)
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

    # --- Models ---
    generator = Pix2PixGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    print(f"Generator params:     {count_parameters(generator):,}")
    print(f"Discriminator params: {count_parameters(discriminator):,}")
    print(f"Total params:         {count_parameters(generator) + count_parameters(discriminator):,}")

    # --- Losses ---
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    ssim_fn = SSIM(window_size=11, sigma=1.5)

    # --- Optimizers ---
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # --- Training ---
    best_val_ssim = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Pix2Pix Baseline Training")
    print(f"Epochs: {args.epochs}, LR_G: {args.lr_g}, LR_D: {args.lr_d}")
    print(f"Lambda_L1: {args.lambda_l1}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        t_start = time.time()

        # --- Train ---
        generator.train()
        discriminator.train()
        train_g_loss = 0.0
        train_d_loss = 0.0
        train_l1 = 0.0
        train_ssim = 0.0
        n_batches = 0

        for ct, gt_bmode in train_loader:
            ct = ct.to(device)
            gt_bmode = gt_bmode.to(device)

            # Downsample CT for discriminator input
            ct_down = F.interpolate(ct, size=(128, 128), mode='bilinear', align_corners=False)

            # --- Generate fake B-mode ---
            fake_bmode = generator(ct)

            # ==================
            # Train Discriminator
            # ==================
            opt_d.zero_grad()

            # Real
            pred_real = discriminator(ct_down, gt_bmode)
            label_real = torch.ones_like(pred_real)
            loss_d_real = criterion_gan(pred_real, label_real)

            # Fake (detach generator)
            pred_fake = discriminator(ct_down, fake_bmode.detach())
            label_fake = torch.zeros_like(pred_fake)
            loss_d_fake = criterion_gan(pred_fake, label_fake)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # ==================
            # Train Generator
            # ==================
            opt_g.zero_grad()

            pred_fake_for_g = discriminator(ct_down, fake_bmode)
            loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = criterion_l1(fake_bmode, gt_bmode)
            loss_g = loss_g_gan + args.lambda_l1 * loss_g_l1
            loss_g.backward()
            opt_g.step()

            # Metrics
            with torch.no_grad():
                ssim_val = ssim_fn(fake_bmode, gt_bmode).item()

            train_g_loss += loss_g.item()
            train_d_loss += loss_d.item()
            train_l1 += loss_g_l1.item()
            train_ssim += ssim_val
            n_batches += 1

        train_g_loss /= max(1, n_batches)
        train_d_loss /= max(1, n_batches)
        train_l1 /= max(1, n_batches)
        train_ssim /= max(1, n_batches)

        # --- Validation ---
        generator.eval()
        val_l1 = 0.0
        val_ssim = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for ct, gt_bmode in val_loader:
                ct = ct.to(device)
                gt_bmode = gt_bmode.to(device)
                fake_bmode = generator(ct)
                val_l1 += F.l1_loss(fake_bmode, gt_bmode).item()
                val_ssim += ssim_fn(fake_bmode, gt_bmode).item()
                n_val_batches += 1

        val_l1 /= max(1, n_val_batches)
        val_ssim /= max(1, n_val_batches)

        elapsed = time.time() - t_start
        print(
            f"Ep {epoch:03d}/{args.epochs} | "
            f"G: {train_g_loss:.4f} | D: {train_d_loss:.4f} | "
            f"L1: {train_l1:.4f} | SSIM: {train_ssim:.4f} | "
            f"Val_L1: {val_l1:.4f} | Val_SSIM: {val_ssim:.4f} | "
            f"{elapsed:.1f}s"
        )

        # --- Checkpoint ---
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'best_val_ssim': best_val_ssim,
                'val_l1': val_l1,
                'generator_params': count_parameters(generator),
                'discriminator_params': count_parameters(discriminator),
            }, ckpt_dir / 'best.pt')
            print(f"  → Best model (SSIM={best_val_ssim:.4f})")
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_ssim': val_ssim,
            }, ckpt_dir / f'epoch_{epoch:03d}.pt')

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*70}")
    print(f"Pix2Pix Training Complete")
    print(f"Best Val SSIM: {best_val_ssim:.4f}")
    print(f"Generator params: {count_parameters(generator):,}")
    print(f"Discriminator params: {count_parameters(discriminator):,}")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
