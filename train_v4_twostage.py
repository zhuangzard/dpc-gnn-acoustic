#!/usr/bin/env python3
"""
DPC-GNN-Acoustic V4: Two-Stage Training

Stage 1 (Supervised): Train GNN to predict c(x,y) and α(x,y) from CT
  - Target: c_gt derived from ct_slice (known linear mapping + spatial context)
  - α_gt estimated from c_gt via empirical tissue models
  - Loss: MSE(c_pred, c_gt) + MSE(α_pred, α_gt)
  - NO wave propagation — fast, ~1s/epoch

Stage 2 (End-to-End): Fine-tune through physics propagator
  - Resume from Stage 1 best checkpoint
  - Loss: L1 + (1-SSIM) on B-mode
  - Full wave propagation — ~44s/epoch
"""

import os
import sys
import time
import math
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.dpc_gnn_acoustic_v4 import DPCGNNAcousticV4, GNNEncoder, ct_to_speed_of_sound


# ============================================================================
# Stage 1 Dataset: CT → (c_gt, α_gt) pairs
# ============================================================================
class CTtoPropertyDataset(Dataset):
    """Load CT slices and derive GT material properties."""

    def __init__(self, data_dir: str, grid_resolution: int = 256,
                 c_min: float = 1400.0, c_max: float = 2000.0):
        self.data_dir = data_dir
        self.grid_resolution = grid_resolution
        self.c_min = c_min
        self.c_max = c_max
        self.samples = []

        data_path = os.path.join(data_dir)
        for d in sorted(os.listdir(data_path)):
            sample_dir = os.path.join(data_path, d)
            ct_path = os.path.join(sample_dir, 'ct_slice.npy')
            if os.path.isdir(sample_dir) and os.path.exists(ct_path):
                self.samples.append(ct_path)

        print(f"Stage 1 dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path = self.samples[idx]
        ct_raw = np.load(ct_path).astype(np.float32)

        # Derive c_gt from CT: reverse the c_to_hu mapping
        # ct = (c - 1400) / 300 * 400  →  c = ct * 0.75 + 1400
        c_gt = ct_raw * 0.75 + 1400.0
        c_gt = np.clip(c_gt, self.c_min, self.c_max)

        # Derive α_gt from c using empirical tissue model
        # Higher c → denser tissue → higher attenuation
        # Typical: α ∈ [1, 10] Np/m at 2 MHz
        # Linear model: α = 1 + (c - 1400) / (2000 - 1400) * 9
        alpha_gt = 1.0 + (c_gt - 1400.0) / 600.0 * 9.0
        alpha_gt = np.clip(alpha_gt, 1.0, 10.0)

        # Normalize CT for model input (same as train_v4.py)
        ct_norm = np.clip(ct_raw / 800.0, 0.0, 1.0)

        # Add channel dimension
        ct_tensor = torch.from_numpy(ct_norm[np.newaxis]).float()
        c_tensor = torch.from_numpy(c_gt[np.newaxis]).float()
        alpha_tensor = torch.from_numpy(alpha_gt[np.newaxis]).float()

        return ct_tensor, c_tensor, alpha_tensor


# ============================================================================
# Stage 1: Supervised pre-training
# ============================================================================
def train_stage1(config, args):
    """Pre-train GNN encoder: CT → (c, α) with direct supervision."""
    print("=" * 80)
    print("STAGE 1: Supervised Pre-training (CT → c, α)")
    print("=" * 80)

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = CTtoPropertyDataset(
        data_dir=data_cfg.get('data_dir', 'data/kwave_gt'),
        grid_resolution=data_cfg.get('grid_resolution', 256),
        c_min=model_cfg.get('c_min', 1400.0),
        c_max=model_cfg.get('c_max', 2000.0),
    )

    val_ratio = data_cfg.get('val_ratio', 0.15)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    batch_size = 32  # Stage 1 is fast, can use large batch
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    # Model — just the encoder
    encoder = GNNEncoder(
        hidden_dim=model_cfg.get('hidden_dim', 96),
        n_mp_layers=model_cfg.get('n_mp_layers', 5),
        k_local=model_cfg.get('k_local', 8),
        c_min=model_cfg.get('c_min', 1400.0),
        c_max=model_cfg.get('c_max', 2000.0),
        use_residual=model_cfg.get('use_residual', True),
        antisymmetric=model_cfg.get('antisymmetric', True),
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {n_params:,}")

    # Optimizer — higher LR for supervised (no gradient vanishing through propagator)
    lr = args.stage1_lr if args.stage1_lr else 1e-3
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)

    epochs = args.stage1_epochs if args.stage1_epochs else 100
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    ckpt_dir = 'checkpoints_v4_stage1'
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"Train: {n_train}, Val: {n_val}")
    print(f"{'=' * 80}\n")

    for epoch in range(epochs):
        t_start = time.time()

        # Train
        encoder.train()
        train_loss_sum = 0.0
        n_batches = 0

        for ct, c_gt, alpha_gt in train_loader:
            ct, c_gt, alpha_gt = ct.to(device), c_gt.to(device), alpha_gt.to(device)
            optimizer.zero_grad()

            c_pred, alpha_pred = encoder(ct)

            # MSE loss on c and α
            loss_c = F.mse_loss(c_pred, c_gt)
            loss_alpha = F.mse_loss(alpha_pred, alpha_gt)
            loss = loss_c + 0.1 * loss_alpha  # α less important

            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(1, n_batches)

        # Validate
        encoder.eval()
        val_loss_sum = 0.0
        val_c_mae_sum = 0.0
        val_alpha_mae_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for ct, c_gt, alpha_gt in val_loader:
                ct, c_gt, alpha_gt = ct.to(device), c_gt.to(device), alpha_gt.to(device)
                c_pred, alpha_pred = encoder(ct)
                loss_c = F.mse_loss(c_pred, c_gt)
                loss_alpha = F.mse_loss(alpha_pred, alpha_gt)
                val_loss_sum += (loss_c + 0.1 * loss_alpha).item()
                val_c_mae_sum += F.l1_loss(c_pred, c_gt).item()
                val_alpha_mae_sum += F.l1_loss(alpha_pred, alpha_gt).item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(1, n_val_batches)
        val_c_mae = val_c_mae_sum / max(1, n_val_batches)
        val_alpha_mae = val_alpha_mae_sum / max(1, n_val_batches)

        elapsed = time.time() - t_start
        print(
            f"S1 Ep {epoch:03d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"c_MAE: {val_c_mae:.2f} m/s | "
            f"α_MAE: {val_alpha_mae:.3f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_c_mae': val_c_mae,
            }, os.path.join(ckpt_dir, 'best_encoder.pt'))
            print(f"  → Saved best encoder (val_loss={val_loss:.6f}, c_MAE={val_c_mae:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nStage 1 complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Best encoder: {ckpt_dir}/best_encoder.pt")
    return os.path.join(ckpt_dir, 'best_encoder.pt')


# ============================================================================
# Stage 2: End-to-end fine-tuning (reuse train_v4.py logic)
# ============================================================================
def train_stage2(config, encoder_ckpt_path, args):
    """Fine-tune full model end-to-end, starting from pre-trained encoder."""
    print("\n" + "=" * 80)
    print("STAGE 2: End-to-End Fine-tuning (B-mode loss)")
    print(f"Loading pre-trained encoder from: {encoder_ckpt_path}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build full model
    model = DPCGNNAcousticV4(config).to(device)

    # Load pre-trained encoder weights
    ckpt = torch.load(encoder_ckpt_path, map_location=device)
    encoder_sd = ckpt['encoder_state_dict']
    model.encoder.load_state_dict(encoder_sd)
    print(f"  Loaded encoder from epoch {ckpt['epoch']}, c_MAE={ckpt.get('val_c_mae', '?')}")

    # Now run standard training (import from train_v4)
    # Save model as a checkpoint that train_v4 can resume from
    stage2_ckpt = 'checkpoints_v4/stage2_init.pt'
    os.makedirs('checkpoints_v4', exist_ok=True)
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'best_val_loss': float('inf'),
    }, stage2_ckpt)
    print(f"  Saved Stage 2 init checkpoint: {stage2_ckpt}")
    print(f"\n  Now run: python3 train_v4.py --config {args.config} --resume {stage2_ckpt}")
    print(f"  This will fine-tune the full model end-to-end with B-mode loss.")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage Training for DPC-GNN-Acoustic V4')
    parser.add_argument('--config', type=str, default='configs/v4_default.yaml')
    parser.add_argument('--stage', type=int, default=0,
                        help='0=both, 1=stage1 only, 2=stage2 only')
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Path to pre-trained encoder (for stage 2 only)')
    parser.add_argument('--stage1_lr', type=float, default=1e-3)
    parser.add_argument('--stage1_epochs', type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stage == 0 or args.stage == 1:
        encoder_path = train_stage1(config, args)
    else:
        encoder_path = args.encoder_ckpt

    if args.stage == 0 or args.stage == 2:
        if encoder_path is None:
            encoder_path = 'checkpoints_v4_stage1/best_encoder.pt'
        train_stage2(config, encoder_path, args)
