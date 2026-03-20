#!/usr/bin/env python3
"""
DPC-GNN-Acoustic V4: B-mode Visualization

Generates comparison figures:
    CT Input | k-Wave GT | V4 Predicted | Error Map

Features:
  - Loads checkpoint and runs inference on test samples
  - SSIM/L1 annotated per sample
  - 300 DPI publication-quality PNG output
  - Optional: compare across epochs (10/50/100)

Usage:
    python scripts/visualize_bmode.py --checkpoint checkpoints_v4/best.pt
    python scripts/visualize_bmode.py --checkpoint checkpoints_v4/epoch_049.pt --n_samples 10
    python scripts/visualize_bmode.py --multi_epoch checkpoints_v4/epoch_009.pt checkpoints_v4/epoch_049.pt checkpoints_v4/best.pt
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dpc_gnn_acoustic_v4 import DPCGNNAcousticV4
from src.losses.combined_loss_v4 import SSIM


# ---------------------------------------------------------------------------
# Dataset (minimal, for inference only)
# ---------------------------------------------------------------------------
class KWaveTestDataset:
    """Load k-Wave GT samples for visualization."""

    def __init__(self, data_dir: str, grid_resolution: int = 256):
        self.data_dir = Path(data_dir)
        self.grid_resolution = grid_resolution
        self.samples = []

        subdirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if subdirs:
            for d in subdirs:
                ct_path = d / 'ct_slice.npy'
                bmode_path = d / 'bmode_gt.npy'
                if ct_path.exists() and bmode_path.exists():
                    self.samples.append((str(ct_path), str(bmode_path), d.name))
        else:
            ct_path = self.data_dir / 'ct_slice.npy'
            bmode_path = self.data_dir / 'bmode_gt.npy'
            if ct_path.exists() and bmode_path.exists():
                self.samples.append((str(ct_path), str(bmode_path), 'single'))

    def __len__(self):
        return len(self.samples)

    def load_sample(self, idx: int):
        """Returns (ct_tensor [1,1,256,256], gt_bmode [1,1,128,128], name, ct_raw, gt_raw)"""
        ct_path, bmode_path, name = self.samples[idx]

        ct_raw = np.load(ct_path).astype(np.float32)
        if ct_raw.ndim == 2:
            ct_raw = ct_raw[np.newaxis, :, :]

        # Normalize CT
        ct = (ct_raw - 0.0) / 400.0
        ct = np.clip(ct, 0.0, 1.0)
        ct_tensor = torch.from_numpy(ct).unsqueeze(0)  # [1,1,H,W]
        if ct_tensor.shape[-2:] != (self.grid_resolution, self.grid_resolution):
            ct_tensor = F.interpolate(ct_tensor, size=(self.grid_resolution, self.grid_resolution),
                                       mode='bilinear', align_corners=False)

        # Load GT B-mode
        gt_raw = np.load(bmode_path).astype(np.float32)
        if gt_raw.ndim == 2:
            gt_raw = gt_raw[np.newaxis, :, :]
        bm_min, bm_max = gt_raw.min(), gt_raw.max()
        if bm_max - bm_min > 1e-8:
            gt_norm = (gt_raw - bm_min) / (bm_max - bm_min)
        else:
            gt_norm = np.zeros_like(gt_raw)
        gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0)
        if gt_tensor.shape[-2:] != (128, 128):
            gt_tensor = F.interpolate(gt_tensor, size=(128, 128),
                                       mode='bilinear', align_corners=False)

        return ct_tensor, gt_tensor, name, ct_raw[0], gt_raw[0] if gt_raw.ndim == 3 else gt_raw


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------
def compute_ssim_value(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute SSIM between two [1,1,H,W] tensors."""
    ssim_fn = SSIM(window_size=11, sigma=1.5)
    with torch.no_grad():
        return ssim_fn(pred, gt).item()


def compute_l1_value(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute L1 between two tensors."""
    with torch.no_grad():
        return F.l1_loss(pred, gt).item()


def visualize_single_sample(ct_raw, gt_bmode, pred_bmode, ssim_val, l1_val,
                             sample_name, save_path, epoch_label=None):
    """
    Generate 4-panel figure: CT Input | k-Wave GT | V4 Predicted | Error Map
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    title_prefix = f"[Ep {epoch_label}] " if epoch_label else ""

    # CT Input
    ax = axes[0]
    im = ax.imshow(ct_raw, cmap='gray', aspect='auto')
    ax.set_title('CT Input (256×256)', fontsize=10)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # k-Wave GT
    ax = axes[1]
    im = ax.imshow(gt_bmode, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title('k-Wave GT B-mode', fontsize=10)
    ax.set_xlabel('x (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # V4 Predicted
    ax = axes[2]
    im = ax.imshow(pred_bmode, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'{title_prefix}V4 Predicted\nSSIM={ssim_val:.4f}  L1={l1_val:.4f}',
                 fontsize=10, color='darkblue')
    ax.set_xlabel('x (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Error Map
    ax = axes[3]
    error = np.abs(pred_bmode - gt_bmode)
    im = ax.imshow(error, cmap='hot', aspect='auto', vmin=0, vmax=0.3)
    ax.set_title(f'|Error| (max={error.max():.3f})', fontsize=10)
    ax.set_xlabel('x (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Sample: {sample_name}', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def visualize_multi_epoch(ct_raw, gt_bmode, predictions, sample_name, save_path):
    """
    Generate multi-epoch comparison:
    Row: CT | GT | Epoch1 Pred | Epoch2 Pred | ... | EpochN Pred
    """
    n_epochs = len(predictions)
    fig, axes = plt.subplots(1, 2 + n_epochs, figsize=(4 * (2 + n_epochs), 4))

    # CT Input
    axes[0].imshow(ct_raw, cmap='gray', aspect='auto')
    axes[0].set_title('CT Input', fontsize=10)

    # GT
    axes[1].imshow(gt_bmode, cmap='gray', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('k-Wave GT', fontsize=10)

    for i, (epoch_label, pred, ssim_val) in enumerate(predictions):
        ax = axes[2 + i]
        ax.imshow(pred, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Epoch {epoch_label}\nSSIM={ssim_val:.4f}', fontsize=10, color='darkblue')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Training Progression — {sample_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='DPC-GNN V4 B-mode Visualization')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v4/best.pt',
                        help='Single checkpoint path')
    parser.add_argument('--multi_epoch', type=str, nargs='+', default=None,
                        help='Multiple checkpoint paths for epoch comparison')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML (overrides checkpoint config)')
    parser.add_argument('--data_dir', type=str, default='data/kwave_gt',
                        help='k-Wave GT data directory')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of test samples to visualize')
    parser.add_argument('--output_dir', type=str, default='figures/bmode',
                        help='Output directory for figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    dataset = KWaveTestDataset(args.data_dir)
    n_samples = min(args.n_samples, len(dataset))
    print(f"Dataset: {len(dataset)} samples, visualizing {n_samples}")

    if args.multi_epoch:
        # --- Multi-epoch comparison mode ---
        print(f"\nMulti-epoch comparison: {len(args.multi_epoch)} checkpoints")

        for sample_idx in range(n_samples):
            ct_tensor, gt_tensor, name, ct_raw, gt_raw = dataset.load_sample(sample_idx)
            gt_bmode_np = gt_tensor[0, 0].numpy()

            predictions = []
            for ckpt_path in args.multi_epoch:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                config = ckpt.get('config', None)
                if config is None:
                    with open('configs/v4_default.yaml', 'r') as f:
                        config = yaml.safe_load(f)

                model = DPCGNNAcousticV4(config).to(device)
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                model.eval()

                with torch.no_grad():
                    outputs = model(ct_tensor.to(device))
                pred_bmode = outputs['bmode'].cpu()
                ssim_val = compute_ssim_value(pred_bmode, gt_tensor)
                pred_np = pred_bmode[0, 0].numpy()

                epoch_label = ckpt.get('epoch', Path(ckpt_path).stem)
                predictions.append((epoch_label, pred_np, ssim_val))

            save_path = output_dir / f'multi_epoch_{name}.png'
            visualize_multi_epoch(ct_raw, gt_bmode_np, predictions, name, save_path)

    else:
        # --- Single checkpoint mode ---
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        elif 'config' in ckpt:
            config = ckpt['config']
        else:
            with open('configs/v4_default.yaml', 'r') as f:
                config = yaml.safe_load(f)

        model = DPCGNNAcousticV4(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()

        epoch = ckpt.get('epoch', '?')
        print(f"Checkpoint epoch: {epoch}")

        ssim_values = []
        l1_values = []

        for sample_idx in range(n_samples):
            ct_tensor, gt_tensor, name, ct_raw, gt_raw = dataset.load_sample(sample_idx)

            with torch.no_grad():
                outputs = model(ct_tensor.to(device))
            pred_bmode = outputs['bmode'].cpu()

            ssim_val = compute_ssim_value(pred_bmode, gt_tensor)
            l1_val = compute_l1_value(pred_bmode, gt_tensor)
            ssim_values.append(ssim_val)
            l1_values.append(l1_val)

            pred_np = pred_bmode[0, 0].numpy()
            gt_np = gt_tensor[0, 0].numpy()

            save_path = output_dir / f'bmode_{name}_ep{epoch}.png'
            visualize_single_sample(
                ct_raw, gt_np, pred_np, ssim_val, l1_val,
                name, save_path, epoch_label=epoch,
            )

        # --- Summary ---
        print(f"\n{'='*50}")
        print(f"B-mode Visualization Summary (Epoch {epoch})")
        print(f"{'='*50}")
        for i, (ssim_v, l1_v) in enumerate(zip(ssim_values, l1_values)):
            name = dataset.samples[i][2]
            print(f"  {name}: SSIM={ssim_v:.4f}, L1={l1_v:.4f}")
        print(f"  Mean:  SSIM={np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
        print(f"         L1={np.mean(l1_values):.4f} ± {np.std(l1_values):.4f}")
        print(f"\nFigures saved to: {output_dir}/")


if __name__ == '__main__':
    main()
