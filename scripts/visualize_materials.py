#!/usr/bin/env python3
"""
DPC-GNN-Acoustic V4: Material Field Visualization

Generates publication-quality visualization of learned material fields:
  c_table (physics prior) | c_residual (GNN correction) | c_total | α (attenuation) | σ (reflectivity)

Usage:
    python scripts/visualize_materials.py --checkpoint checkpoints_v4/best.pt
    python scripts/visualize_materials.py --checkpoint checkpoints_v4/best.pt --sample_idx 3
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dpc_gnn_acoustic_v4 import DPCGNNAcousticV4, hu_to_speed_of_sound


# ---------------------------------------------------------------------------
# Load sample
# ---------------------------------------------------------------------------
def load_sample(data_dir: str, sample_idx: int = 0, grid_resolution: int = 256):
    """Load a single CT sample for visualization."""
    data_path = Path(data_dir)
    subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if subdirs and sample_idx < len(subdirs):
        ct_path = subdirs[sample_idx] / 'ct_slice.npy'
        name = subdirs[sample_idx].name
    else:
        ct_path = data_path / 'ct_slice.npy'
        name = 'single'

    if not ct_path.exists():
        print(f"CT file not found: {ct_path}")
        print("Using random input for demo...")
        ct_raw = np.random.rand(grid_resolution, grid_resolution).astype(np.float32) * 400
        name = 'random_demo'
    else:
        ct_raw = np.load(str(ct_path)).astype(np.float32)

    if ct_raw.ndim == 2:
        ct_raw = ct_raw[np.newaxis, :, :]

    ct_norm = (ct_raw - 0.0) / 400.0
    ct_norm = np.clip(ct_norm, 0.0, 1.0)
    ct_tensor = torch.from_numpy(ct_norm).unsqueeze(0)  # [1,1,H,W]

    if ct_tensor.shape[-2:] != (grid_resolution, grid_resolution):
        ct_tensor = F.interpolate(ct_tensor, size=(grid_resolution, grid_resolution),
                                   mode='bilinear', align_corners=False)

    return ct_tensor, ct_raw[0], name


# ---------------------------------------------------------------------------
# Extract material fields with decomposition
# ---------------------------------------------------------------------------
def extract_material_fields(model, ct_tensor, device):
    """
    Run encoder and extract material fields with physics prior decomposition.
    Returns dict of numpy arrays.
    """
    model.eval()
    ct = ct_tensor.to(device)

    with torch.no_grad():
        # Get c, alpha, sigma from encoder
        c_total, alpha, sigma = model.encoder(ct)

        # Extract c_table (physics prior) and c_residual
        c_min = model.encoder.c_min
        c_max = model.encoder.c_max
        c_table = hu_to_speed_of_sound(ct, c_min, c_max)

        # c_residual = c_total - c_table (the GNN's learned correction)
        c_residual = c_total - c_table

    fields = {
        'ct_input': ct[0, 0].cpu().numpy(),
        'c_table': c_table[0, 0].cpu().numpy(),
        'c_residual': c_residual[0, 0].cpu().numpy(),
        'c_total': c_total[0, 0].cpu().numpy(),
        'alpha': alpha[0, 0].cpu().numpy(),
        'sigma': sigma[0, 0].cpu().numpy(),
    }

    # Statistics
    stats = {
        'c_total_mean': c_total.mean().item(),
        'c_total_std': c_total.std().item(),
        'c_total_min': c_total.min().item(),
        'c_total_max': c_total.max().item(),
        'c_residual_mean': c_residual.mean().item(),
        'c_residual_std': c_residual.std().item(),
        'c_residual_min': c_residual.min().item(),
        'c_residual_max': c_residual.max().item(),
        'alpha_mean': alpha.mean().item(),
        'alpha_max': alpha.max().item(),
        'sigma_mean': sigma.mean().item(),
        'sigma_std': sigma.std().item(),
    }

    return fields, stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def add_colorbar(ax, im, label=''):
    """Add a properly sized colorbar to an axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label, fontsize=8)
    return cbar


def visualize_materials(fields, stats, sample_name, save_path, epoch=None):
    """
    Generate 2-row material field visualization.
    Row 1: CT Input | c_table (prior) | c_residual (correction) | c_total
    Row 2: α (attenuation) | σ (reflectivity) | stats text
    """
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.35)

    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""

    # --- Row 1: Speed of sound decomposition ---

    # CT Input
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(fields['ct_input'], cmap='gray', aspect='equal')
    ax.set_title('CT Input (normalized)', fontsize=11)
    add_colorbar(ax, im)

    # c_table (physics prior)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(fields['c_table'], cmap='viridis', aspect='equal',
                    vmin=1400, vmax=1700)
    ax.set_title(r'$c_{table}$ (physics prior)', fontsize=11, color='darkgreen')
    add_colorbar(ax, im, label='m/s')

    # c_residual (GNN correction)
    ax = fig.add_subplot(gs[0, 2])
    abs_max = max(abs(fields['c_residual'].min()), abs(fields['c_residual'].max()), 1.0)
    im = ax.imshow(fields['c_residual'], cmap='coolwarm', aspect='equal',
                    vmin=-abs_max, vmax=abs_max)
    ax.set_title(r'$c_{residual}$ (GNN correction)', fontsize=11, color='darkred')
    add_colorbar(ax, im, label='m/s')

    # c_total
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(fields['c_total'], cmap='viridis', aspect='equal',
                    vmin=1400, vmax=1700)
    ax.set_title(r'$c_{total} = c_{table} + c_{res}$', fontsize=11, color='darkblue')
    add_colorbar(ax, im, label='m/s')

    # --- Row 2: Attenuation, Reflectivity, Stats ---

    # Alpha (attenuation)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(fields['alpha'], cmap='magma', aspect='equal')
    ax.set_title(r'$\alpha$ (attenuation)', fontsize=11)
    add_colorbar(ax, im, label='Np/m')

    # Sigma (reflectivity)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(fields['sigma'], cmap='plasma', aspect='equal', vmin=0, vmax=1)
    ax.set_title(r'$\sigma$ (reflectivity)', fontsize=11)
    add_colorbar(ax, im)

    # Histogram of c_residual
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(fields['c_residual'].flatten(), bins=80, color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(r'$c_{residual}$ distribution', fontsize=11)
    ax.set_xlabel('m/s')
    ax.set_ylabel('Count')

    # Statistics text box
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    stats_text = (
        f"Material Field Statistics{epoch_str}\n"
        f"{'='*35}\n\n"
        f"Speed of Sound (c):\n"
        f"  Total:    {stats['c_total_mean']:.1f} ± {stats['c_total_std']:.1f} m/s\n"
        f"  Range:    [{stats['c_total_min']:.1f}, {stats['c_total_max']:.1f}] m/s\n"
        f"  Residual: {stats['c_residual_mean']:.2f} ± {stats['c_residual_std']:.2f} m/s\n"
        f"  Res range:[{stats['c_residual_min']:.1f}, {stats['c_residual_max']:.1f}] m/s\n\n"
        f"Attenuation (α):\n"
        f"  Mean:     {stats['alpha_mean']:.2f} Np/m\n"
        f"  Max:      {stats['alpha_max']:.2f} Np/m\n\n"
        f"Reflectivity (σ):\n"
        f"  Mean:     {stats['sigma_mean']:.4f}\n"
        f"  Std:      {stats['sigma_std']:.4f}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'DPC-GNN-Acoustic V4 — Material Fields — {sample_name}{epoch_str}',
                 fontsize=14, fontweight='bold', y=0.98)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='DPC-GNN V4 Material Field Visualization')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v4/best.pt',
                        help='Checkpoint path')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML (overrides checkpoint config)')
    parser.add_argument('--data_dir', type=str, default='data/kwave_gt',
                        help='k-Wave GT data directory')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--output_dir', type=str, default='figures/materials',
                        help='Output directory for figures')
    parser.add_argument('--multi_checkpoint', type=str, nargs='+', default=None,
                        help='Multiple checkpoints for training progression')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load sample ---
    ct_tensor, ct_raw, sample_name = load_sample(
        args.data_dir, args.sample_idx, grid_resolution=256)
    print(f"Sample: {sample_name}, CT shape: {ct_raw.shape}")

    checkpoints = args.multi_checkpoint or [args.checkpoint]

    for ckpt_path in checkpoints:
        print(f"\nProcessing: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

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

        epoch = ckpt.get('epoch', Path(ckpt_path).stem)
        fields, stats = extract_material_fields(model, ct_tensor, device)

        save_path = output_dir / f'materials_{sample_name}_ep{epoch}.png'
        visualize_materials(fields, stats, sample_name, save_path, epoch=epoch)

        # Print stats
        print(f"  c_total: {stats['c_total_mean']:.1f} ± {stats['c_total_std']:.1f} m/s "
              f"[{stats['c_total_min']:.1f}, {stats['c_total_max']:.1f}]")
        print(f"  c_residual: {stats['c_residual_mean']:.2f} ± {stats['c_residual_std']:.2f} m/s")
        print(f"  alpha: mean={stats['alpha_mean']:.2f}, max={stats['alpha_max']:.2f} Np/m")
        print(f"  sigma: mean={stats['sigma_mean']:.4f} ± {stats['sigma_std']:.4f}")

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == '__main__':
    main()
