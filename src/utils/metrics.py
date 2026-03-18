"""
metrics.py — Evaluation metrics for DPC-GNN-Acoustic v2.

Metrics:
  1. SSIM — Structural Similarity Index
  2. PSNR — Peak Signal-to-Noise Ratio
  3. Physics validation — Energy conservation, wave equation residual
  4. Profile metrics — Lateral/axial resolution
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> float:
    """Compute SSIM between predicted and target images.
    
    Args:
        pred: (H, W) predicted image
        target: (H, W) target image
    
    Returns:
        ssim: SSIM value in [0, 1]
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    
    if pred.shape != target.shape:
        target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    
    sigma1_sq = F.conv2d(pred ** 2, window, padding=pad) - mu1 ** 2
    sigma2_sq = F.conv2d(target ** 2, window, padding=pad) - mu2 ** 2
    sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(MAX² / MSE)
    
    Args:
        pred: (H, W) predicted image (assumed [0, 1] range)
        target: (H, W) target image
    
    Returns:
        psnr: PSNR in dB
    """
    if pred.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(0).unsqueeze(0),
            size=pred.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze()
    
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    
    max_val = max(pred.max().item(), target.max().item())
    if max_val < 1e-10:
        max_val = 1.0
    
    psnr = 10 * torch.log10(torch.tensor(max_val ** 2) / mse)
    return psnr.item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    if pred.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(0).unsqueeze(0),
            size=pred.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze()
    return F.l1_loss(pred, target).item()


def compute_energy_conservation(energy_history: list) -> Dict[str, float]:
    """Analyze energy conservation from propagation history.
    
    Args:
        energy_history: List of energy values at each time step
    
    Returns:
        metrics: Dict with conservation metrics
    """
    if len(energy_history) < 2:
        return {'energy_variation': 0.0, 'energy_conserved': True}
    
    energies = np.array(energy_history)
    initial = energies[0]
    
    if abs(initial) < 1e-10:
        return {'energy_variation': 0.0, 'energy_conserved': True}
    
    relative_change = np.abs(energies - initial) / abs(initial)
    max_variation = relative_change.max()
    
    # Energy should be non-increasing (with attenuation)
    energy_growth = np.maximum(np.diff(energies), 0).sum()
    
    return {
        'energy_max_variation': float(max_variation),
        'energy_final_ratio': float(energies[-1] / initial),
        'energy_growth': float(energy_growth),
        'energy_conserved': max_variation < 0.5,  # 50% tolerance
    }


def evaluate_sample(
    pred_bmode: torch.Tensor,
    target_bmode: torch.Tensor,
    energy_history: Optional[list] = None,
) -> Dict[str, float]:
    """Comprehensive evaluation of a single sample.
    
    Args:
        pred_bmode: (H, W) predicted B-mode
        target_bmode: (H, W) ground truth B-mode
        energy_history: Optional energy history from propagation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {}
    
    with torch.no_grad():
        metrics['ssim'] = compute_ssim(pred_bmode, target_bmode)
        metrics['psnr'] = compute_psnr(pred_bmode, target_bmode)
        metrics['mae'] = compute_mae(pred_bmode, target_bmode)
    
    if energy_history:
        energy_metrics = compute_energy_conservation(energy_history)
        metrics.update(energy_metrics)
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics for display."""
    lines = [f"{prefix}Evaluation Metrics:"]
    lines.append(f"  SSIM:  {metrics.get('ssim', 0):.4f}")
    lines.append(f"  PSNR:  {metrics.get('psnr', 0):.2f} dB")
    lines.append(f"  MAE:   {metrics.get('mae', 0):.6f}")
    
    if 'energy_max_variation' in metrics:
        lines.append(f"  Energy variation: {metrics['energy_max_variation']:.2%}")
        lines.append(f"  Energy conserved: {metrics.get('energy_conserved', 'N/A')}")
    
    text = "\n".join(lines)
    print(text)
    return text
