"""
combined_loss_v2.py — Clean multi-term loss for DPC-GNN-Acoustic v3.

Three terms:
  1. L1 loss (primary supervision)
  2. SSIM loss (structural similarity, inputs normalized to [0,1])
  3. Physics loss (wave equation residual, weight 0.01)

Total = L1 + λ_ssim * (1 - SSIM) + λ_phys * physics_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SSIMLossV2(nn.Module):
    """Differentiable SSIM loss.

    SSIM is computed on inputs normalized to [0, 1].
    Loss = 1 - SSIM.

    Args:
        window_size: Gaussian window size
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.register_buffer('window', window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            pred: (H, W) or (B, 1, H, W) predicted image in [0, 1]
            target: (H, W) or (B, 1, H, W) target image in [0, 1]

        Returns:
            loss: 1 - SSIM
        """
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)

        # Ensure same spatial size
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)

        # Clamp to [0, 1]
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        pad = self.window_size // 2
        mu_pred = F.conv2d(pred, self.window, padding=pad)
        mu_target = F.conv2d(target, self.window, padding=pad)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, self.window, padding=pad) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, self.window, padding=pad) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, self.window, padding=pad) - mu_cross

        ssim_map = (
            (2 * mu_cross + C1) * (2 * sigma_cross + C2)
        ) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )

        return 1.0 - ssim_map.mean()


class CombinedLossV2(nn.Module):
    """Combined loss for DPC-GNN-Acoustic v3.

    Total = L1 + λ_ssim * (1-SSIM) + λ_phys * physics_loss

    Args:
        lambda_l1: Weight for L1 loss (default 1.0)
        lambda_ssim: Weight for SSIM loss (default 0.5)
        lambda_physics: Weight for physics loss (default 0.01)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_physics: float = 0.01,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_physics = lambda_physics

        self.ssim_loss = SSIMLossV2()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None,
        energy_history: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            pred: (H, W) predicted B-mode image
            target: (H, W) ground truth B-mode image
            physics_loss: scalar physics loss (from model.compute_physics_loss())
            energy_history: list of energy values (unused, for logging only)

        Returns:
            total_loss: scalar
            loss_dict: dictionary with individual loss components (for logging)
        """
        # Ensure same spatial size
        if pred.dim() == 2 and target.dim() == 2:
            if pred.shape != target.shape:
                target = F.interpolate(
                    target.unsqueeze(0).unsqueeze(0),
                    size=pred.shape,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)

        # Normalize both to [0, 1] for fair comparison
        eps = 1e-8
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + eps)
        target_norm = (target - target.min()) / (target.max() - target.min() + eps)

        # L1 loss
        l1 = F.l1_loss(pred_norm, target_norm)

        # SSIM loss
        ssim_loss = self.ssim_loss(pred_norm, target_norm)

        # Physics loss
        phys = physics_loss if physics_loss is not None else torch.tensor(0.0, device=pred.device)

        # Total
        total = (
            self.lambda_l1 * l1
            + self.lambda_ssim * ssim_loss
            + self.lambda_physics * phys
        )

        loss_dict = {
            'l1': l1.item(),
            'ssim': 1.0 - ssim_loss.item(),  # report SSIM value (higher=better)
            'physics': phys.item(),
            'total': total.item(),
        }

        return total, loss_dict


def create_loss_v2(config: dict) -> CombinedLossV2:
    """Factory function to create v2 loss from config.

    Args:
        config: Configuration dictionary

    Returns:
        loss: CombinedLossV2 instance
    """
    loss_cfg = config.get('loss', {})
    return CombinedLossV2(
        lambda_l1=loss_cfg.get('lambda_l1', 1.0),
        lambda_ssim=loss_cfg.get('lambda_ssim', 0.5),
        lambda_physics=loss_cfg.get('lambda_physics', 0.01),
    )
