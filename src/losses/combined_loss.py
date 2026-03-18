"""
combined_loss.py — Multi-term loss function for DPC-GNN-Acoustic v2.

Combines:
  1. L1 loss: GNN B-mode vs k-Wave GT
  2. SSIM loss: Structural similarity
  3. Physics loss: Wave equation residual
  4. Perceptual loss: VGG feature matching (optional)
  5. Energy conservation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss.
    
    SSIM measures structural similarity between two images.
    Loss = 1 - SSIM (so minimizing loss maximizes SSIM).
    
    Args:
        window_size: Gaussian window size
        n_channels: Number of image channels
    """
    
    def __init__(self, window_size: int = 11, n_channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        
        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # 2D window
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.register_buffer('window', window)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.
        
        Args:
            pred: (H, W) or (B, 1, H, W) predicted image
            target: (H, W) or (B, 1, H, W) target image
        
        Returns:
            loss: 1 - SSIM
        """
        # Ensure 4D
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        
        # Resize target if needed
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        window = self.window.to(pred.device)
        pad = self.window_size // 2
        
        mu1 = F.conv2d(pred, window, padding=pad)
        mu2 = F.conv2d(target, window, padding=pad)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred ** 2, window, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=pad) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_val = ssim_map.mean()
        
        return 1.0 - ssim_val


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss (optional, requires torchvision).
    
    Compares features from intermediate VGG layers.
    Falls back to L1 if torchvision not available.
    """
    
    def __init__(self):
        super().__init__()
        self._vgg = None
        self._available = False
        
        try:
            from torchvision.models import vgg16
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self._vgg = vgg
            self._available = True
        except (ImportError, Exception):
            print("⚠️ VGG perceptual loss unavailable, using L1 fallback")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available or self._vgg is None:
            return F.l1_loss(pred, target)
        
        # Ensure 4D, 3-channel
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        
        # Resize target if needed
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        
        # Repeat to 3 channels (VGG expects RGB)
        pred_3ch = pred.repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred
        target_3ch = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        
        self._vgg = self._vgg.to(pred.device)
        
        feat_pred = self._vgg(pred_3ch)
        feat_target = self._vgg(target_3ch)
        
        return F.l1_loss(feat_pred, feat_target)


class CombinedLoss(nn.Module):
    """Combined loss function for DPC-GNN-Acoustic v2 training.
    
    L_total = λ₁·L_L1 + λ₂·L_SSIM + λ₃·L_physics + λ₄·L_perceptual + λ₅·L_energy
    
    Args:
        lambda_l1: Weight for L1 loss
        lambda_ssim: Weight for SSIM loss
        lambda_physics: Weight for physics (wave equation residual) loss
        lambda_perceptual: Weight for perceptual loss
        lambda_energy: Weight for energy conservation loss
        use_perceptual: Enable VGG perceptual loss
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_physics: float = 0.1,
        lambda_perceptual: float = 0.05,
        lambda_energy: float = 0.01,
        use_perceptual: bool = False,
    ):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_physics = lambda_physics
        self.lambda_perceptual = lambda_perceptual
        self.lambda_energy = lambda_energy
        
        self.ssim_loss = SSIMLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        pred_bmode: torch.Tensor,       # (H, W) predicted B-mode
        target_bmode: torch.Tensor,      # (H, W) k-Wave GT B-mode
        physics_loss: Optional[torch.Tensor] = None,
        energy_history: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.
        
        Args:
            pred_bmode: Predicted B-mode image
            target_bmode: k-Wave ground truth B-mode
            physics_loss: Pre-computed wave equation residual
            energy_history: Energy values from propagation
        
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_bmode.device)
        
        # Resize target to match prediction if needed
        if pred_bmode.shape != target_bmode.shape:
            target_bmode = F.interpolate(
                target_bmode.unsqueeze(0).unsqueeze(0),
                size=pred_bmode.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
        
        # ── L1 loss ──
        l1 = F.l1_loss(pred_bmode, target_bmode)
        total_loss = total_loss + self.lambda_l1 * l1
        loss_dict['l1'] = l1.item()
        
        # ── SSIM loss ──
        ssim = self.ssim_loss(pred_bmode, target_bmode)
        total_loss = total_loss + self.lambda_ssim * ssim
        loss_dict['ssim'] = ssim.item()
        
        # ── Physics loss ──
        if physics_loss is not None:
            total_loss = total_loss + self.lambda_physics * physics_loss
            loss_dict['physics'] = physics_loss.item()
        
        # ── Perceptual loss ──
        if self.perceptual_loss is not None:
            perc = self.perceptual_loss(pred_bmode, target_bmode)
            total_loss = total_loss + self.lambda_perceptual * perc
            loss_dict['perceptual'] = perc.item()
        
        # ── Energy conservation loss ──
        if energy_history is not None and len(energy_history) > 1:
            energies = torch.tensor(energy_history, device=pred_bmode.device)
            # Penalize energy growth (should be non-increasing with attenuation)
            energy_diff = energies[1:] - energies[:-1]
            energy_growth = F.relu(energy_diff).mean()  # Only penalize growth
            total_loss = total_loss + self.lambda_energy * energy_growth
            loss_dict['energy'] = energy_growth.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def create_loss(config: dict) -> CombinedLoss:
    """Create loss from config dict."""
    loss_cfg = config.get('loss', {})
    return CombinedLoss(
        lambda_l1=loss_cfg.get('lambda_l1', 1.0),
        lambda_ssim=loss_cfg.get('lambda_ssim', 0.5),
        lambda_physics=loss_cfg.get('lambda_physics', 0.1),
        lambda_perceptual=loss_cfg.get('lambda_perceptual', 0.05),
        lambda_energy=loss_cfg.get('lambda_energy', 0.01),
        use_perceptual=loss_cfg.get('lambda_perceptual', 0.05) > 0,
    )
