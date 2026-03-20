"""
DPC-GNN-Acoustic V4: Combined Loss Function

Training loss: L1 + (1 - SSIM)
Monitoring metrics (torch.no_grad): physics_residual, c_mean, c_std, alpha_mean, energy_ratio

NO physics_weight — physics is NOT a loss term, only monitored!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) — differentiable implementation.
    Computed over [B, 1, H, W] images.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self._window = None

    def _create_gaussian_window(self, channels: int, device: torch.device,
                                 dtype: torch.dtype) -> torch.Tensor:
        """Create 2D Gaussian window for SSIM."""
        size = self.window_size
        sigma = self.sigma
        coords = torch.arange(size, device=device, dtype=dtype) - size // 2
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        window = g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
        return window.contiguous()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between x and y.
        
        Args:
            x, y: [B, 1, H, W] images
        Returns:
            ssim_val: scalar, mean SSIM
        """
        C = x.size(1)
        if self._window is None or self._window.device != x.device:
            self._window = self._create_gaussian_window(C, x.device, x.dtype)

        pad = self.window_size // 2

        mu_x = F.conv2d(x, self._window, padding=pad, groups=C)
        mu_y = F.conv2d(y, self._window, padding=pad, groups=C)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, self._window, padding=pad, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, self._window, padding=pad, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self._window, padding=pad, groups=C) - mu_xy

        # Clamp for numerical stability
        sigma_x_sq = sigma_x_sq.clamp(min=0)
        sigma_y_sq = sigma_y_sq.clamp(min=0)

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))

        return ssim_map.mean()


class DPCGNNAcousticLossV4(nn.Module):
    """
    Combined loss for DPC-GNN-Acoustic V4.
    
    Training loss = L1(pred, gt) + (1 - SSIM(pred, gt))
    
    Monitoring metrics (no gradient):
      - physics_residual: wave equation residual (should be ≈0)
      - c_mean, c_std: speed-of-sound field statistics
      - alpha_mean: attenuation statistics
      - energy_ratio: energy conservation check
    """

    def __init__(self, loss_type: str = 'l1_ssim'):
        super().__init__()
        self.ssim_fn = SSIM(window_size=11, sigma=1.5)
        self.l1_fn = nn.L1Loss()
        self.loss_type = loss_type  # 'l1_ssim' (default) or 'l1_only'

    def forward(self, pred_bmode: torch.Tensor, gt_bmode: torch.Tensor,
                model_outputs: dict = None) -> tuple:
        """
        Args:
            pred_bmode: [B, 1, H, W] predicted B-mode
            gt_bmode: [B, 1, H, W] ground truth B-mode
            model_outputs: dict with 'c', 'alpha', 'sigma', 'sensor_data' for monitoring
        Returns:
            loss: scalar training loss
            metrics: dict of monitoring metrics
        """
        # --- Training loss ---
        l1_loss = self.l1_fn(pred_bmode, gt_bmode)
        ssim_val = self.ssim_fn(pred_bmode, gt_bmode)
        ssim_loss = 1.0 - ssim_val

        if self.loss_type == 'l1_only':
            loss = l1_loss  # Ablation: no SSIM component
        else:
            loss = l1_loss + ssim_loss

        # --- Monitoring metrics (no gradient!) ---
        metrics = {
            'l1': l1_loss.item(),
            'ssim': ssim_val.item(),
            'ssim_loss': ssim_loss.item(),
            'total_loss': loss.item(),
        }

        if model_outputs is not None:
            with torch.no_grad():
                c = model_outputs.get('c')
                alpha = model_outputs.get('alpha')
                sigma = model_outputs.get('sigma')
                sensor_data = model_outputs.get('sensor_data')

                if c is not None:
                    metrics['c_mean'] = c.mean().item()
                    metrics['c_std'] = c.std().item()
                    metrics['c_min'] = c.min().item()
                    metrics['c_max'] = c.max().item()

                if alpha is not None:
                    metrics['alpha_mean'] = alpha.mean().item()
                    metrics['alpha_max'] = alpha.max().item()

                if sigma is not None:
                    metrics['sigma_mean'] = sigma.mean().item()

                if sensor_data is not None:
                    # Energy ratio: compare energy at start vs end of signal
                    n_samples = sensor_data.size(2)
                    quarter = max(1, n_samples // 4)
                    energy_start = (sensor_data[:, :, :quarter] ** 2).sum()
                    energy_end = (sensor_data[:, :, -quarter:] ** 2).sum()
                    total_energy = (sensor_data ** 2).sum()
                    if total_energy > 1e-12:
                        metrics['energy_ratio'] = (energy_end / (total_energy + 1e-12)).item()
                    else:
                        metrics['energy_ratio'] = 0.0

                # Physics residual placeholder — computed externally if needed
                # (requires consecutive pressure fields, not stored by default)
                metrics['physics_residual'] = 0.0

        return loss, metrics


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing DPCGNNAcousticLossV4...")
    loss_fn = DPCGNNAcousticLossV4()

    # Fake data
    pred = torch.rand(2, 1, 128, 128, requires_grad=True)
    gt = torch.rand(2, 1, 128, 128)
    model_out = {
        'c': torch.ones(2, 1, 256, 256) * 1540 + torch.randn(2, 1, 256, 256) * 20,
        'alpha': torch.rand(2, 1, 256, 256) * 5,
        'sigma': torch.rand(2, 1, 256, 256),
        'sensor_data': torch.randn(2, 128, 200),
    }

    loss, metrics = loss_fn(pred, gt, model_out)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Gradient check
    loss.backward()
    print(f"Gradient on pred exists: {pred.grad is not None}")
    print(f"Gradient norm: {pred.grad.norm():.6f}")
