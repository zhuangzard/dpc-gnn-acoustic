"""
DPC-GNN-Acoustic V4: Differentiable DAS Beamformer

Delay-and-Sum beamforming with:
  - Linear interpolation for differentiable delay computation
  - FFT-based Hilbert transform for envelope detection
  - Smooth log compression

NO learnable parameters — pure signal processing.

Input:  sensor_data [B, n_elements, n_samples]
Output: B-mode image [B, 1, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBeamformerV4(nn.Module):
    """
    Differentiable Delay-and-Sum (DAS) beamformer.
    All operations are differentiable for end-to-end training.
    Zero learnable parameters.
    """

    def __init__(self, n_elements: int = 128, c_ref: float = 1540.0,
                 image_size: tuple = (128, 128),
                 dx: float = 2.34e-4, dt: float = 2.0e-8,
                 eps: float = 1e-6):
        super().__init__()
        self.n_elements = n_elements
        self.c_ref = c_ref
        self.image_h, self.image_w = image_size
        self.dx = dx
        self.dt = dt
        self.eps = eps

        # Pre-compute pixel-to-element delay indices
        # These are registered as buffers (non-learnable, moved with .to())
        self._delays_computed = False

    def _compute_delay_table(self, n_samples: int, device: torch.device,
                              dtype: torch.dtype):
        """
        Compute the delay table for DAS beamforming.
        
        For each pixel (iz, ix) and each element e:
            delay[iz, ix, e] = distance(pixel, element) / c_ref / dt
            
        Pixel grid maps to physical space matching the simulation grid.
        Elements are along y=0 (top row), uniformly spaced.
        """
        # Physical extent of imaging region
        # Image covers the interior (excluding PML)
        pml = 20  # PML width in grid points
        phys_width = (self.image_w) * self.dx  # physical width
        phys_depth = (self.image_h) * self.dx  # physical depth

        # Element positions along x-axis (at y=0)
        elem_x = torch.linspace(pml * self.dx, (256 - pml) * self.dx,
                                 self.n_elements, device=device, dtype=dtype)
        elem_y = torch.zeros(self.n_elements, device=device, dtype=dtype)

        # Pixel positions
        px = torch.linspace(pml * self.dx, (256 - pml) * self.dx,
                             self.image_w, device=device, dtype=dtype)
        py = torch.linspace(0, phys_depth, self.image_h, device=device, dtype=dtype)

        # Grid of pixel positions [H, W]
        grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')

        # Distance from each pixel to each element
        # [H, W, 1] - [1, 1, n_elements]
        dist_x = grid_x.unsqueeze(-1) - elem_x.reshape(1, 1, -1)
        dist_y = grid_y.unsqueeze(-1) - elem_y.reshape(1, 1, -1)
        distance = torch.sqrt(dist_x ** 2 + dist_y ** 2 + 1e-12)

        # Convert to sample index (delay in samples)
        delay_samples = distance / (self.c_ref * self.dt)

        # Clamp to valid range
        delay_samples = delay_samples.clamp(0, n_samples - 2)

        self.register_buffer('delay_samples', delay_samples)  # [H, W, n_elements]
        self._delays_computed = True

    def _das_beamform(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Delay-and-Sum with differentiable linear interpolation.
        
        Args:
            sensor_data: [B, n_elements, n_samples]
        Returns:
            rf_image: [B, H, W] beamformed RF image
        """
        B, n_elem, n_samples = sensor_data.shape

        if not self._delays_computed:
            self._compute_delay_table(n_samples, sensor_data.device, sensor_data.dtype)

        delay = self.delay_samples  # [H, W, n_elements]
        H, W, E = delay.shape

        # Integer and fractional parts for linear interpolation
        delay_floor = delay.long()
        delay_frac = delay - delay_floor.float()

        # Flatten spatial dims for gather
        delay_floor_flat = delay_floor.reshape(-1, E)  # [H*W, E]
        delay_frac_flat = delay_frac.reshape(-1, E)    # [H*W, E]

        # Vectorized DAS: no Python loop over elements
        # idx_lo/hi: [H*W, E], sensor_data: [B, E, n_samples]
        idx_lo = delay_floor_flat  # [H*W, E]
        idx_hi = (idx_lo + 1).clamp(max=n_samples - 1)
        frac = delay_frac_flat  # [H*W, E]

        HW, E2 = idx_lo.shape
        # Flatten sensor_data to [B*E, n_samples] for gather
        sd_flat = sensor_data.reshape(B * E2, n_samples)

        # Build gather indices: for each (element, pixel) pair
        # idx_lo.t() -> [E, H*W], expand for batch -> [B, E, H*W] -> [B*E, H*W]
        idx_lo_t = idx_lo.t().unsqueeze(0).expand(B, -1, -1).reshape(B * E2, HW)
        idx_hi_t = idx_hi.t().unsqueeze(0).expand(B, -1, -1).reshape(B * E2, HW)

        val_lo = torch.gather(sd_flat, 1, idx_lo_t).reshape(B, E2, HW)
        val_hi = torch.gather(sd_flat, 1, idx_hi_t).reshape(B, E2, HW)

        frac_t = frac.t().unsqueeze(0).expand(B, -1, -1)  # [B, E, H*W]
        val = val_lo * (1.0 - frac_t) + val_hi * frac_t
        rf_image = val.sum(dim=1).reshape(B, H, W)  # sum over elements
        return rf_image

    def _hilbert_envelope(self, rf: torch.Tensor) -> torch.Tensor:
        """
        Compute analytic signal envelope via FFT-based Hilbert transform.
        Applied along the depth (H) dimension.
        
        Args:
            rf: [B, H, W] RF image
        Returns:
            envelope: [B, H, W] envelope
        """
        B, H, W = rf.shape

        # FFT along depth dimension
        Rf = torch.fft.rfft(rf, dim=1)

        # Build Hilbert multiplier: h[0]=1, h[1..N/2-1]=2, h[N/2]=1, rest=0
        N = H
        n_rfft = Rf.shape[1]
        h = torch.zeros(n_rfft, device=rf.device, dtype=rf.dtype)
        h[0] = 1.0
        if N % 2 == 0:
            h[N // 2] = 1.0
            h[1:N // 2] = 2.0
        else:
            h[1:(N + 1) // 2] = 2.0

        # Apply multiplier
        analytic_fft = Rf * h.reshape(1, -1, 1)
        analytic = torch.fft.irfft(analytic_fft, n=H, dim=1)

        # Envelope = |analytic signal|
        # Use rf as real part, analytic as the full complex reconstruction
        envelope = torch.sqrt(rf ** 2 + analytic ** 2 + self.eps)
        return envelope

    def _log_compress(self, envelope: torch.Tensor) -> torch.Tensor:
        """
        Smooth log compression: log(envelope + eps).
        Then normalise to [0, 1] per sample.
        
        Args:
            envelope: [B, H, W]
        Returns:
            compressed: [B, H, W] in [0, 1]
        """
        log_env = torch.log(envelope + self.eps)

        # Normalise per sample to [0, 1]
        B = log_env.shape[0]
        log_flat = log_env.reshape(B, -1)
        min_val = log_flat.min(dim=1, keepdim=True)[0]
        max_val = log_flat.max(dim=1, keepdim=True)[0]
        range_val = (max_val - min_val).clamp(min=self.eps)

        normalised = (log_flat - min_val) / range_val
        return normalised.reshape_as(log_env)

    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Full beamforming pipeline.
        
        Args:
            sensor_data: [B, n_elements, n_samples]
        Returns:
            bmode: [B, 1, H, W] B-mode image in [0, 1]
        """
        # 1. DAS beamforming
        rf_image = self._das_beamform(sensor_data)  # [B, H, W]

        # 2. Hilbert envelope detection
        envelope = self._hilbert_envelope(rf_image)  # [B, H, W]

        # 3. Log compression + normalisation
        bmode = self._log_compress(envelope)  # [B, H, W]

        return bmode.unsqueeze(1)  # [B, 1, H, W]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing DifferentiableBeamformerV4...")

    bf = DifferentiableBeamformerV4(
        n_elements=128, c_ref=1540.0, image_size=(128, 128),
        dx=2.34e-4, dt=2.0e-8,
    )

    # Fake sensor data
    B = 2
    n_samples = 200
    sensor_data = torch.randn(B, 128, n_samples, requires_grad=True)

    bmode = bf(sensor_data)
    print(f"B-mode shape: {bmode.shape}")  # [2, 1, 128, 128]
    print(f"B-mode range: [{bmode.min():.4f}, {bmode.max():.4f}]")

    # Check no learnable params
    n_params = sum(p.numel() for p in bf.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params} (should be 0)")

    # Check gradient flow
    loss = bmode.sum()
    loss.backward()
    print(f"Gradient on sensor_data exists: {sensor_data.grad is not None}")
    print(f"Gradient norm: {sensor_data.grad.norm():.6f}")
