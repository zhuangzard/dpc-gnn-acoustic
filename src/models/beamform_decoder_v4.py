"""
DPC-GNN-Acoustic V4: Differentiable DAS Beamformer

Delay-and-Sum beamforming with:
  - Plane-wave TX + per-element RX delay model
  - Linear interpolation for differentiable delay computation
  - FFT-based Hilbert transform for envelope detection
  - dB log compression matching GT convention

NO learnable parameters — pure signal processing.

Coordinate system: UNIFIED with wave propagator (V4 grid: 256×256, dx=2.34e-4m)

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
    
    Coordinate system matches wave_propagator_v4:
      - dx, grid_size, pml_width all from V4's 256-grid
      - transducer_row = pml_width + 1 (same as source/sensor in propagator)
    """

    def __init__(self, n_elements: int = 128, c_ref: float = 1540.0,
                 image_size: tuple = (128, 128),
                 dx: float = 2.34e-4, dt: float = 2.0e-8,
                 grid_size: int = 256, pml_size: int = 20,
                 # GT FOV parameters for pixel alignment
                 # Both V4 and GT beamformers must image the same physical region
                 gt_dx: float = 4.69e-4, gt_grid_size: int = 128, gt_pml: int = 20,
                 eps: float = 1e-6, dynamic_range_db: float = 60.0):
        super().__init__()
        self.n_elements = n_elements
        self.c_ref = c_ref
        self.image_h, self.image_w = image_size
        self.dx = dx  # V4's dx for element positioning
        self.dt = dt
        self.grid_size = grid_size
        self.pml_size = pml_size
        # GT FOV: pixel grid covers GT's physical region for alignment
        self.gt_dx = gt_dx
        self.gt_grid_size = gt_grid_size
        self.gt_pml = gt_pml
        self.eps = eps
        self.dynamic_range_db = dynamic_range_db

        self._delays_computed = False

    def _compute_delay_table(self, n_samples: int, device: torch.device,
                              dtype: torch.dtype):
        """
        Compute the delay table for DAS beamforming.
        
        Element positions use V4's coordinate system.
        Pixel grid uses GT's FOV for alignment with GT B-mode.
        """
        pml = self.pml_size
        G = self.grid_size
        dx = self.dx
        
        # Element positions in V4 coordinates
        # Must match wave_propagator_v4's sensor_x and transducer_row
        transducer_row = pml + 1  # row 21 in 256-grid
        active_start = pml  # grid index 20
        active_end = G - pml - 1  # grid index 235
        
        elem_lateral = torch.linspace(active_start * dx, active_end * dx,
                                       self.n_elements, device=device, dtype=dtype)
        elem_axial = transducer_row * dx  # V4 sensor depth

        # Pixel grid uses GT's FOV to ensure spatial alignment
        # GT lateral: gt_pml*gt_dx to (gt_G-gt_pml-1)*gt_dx
        # GT axial: (gt_pml+1)*gt_dx to (gt_G-gt_pml)*gt_dx
        gt_dx = self.gt_dx
        gt_G = self.gt_grid_size
        gt_pml = self.gt_pml
        gt_active_width = gt_G - 2 * gt_pml
        gt_n_elem = min(128, gt_active_width)
        gt_start_col = (gt_G - gt_n_elem) // 2
        
        px = torch.linspace(gt_start_col * gt_dx,
                             (gt_start_col + gt_n_elem - 1) * gt_dx,
                             self.image_w, device=device, dtype=dtype)
        axial_start = (gt_pml + 1) * gt_dx  # GT sensor depth
        axial_end = (gt_G - gt_pml) * gt_dx
        py = torch.linspace(axial_start, axial_end,
                             self.image_h, device=device, dtype=dtype)

        # Grid of pixel positions [H, W]
        grid_y, grid_x = torch.meshgrid(py, px, indexing='ij')

        # Distance from each pixel to each element
        dist_x = grid_x.unsqueeze(-1) - elem_lateral.reshape(1, 1, -1)
        dist_y = grid_y.unsqueeze(-1) - elem_axial
        distance = torch.sqrt(dist_x ** 2 + dist_y ** 2 + 1e-12)

        # Plane-wave transmit + per-element receive (pulse-echo)
        # TX: all elements fire simultaneously → plane wave → delay = d_axial / c
        # RX: scattered wave → delay = dist(pixel, element) / c
        d_axial = (grid_y - elem_axial).abs().unsqueeze(-1)
        delay_samples = (d_axial + distance) / (self.c_ref * self.dt)

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

        # Flatten for gather
        delay_floor_flat = delay_floor.reshape(-1, E)  # [H*W, E]
        delay_frac_flat = delay_frac.reshape(-1, E)

        idx_lo = delay_floor_flat
        idx_hi = (idx_lo + 1).clamp(max=n_samples - 1)
        frac = delay_frac_flat

        HW = idx_lo.shape[0]
        sd_flat = sensor_data.reshape(B * E, n_samples)

        idx_lo_t = idx_lo.t().unsqueeze(0).expand(B, -1, -1).reshape(B * E, HW)
        idx_hi_t = idx_hi.t().unsqueeze(0).expand(B, -1, -1).reshape(B * E, HW)

        val_lo = torch.gather(sd_flat, 1, idx_lo_t).reshape(B, E, HW)
        val_hi = torch.gather(sd_flat, 1, idx_hi_t).reshape(B, E, HW)

        frac_t = frac.t().unsqueeze(0).expand(B, -1, -1)
        val = val_lo * (1.0 - frac_t) + val_hi * frac_t
        rf_image = val.sum(dim=1).reshape(B, H, W)
        return rf_image

    def _hilbert_envelope(self, rf: torch.Tensor) -> torch.Tensor:
        """
        Compute analytic signal envelope via FFT-based Hilbert transform.
        Applied along the axial (depth/H) dimension for each A-line.
        
        Args:
            rf: [B, H, W] RF image
        Returns:
            envelope: [B, H, W] envelope (always positive)
        """
        B, H, W = rf.shape
        N = H

        # Full FFT along depth dimension
        Rf = torch.fft.fft(rf, dim=1)

        # Build Hilbert multiplier for analytic signal
        h = torch.zeros(N, device=rf.device, dtype=rf.dtype)
        h[0] = 1.0
        if N % 2 == 0:
            h[N // 2] = 1.0
            h[1:N // 2] = 2.0
        else:
            h[1:(N + 1) // 2] = 2.0

        # Apply and inverse FFT → complex analytic signal
        analytic = torch.fft.ifft(Rf * h.reshape(1, -1, 1), dim=1)

        # Envelope = magnitude of complex analytic signal
        envelope = analytic.abs() + self.eps
        return envelope

    def _log_compress(self, envelope: torch.Tensor) -> torch.Tensor:
        """
        Log compression EXACTLY matching GT convention (regenerate_gt_bmode.py):
        log_env = ln(envelope + eps)
        bmode = (log_env - min) / (max - min)   → [0, 1]
        
        MUST use natural log (ln), NOT 20·log10.
        MUST use per-sample min-max, NOT clip-and-shift.
        
        Args:
            envelope: [B, H, W]
        Returns:
            compressed: [B, H, W] in [0, 1]
        """
        B = envelope.shape[0]
        
        # Natural log (matching GT's np.log)
        log_env = torch.log(envelope + self.eps)
        
        # Per-sample min-max normalization (matching GT's min-max)
        log_flat = log_env.reshape(B, -1)
        log_min = log_flat.min(dim=1, keepdim=True)[0]
        log_max = log_flat.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denom = (log_max - log_min).clamp(min=1e-8)
        compressed = (log_flat - log_min) / denom
        
        return compressed.reshape_as(envelope)

    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Full beamforming pipeline.
        
        Args:
            sensor_data: [B, n_elements, n_samples]
        Returns:
            bmode: [B, 1, H, W] B-mode image in [0, 1]
        """
        rf_image = self._das_beamform(sensor_data)
        envelope = self._hilbert_envelope(rf_image)
        bmode = self._log_compress(envelope)
        return bmode.unsqueeze(1)


if __name__ == '__main__':
    print("Testing DifferentiableBeamformerV4...")

    bf = DifferentiableBeamformerV4(
        n_elements=128, c_ref=1540.0, image_size=(128, 128),
        dx=2.34e-4, dt=2.0e-8, grid_size=256, pml_size=20,
    )

    B = 2
    n_samples = 200
    sensor_data = torch.randn(B, 128, n_samples, requires_grad=True)

    bmode = bf(sensor_data)
    print(f"B-mode shape: {bmode.shape}")
    print(f"B-mode range: [{bmode.min():.4f}, {bmode.max():.4f}]")

    n_params = sum(p.numel() for p in bf.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params} (should be 0)")

    loss = bmode.sum()
    loss.backward()
    print(f"Gradient on sensor_data exists: {sensor_data.grad is not None}")
    print(f"Gradient norm: {sensor_data.grad.norm():.6f}")
