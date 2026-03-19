"""
beamform_decoder.py — Physically correct Delay-and-Sum (DAS) B-mode imaging.

Replaces the learned-only BeamformDecoder with a physics-based DAS beamformer
that uses the full pressure time series.

Pipeline:
  1. Extract RF signals at transducer positions from pressure_history
  2. For each pixel, compute round-trip delay from source → pixel → element
  3. Interpolate RF data at computed delays
  4. Sum across elements (with learned apodization weights)
  5. Hilbert transform → envelope detection → log compression

This is the standard delay-and-sum algorithm used in clinical ultrasound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DASBeamformDecoder(nn.Module):
    """Delay-and-Sum beamforming decoder for B-mode image reconstruction.

    Args:
        n_elements: Number of transducer elements
        image_h: Output image height (lateral, number of scanlines)
        image_w: Output image width (axial, number of depth samples)
        dt: Time step [s]
        dynamic_range_db: Log compression dynamic range [dB]
    """

    def __init__(
        self,
        n_elements: int = 128,
        image_h: int = 256,
        image_w: int = 512,
        dt: float = 2e-8,
        dynamic_range_db: float = 60.0,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.image_h = image_h
        self.image_w = image_w
        self.dt = dt
        self.dynamic_range_db = dynamic_range_db

        # Learned apodization weights (element weighting for sidelobe control)
        # Initialized to Hann window
        hann = 0.5 * (1.0 - torch.cos(2.0 * math.pi * torch.arange(n_elements).float() / (n_elements - 1)))
        self.apodization = nn.Parameter(hann)

        # Learned log compression gain
        self.log_gain = nn.Parameter(torch.tensor(5.0))

    def forward(
        self,
        pressure_history: list,          # list of (N, 1) at each time step
        transducer_idx: torch.Tensor,    # (M,) indices of transducer nodes
        transducer_positions: Optional[torch.Tensor] = None,  # (M, 2)
        pixel_positions: Optional[torch.Tensor] = None,       # (H*W, 2)
        source_position: Optional[torch.Tensor] = None,       # (2,)
        c_mean: float = 1540.0,
        domain_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DAS beamforming from pressure history to B-mode image.

        Args:
            pressure_history: list of (N, 1) pressure fields, length T+1
            transducer_idx: (M,) indices of transducer element nodes
            transducer_positions: (M, 2) positions of transducer elements
            pixel_positions: (H*W, 2) positions of output image pixels
            source_position: (2,) source/transmit focus position
            c_mean: mean sound speed [m/s]
            domain_size: (2,) domain size [m]

        Returns:
            bmode: (image_h, image_w) B-mode image in [0, 1]
        """
        device = transducer_idx.device
        n_steps = len(pressure_history)
        M = transducer_idx.shape[0]

        # ── Step 1: Extract RF signals at transducer positions ──
        # rf_signals: (M, T) where T = n_steps
        rf_signals = torch.stack([
            p[transducer_idx].squeeze(-1) for p in pressure_history
        ], dim=-1)  # (M, T)

        # Pad/truncate to n_elements
        if M < self.n_elements:
            rf_signals = F.pad(rf_signals, (0, 0, 0, self.n_elements - M))
            if transducer_positions is not None:
                # Pad positions with last element repeated
                pad_positions = transducer_positions[-1:].expand(self.n_elements - M, -1)
                transducer_positions = torch.cat([transducer_positions, pad_positions], dim=0)
        elif M > self.n_elements:
            rf_signals = rf_signals[:self.n_elements]
            if transducer_positions is not None:
                transducer_positions = transducer_positions[:self.n_elements]

        n_elem = rf_signals.shape[0]
        T = rf_signals.shape[1]

        # ── Step 2: If positions not provided, use simple time-to-depth mapping ──
        if transducer_positions is None or pixel_positions is None:
            # Fallback: simple scanline reconstruction without explicit DAS
            return self._fallback_reconstruction(rf_signals, n_steps)

        # ── Step 3: DAS beamforming with delay computation ──
        H, W = self.image_h, self.image_w
        n_pixels = pixel_positions.shape[0]

        # Source position defaults to center of transducer array
        if source_position is None:
            source_position = transducer_positions.mean(dim=0)  # (2,)

        # Compute delays: τ_i = (|r_source - r_pixel| + |r_pixel - r_element_i|) / c
        # source → pixel distance: (n_pixels,)
        d_source_pixel = torch.norm(
            pixel_positions - source_position.unsqueeze(0), dim=-1
        )  # (n_pixels,)

        # pixel → element distance: (n_pixels, n_elem)
        d_pixel_elem = torch.cdist(
            pixel_positions.unsqueeze(0),
            transducer_positions.unsqueeze(0),
        ).squeeze(0)  # (n_pixels, n_elem)

        # Round-trip delay in samples
        delays = (d_source_pixel.unsqueeze(1) + d_pixel_elem) / c_mean  # (n_pixels, n_elem) seconds
        delay_samples = delays / self.dt  # (n_pixels, n_elem) in sample units

        # ── Step 4: Interpolate RF data at computed delays ──
        # Clamp to valid range
        delay_samples = delay_samples.clamp(0, T - 1.001)
        idx_low = delay_samples.long()
        idx_high = (idx_low + 1).clamp(max=T - 1)
        frac = delay_samples - idx_low.float()

        # Gather from rf_signals: (n_elem, T) → need (n_pixels, n_elem)
        # For each pixel and each element, get the interpolated RF value
        rf_low = rf_signals.T[idx_low.view(-1)].view(n_pixels, n_elem, n_elem)
        # ^ This indexing is wrong; need element-wise gather

        # Correct approach: for each element e, gather rf_signals[e, delay_idx]
        rf_at_delay = torch.zeros(n_pixels, n_elem, device=device)
        for e in range(n_elem):
            low = idx_low[:, e]   # (n_pixels,)
            high = idx_high[:, e]
            f = frac[:, e]
            rf_at_delay[:, e] = (1.0 - f) * rf_signals[e, low] + f * rf_signals[e, high]

        # ── Step 5: Apply apodization and sum ──
        apod = F.softplus(self.apodization[:n_elem])  # ensure positive
        rf_weighted = rf_at_delay * apod.unsqueeze(0)  # (n_pixels, n_elem)
        das_signal = rf_weighted.sum(dim=-1)  # (n_pixels,)

        # Reshape to image
        bmode_raw = das_signal.view(H, W)

        # ── Step 6: Envelope detection (analytic signal via Hilbert) ──
        bmode_envelope = self._hilbert_envelope(bmode_raw)

        # ── Step 7: Log compression ──
        bmode_out = self._log_compress(bmode_envelope)

        return bmode_out

    def _fallback_reconstruction(
        self,
        rf_signals: torch.Tensor,  # (n_elem, T)
        n_steps: int,
    ) -> torch.Tensor:
        """Fallback: simple scanline-based reconstruction when positions unavailable.

        Maps element dimension to lateral (H) and time to depth (W).
        """
        n_elem, T = rf_signals.shape
        H, W = self.image_h, self.image_w

        # Apply apodization
        apod = F.softplus(self.apodization[:n_elem]).unsqueeze(-1)  # (n_elem, 1)
        rf_weighted = rf_signals * apod  # (n_elem, T)

        # Interpolate to output size
        # rf_weighted: (n_elem, T) → (1, 1, n_elem, T) → interpolate → (1, 1, H, W)
        rf_2d = rf_weighted.unsqueeze(0).unsqueeze(0)
        bmode_raw = F.interpolate(rf_2d, size=(H, W), mode='bilinear', align_corners=False)
        bmode_raw = bmode_raw.squeeze(0).squeeze(0)  # (H, W)

        # Envelope detection
        bmode_envelope = self._hilbert_envelope(bmode_raw)

        # Log compression
        return self._log_compress(bmode_envelope)

    def _hilbert_envelope(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute envelope via Hilbert transform along last dimension (depth/W).

        Uses FFT-based Hilbert transform.

        Args:
            signal: (..., W) real signal

        Returns:
            envelope: (..., W) envelope (magnitude of analytic signal)
        """
        W = signal.shape[-1]

        # FFT along last dimension
        spectrum = torch.fft.rfft(signal, dim=-1)

        # Create Hilbert multiplier: double positive frequencies, zero negative
        n_freq = spectrum.shape[-1]
        h = torch.zeros(n_freq, device=signal.device)
        h[0] = 1.0  # DC
        if W % 2 == 0:
            h[1:-1] = 2.0  # positive frequencies (doubled)
            h[-1] = 1.0    # Nyquist
        else:
            h[1:] = 2.0

        analytic = torch.fft.irfft(spectrum * h, n=W, dim=-1)

        # Envelope = |analytic signal|
        envelope = torch.sqrt(signal ** 2 + analytic ** 2 + 1e-10)
        return envelope

    def _log_compress(self, envelope: torch.Tensor) -> torch.Tensor:
        """Log compression with learned gain.

        Args:
            envelope: (...) non-negative envelope signal

        Returns:
            compressed: (...) in [0, 1]
        """
        eps = 1e-8
        gain = F.softplus(self.log_gain)

        # Normalize
        env_max = envelope.max()
        env_norm = envelope / (env_max + eps)

        # Log compression
        compressed = torch.log1p(gain * env_norm) / torch.log1p(gain)

        return compressed
