"""
DPC-GNN-Acoustic V4: Deterministic Leapfrog Wave Propagator with k-space Correction

2D acoustic wave equation with k-space pseudospectral Laplacian:
    p^{n+1} = (2p^n - (1-σ_total·Δt/2)·p^{n-1} 
               + κ · (c/c_ref)² · Δt²·c_ref² · ∇²_spectral(p^n)
               + Δt²·f_source) / (1+σ_total·Δt/2)

where:
  - ∇²_spectral = ifft2(-k² · fft2(p))  — exact spectral Laplacian (no numerical dispersion)
  - κ = sinc(c_ref·|k|·Δt/2)  — k-space time correction factor
  - σ_total = α (physical attenuation) + σ_pml (PML damping)
  - c_ref = reference speed of sound for k-space correction

Key improvement over standard 2nd-order FD:
  - FD at 3 PPW: >30% phase velocity error
  - k-space corrected: <1% phase velocity error at same PPW
  - Matches k-Wave PSTD accuracy that generated the GT data

Features:
  - NO learnable parameters (pure physics)
  - k-space corrected pseudospectral Laplacian (near-zero numerical dispersion)
  - PML absorbing boundary (20 cells, cubic polynomial decay)
  - Gradient checkpointing every 50 steps
  - Source and sensor at SAME ROW (pulse-echo ultrasound)
  - CFL stability check
  - torch.fft is fully differentiable for end-to-end training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AcousticLeapfrogV4(nn.Module):
    """
    Deterministic 2D acoustic Leapfrog propagator with k-space correction.
    Zero learnable parameters — all physics.
    """

    def __init__(self, nx: int = 256, ny: int = 256,
                 dx: float = 2.34e-4, dt: float = 2.0e-8,
                 n_steps: int = 200, pml_width: int = 20,
                 n_elements: int = 128, checkpoint_every: int = 50,
                 c_ref: float = 1700.0):
        """
        c_ref MUST be c_max (not c_mean). When c_local > c_ref, kappa
        under-corrects → numerical instability risk. c_ref=c_max=1700
        ensures stable over-correction for all c values.
        """
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.n_steps = n_steps
        self.pml_width = pml_width
        self.n_elements = n_elements
        self.checkpoint_every = checkpoint_every
        self.c_ref = c_ref

        # --- k-space wavenumber grid (precomputed) ---
        # For rfft2: kx has nx//2+1 components, ky has ny components
        kx = torch.fft.rfftfreq(nx, d=dx) * 2.0 * math.pi  # [nx//2+1]
        ky = torch.fft.fftfreq(ny, d=dx) * 2.0 * math.pi   # [ny]
        # k² grid: [ny, nx//2+1]
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
        k_sq = kx_grid ** 2 + ky_grid ** 2  # [ny, nx//2+1]
        k_mag = torch.sqrt(k_sq + 1e-12)
        self.register_buffer('k_sq', k_sq)

        # --- k-space correction factor: sinc(c_ref * |k| * dt / 2) ---
        # This corrects for temporal discretization dispersion
        # sinc(x) = sin(x)/x, using unnormalized sinc
        arg = c_ref * k_mag * dt / 2.0
        # Avoid division by zero at k=0
        kappa = torch.ones_like(arg)
        nonzero = arg > 1e-10
        kappa[nonzero] = torch.sin(arg[nonzero]) / arg[nonzero]
        self.register_buffer('kappa', kappa)  # [ny, nx//2+1]

        # Precompute dt² * c_ref² * kappa for efficiency
        self.register_buffer('dt2_cref2_kappa',
                             dt * dt * c_ref * c_ref * kappa)  # [ny, nx//2+1]

        # Source kappa correction (kappa alone, for source injection)
        self.register_buffer('source_kappa', kappa)  # [ny, nx//2+1]

        # --- PML damping profile ---
        sigma_max = 1700.0 / (pml_width * dx) * 3.0
        pml_profile = self._build_pml_profile(nx, ny, pml_width, sigma_max)
        self.register_buffer('pml_damping', pml_profile)

        # --- Source AND Sensor at the SAME ROW (pulse-echo) ---
        self.transducer_row = pml_width + 1

        # --- Sensor/source lateral positions ---
        sensor_x = torch.linspace(pml_width, nx - pml_width - 1, n_elements).long()
        self.register_buffer('sensor_x', sensor_x)
        self.register_buffer('source_x', sensor_x)

    def _build_pml_profile(self, nx: int, ny: int, width: int,
                            sigma_max: float) -> torch.Tensor:
        """Build 2D PML damping coefficient field (cubic polynomial decay)."""
        damping = torch.zeros(ny, nx)
        for i in range(width):
            d = (width - i) / width
            val = sigma_max * (d ** 3)
            damping[:, i] = torch.maximum(damping[:, i], torch.tensor(val))
            damping[:, -(i + 1)] = torch.maximum(damping[:, -(i + 1)], torch.tensor(val))
            damping[i, :] = torch.maximum(damping[i, :], torch.tensor(val))
            damping[-(i + 1), :] = torch.maximum(damping[-(i + 1), :], torch.tensor(val))
        return damping.unsqueeze(0).unsqueeze(0)

    def _spectral_laplacian_kspace(self, p: torch.Tensor,
                                     c_ratio_sq: torch.Tensor) -> torch.Tensor:
        """
        k-space corrected pseudospectral Laplacian.
        
        Computes: κ · (c/c_ref)² · dt² · c_ref² · ∇²p
        
        where ∇² is computed exactly in Fourier space as -k²·P(k).
        
        For spatially varying c: apply (c/c_ref)² in spatial domain after
        the spectral Laplacian (k-Wave's approach for heterogeneous media).
        
        Args:
            p: [B, 1, ny, nx] pressure field
            c_ratio_sq: [B, 1, ny, nx] = (c(x,y) / c_ref)²
        Returns:
            result: [B, 1, ny, nx] = κ · (c/c_ref)² · dt² · c_ref² · ∇²p
        """
        # FFT of pressure field
        P = torch.fft.rfft2(p)
        
        # Spectral Laplacian with k-space correction:
        # -k² · κ · dt² · c_ref² · P(k)
        P_lap = -self.k_sq * self.dt2_cref2_kappa * P
        
        # Back to spatial domain
        lap_corrected = torch.fft.irfft2(P_lap, s=(self.ny, self.nx))
        
        # Apply spatially varying (c/c_ref)² in spatial domain
        # This is the standard k-Wave approach for heterogeneous media
        result = c_ratio_sq * lap_corrected
        
        return result

    def _apply_source_kspace(self, source_field: torch.Tensor) -> torch.Tensor:
        """
        Apply kappa correction to source injection (k-Wave consistency).
        Point source has broadband spatial spectrum; high-freq components
        need kappa correction to avoid temporal dispersion artifacts.
        """
        S = torch.fft.rfft2(source_field)
        S_corrected = self.source_kappa * S
        return torch.fft.irfft2(S_corrected, s=(self.ny, self.nx))

    def _single_step(self, p_curr: torch.Tensor, p_prev: torch.Tensor,
                     c_ratio_sq: torch.Tensor, alpha: torch.Tensor,
                     source_val: torch.Tensor) -> torch.Tensor:
        """
        Single Leapfrog time step with k-space correction.
        
        Args:
            p_curr: [B, 1, ny, nx] current pressure
            p_prev: [B, 1, ny, nx] previous pressure
            c_ratio_sq: [B, 1, ny, nx] = (c / c_ref)²
            alpha: [B, 1, ny, nx] attenuation (Np/m)
            source_val: [B] source amplitude at this time step
        Returns:
            p_next: [B, 1, ny, nx] next pressure
        """
        dt = self.dt
        dt2 = dt * dt

        # Total damping = physical attenuation + PML
        total_damping = alpha + self.pml_damping
        denom = 1.0 + total_damping * dt / 2.0
        coeff_prev = 1.0 - total_damping * dt / 2.0

        # k-space corrected spectral Laplacian term
        lap_term = self._spectral_laplacian_kspace(p_curr, c_ratio_sq)

        # Source injection with k-space correction
        source_field = torch.zeros_like(p_curr)
        source_field[:, 0, self.transducer_row, self.source_x] = source_val.unsqueeze(-1)
        source_corrected = self._apply_source_kspace(dt2 * source_field)

        # Leapfrog update
        p_next = (2.0 * p_curr - coeff_prev * p_prev
                  + lap_term + source_corrected) / denom

        return p_next

    def _run_chunk(self, p_curr: torch.Tensor, p_prev: torch.Tensor,
                   c_ratio_sq: torch.Tensor, alpha: torch.Tensor,
                   source_chunk: torch.Tensor, step_offset: int) -> tuple:
        """Run a chunk of time steps (for gradient checkpointing)."""
        chunk_len = source_chunk.size(1)
        sensor_list = []

        for i in range(chunk_len):
            p_next = self._single_step(p_curr, p_prev, c_ratio_sq, alpha,
                                        source_chunk[:, i])
            sensor_row = p_next[:, 0, self.transducer_row, :]
            sensor_data = sensor_row[:, self.sensor_x]
            sensor_list.append(sensor_data)

            p_prev = p_curr
            p_curr = p_next

        sensors = torch.stack(sensor_list, dim=2)
        return p_curr, p_prev, sensors

    def forward(self, c: torch.Tensor, alpha: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        """
        Run full wave propagation with k-space correction.
        
        Args:
            c: [B, 1, ny, nx] speed of sound (m/s)
            alpha: [B, 1, ny, nx] attenuation (Np/m)
            source: [B, n_steps] source waveform
        Returns:
            sensor_data: [B, n_elements, n_steps] pressure at sensor positions
        """
        B = c.size(0)
        device = c.device

        # CFL check (still applies for stability)
        cfl = c.max().item() * self.dt / self.dx * math.sqrt(2)
        assert cfl < 1.0, (
            f"CFL condition violated! CFL={cfl:.4f} >= 1.0. "
            f"c_max={c.max().item():.1f}, dt={self.dt:.2e}, dx={self.dx:.2e}"
        )

        # Precompute (c / c_ref)² for spatially varying medium
        c_ratio_sq = (c / self.c_ref) ** 2

        # Initialise pressure fields
        p_curr = torch.zeros(B, 1, self.ny, self.nx, device=device, dtype=c.dtype)
        p_prev = torch.zeros_like(p_curr)

        # Time stepping with gradient checkpointing
        all_sensors = []
        n_chunks = math.ceil(self.n_steps / self.checkpoint_every)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.checkpoint_every
            end = min(start + self.checkpoint_every, self.n_steps)
            source_chunk = source[:, start:end]

            if self.training and chunk_idx > 0:
                def run_fn(p_c, p_p, cr2, a_, src_, offset_):
                    return self._run_chunk(p_c, p_p, cr2, a_, src_, offset_)

                p_curr, p_prev, sensors = checkpoint(
                    run_fn, p_curr, p_prev, c_ratio_sq, alpha,
                    source_chunk, start,
                    use_reentrant=False,
                )
            else:
                p_curr, p_prev, sensors = self._run_chunk(
                    p_curr, p_prev, c_ratio_sq, alpha, source_chunk, start
                )

            all_sensors.append(sensors)

        sensor_data = torch.cat(all_sensors, dim=2)
        return sensor_data


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing AcousticLeapfrogV4 with k-space correction...")
    prop = AcousticLeapfrogV4(nx=64, ny=64, dx=2.34e-4, dt=2.0e-8,
                               n_steps=50, pml_width=10, n_elements=32,
                               c_ref=1700.0)

    B = 1
    c = torch.ones(B, 1, 64, 64) * 1540.0
    alpha = torch.zeros(B, 1, 64, 64)

    # 2MHz tone burst source (matching GT)
    f0 = 2e6
    n_cycles = 3
    dt = 2.0e-8
    t = torch.arange(50) * dt
    burst_len = n_cycles / f0
    source = torch.zeros(50)
    n_burst = int(burst_len / dt)
    tw = t[:n_burst]
    source[:n_burst] = torch.sin(2 * math.pi * f0 * tw) * \
                        torch.exp(-0.5 * ((tw - burst_len/2) / (burst_len/6))**2)
    source = source.unsqueeze(0)

    cfl = c.max().item() * dt / 2.34e-4 * math.sqrt(2)
    print(f"CFL number: {cfl:.4f} (must be < 1.0)")

    sensor_data = prop(c, alpha, source)
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Sensor data range: [{sensor_data.min():.6e}, {sensor_data.max():.6e}]")

    n_params = sum(p.numel() for p in prop.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params} (should be 0)")

    # Check gradient flow
    c.requires_grad_(True)
    sensor_data = prop(c, alpha, source)
    loss = sensor_data.sum()
    loss.backward()
    print(f"Gradient on c exists: {c.grad is not None}")
    if c.grad is not None:
        print(f"Gradient norm: {c.grad.norm():.6e}")
