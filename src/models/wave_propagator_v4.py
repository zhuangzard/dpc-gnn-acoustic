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
                 n_elements: int = 128, checkpoint_every: int = 200,
                 c_ref: float = 2000.0):
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
        # MUST match GT convention: start_col = (G - n_elements) // 2
        # For 256-grid, 128 elements: cols 64 to 191
        start_col = (nx - n_elements) // 2  # = 64
        sensor_x = torch.linspace(start_col, start_col + n_elements - 1, n_elements).long()
        self.register_buffer('sensor_x', sensor_x)
        self.register_buffer('source_x', sensor_x)

        # Source mask for differentiable Dirichlet injection
        # 1.0 at source positions, 0.0 elsewhere
        smask = torch.zeros(1, 1, ny, nx)
        smask[0, 0, self.transducer_row, sensor_x] = 1.0
        self.register_buffer('source_mask', smask)

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
                                     c_ratio_sq: torch.Tensor,
                                     rho_inv: torch.Tensor = None) -> torch.Tensor:
        """
        k-space corrected pseudospectral Laplacian with optional density.
        
        Without density (rho_inv=None):
            κ · (c/c_ref)² · dt² · c_ref² · ∇²p
        
        With density (rho_inv provided):
            κ · ρ · c² · dt² · ∇·(1/ρ · ∇p)
            Approximated as: κ · (c/c_ref)² · dt² · c_ref² · ∇²p
            (density effect handled via acoustic impedance in reflection)
        
        For spatially varying c: apply (c/c_ref)² in spatial domain after
        the spectral Laplacian (k-Wave's approach for heterogeneous media).
        
        Args:
            p: [B, 1, ny, nx] pressure field
            c_ratio_sq: [B, 1, ny, nx] = (c(x,y) / c_ref)²
            rho_inv: [B, 1, ny, nx] = 1/ρ(x,y) (optional, for density effects)
        Returns:
            result: [B, 1, ny, nx]
        """
        if rho_inv is not None:
            # Full heterogeneous equation: ρ·c²·∇·(∇p/ρ)
            # Step 1: compute ∇p/ρ in spectral domain, then ∇· in spectral domain
            # Approximate: compute ∇²(p) spectrally, then modulate by ρ effect
            # k-Wave approach: ∇·(1/ρ · ∇p) via spectral derivatives
            
            # Compute spectral gradient components of p
            P = torch.fft.rfft2(p)
            # ikx and iky wavenumber grids
            kx = torch.fft.rfftfreq(self.nx, d=self.dx).to(p.device) * 2.0 * math.pi
            ky = torch.fft.fftfreq(self.ny, d=self.dx).to(p.device) * 2.0 * math.pi
            ky_g, kx_g = torch.meshgrid(ky, kx, indexing='ij')
            
            # ∂p/∂x and ∂p/∂y in spectral domain
            dpdx = torch.fft.irfft2(1j * kx_g * P, s=(self.ny, self.nx))
            dpdy = torch.fft.irfft2(1j * ky_g * P, s=(self.ny, self.nx))
            
            # (1/ρ) · ∇p
            rho_inv_dpdx = rho_inv * dpdx
            rho_inv_dpdy = rho_inv * dpdy
            
            # ∇·((1/ρ)·∇p) in spectral domain
            Fx = torch.fft.rfft2(rho_inv_dpdx)
            Fy = torch.fft.rfft2(rho_inv_dpdy)
            div_F = torch.fft.irfft2(1j * kx_g * Fx + 1j * ky_g * Fy, 
                                      s=(self.ny, self.nx))
            
            # Apply kappa correction and c²·ρ factor
            # result = κ · dt² · c² · ρ · ∇·(∇p/ρ)
            # But we precomputed dt2_cref2_kappa, so need to adjust
            # Simple approach: apply kappa to div_F in spectral domain
            Div = torch.fft.rfft2(div_F)
            Div_corrected = self.kappa * Div
            div_corrected = torch.fft.irfft2(Div_corrected, s=(self.ny, self.nx))
            
            dt2 = self.dt * self.dt
            # c_ratio_sq already contains (c/c_ref)², multiply by c_ref² and dt²
            result = c_ratio_sq * (self.c_ref ** 2) * dt2 * div_corrected
            
            return result
        else:
            # Original: no density
            P = torch.fft.rfft2(p)
            P_lap = -self.k_sq * self.dt2_cref2_kappa * P
            lap_corrected = torch.fft.irfft2(P_lap, s=(self.ny, self.nx))
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
                     source_val: torch.Tensor, rho_inv: torch.Tensor = None) -> torch.Tensor:
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
        lap_term = self._spectral_laplacian_kspace(p_curr, c_ratio_sq, rho_inv)

        # Leapfrog update (without source)
        p_next = (2.0 * p_curr - coeff_prev * p_prev
                  + lap_term) / denom

        # Source injection: DIRICHLET mode (matching k-Wave's source.p_mode="dirichlet")
        # k-Wave REPLACES pressure at source positions with signal value.
        # Using mask-based blending to preserve autograd gradient flow.
        #
        # CRITICAL: Only inject when source_val != 0 (during burst).
        # After burst, source_val=0 and we must let the propagated field
        # pass through — otherwise sensor (same row) only sees source signal,
        # not the c-dependent propagated wavefield, killing all gradients.
        source_field = torch.zeros_like(p_next)
        source_field[:, 0, self.transducer_row, self.source_x] = source_val.unsqueeze(-1)
        # active_mask: only apply Dirichlet where source is nonzero
        is_active = (source_val.abs() > 1e-12).float()  # [B]
        active_mask = self.source_mask * is_active.view(-1, 1, 1, 1)  # [B,1,ny,nx]
        p_next = p_next * (1.0 - active_mask) + source_field * active_mask

        return p_next

    def _run_chunk(self, p_curr: torch.Tensor, p_prev: torch.Tensor,
                   c_ratio_sq: torch.Tensor, alpha: torch.Tensor,
                   source_chunk: torch.Tensor, step_offset: int,
                   rho_inv: torch.Tensor = None) -> tuple:
        """Run a chunk of time steps (for gradient checkpointing)."""
        chunk_len = source_chunk.size(1)
        sensor_list = []

        for i in range(chunk_len):
            p_next = self._single_step(p_curr, p_prev, c_ratio_sq, alpha,
                                        source_chunk[:, i], rho_inv)
            sensor_row = p_next[:, 0, self.transducer_row, :]
            sensor_data = sensor_row[:, self.sensor_x]
            sensor_list.append(sensor_data)

            p_prev = p_curr
            p_curr = p_next

        sensors = torch.stack(sensor_list, dim=2)
        return p_curr, p_prev, sensors

    def forward(self, c: torch.Tensor, alpha: torch.Tensor,
                source: torch.Tensor, rho: torch.Tensor = None) -> torch.Tensor:
        """
        Run full wave propagation with k-space correction.
        
        Args:
            c: [B, 1, ny, nx] speed of sound (m/s)
            alpha: [B, 1, ny, nx] attenuation (Np/m)
            source: [B, n_steps] source waveform
            rho: [B, 1, ny, nx] density (kg/m³), optional
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
        
        # Precompute 1/rho if density is provided
        rho_inv = (1.0 / rho) if rho is not None else None

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
                def run_fn(p_c, p_p, cr2, a_, src_, offset_, ri_):
                    return self._run_chunk(p_c, p_p, cr2, a_, src_, offset_, ri_)

                p_curr, p_prev, sensors = checkpoint(
                    run_fn, p_curr, p_prev, c_ratio_sq, alpha,
                    source_chunk, start, rho_inv,
                    use_reentrant=False,
                )
            else:
                p_curr, p_prev, sensors = self._run_chunk(
                    p_curr, p_prev, c_ratio_sq, alpha, source_chunk, start, rho_inv
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
