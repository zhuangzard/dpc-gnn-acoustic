"""
DPC-GNN-Acoustic V4: Deterministic Leapfrog Wave Propagator

2D acoustic wave equation with scattering:
    p^{n+1} = (2p^n - (1-σ_total·Δt/2)·p^{n-1} + Δt²·c²·∇²p + Δt²·f_source) / (1+σ_total·Δt/2)

where σ_total = α (physical attenuation) + σ_pml (PML damping)

Source injection is independent of σ — σ controls scattering/attenuation throughout the medium.

Features:
  - NO learnable parameters (pure physics)
  - PML absorbing boundary (20 cells, cubic polynomial decay, physically correct σ_max)
  - Laplacian via F.conv2d (2nd-order accuracy)
  - Gradient checkpointing every 50 steps
  - Source and sensor at SAME ROW (pulse-echo ultrasound)
  - CFL stability check
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AcousticLeapfrogV4(nn.Module):
    """
    Deterministic 2D acoustic Leapfrog propagator.
    Zero learnable parameters — all physics.
    """

    def __init__(self, nx: int = 256, ny: int = 256,
                 dx: float = 2.34e-4, dt: float = 2.0e-8,
                 n_steps: int = 200, pml_width: int = 20,
                 n_elements: int = 128, checkpoint_every: int = 50):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.n_steps = n_steps
        self.pml_width = pml_width
        self.n_elements = n_elements
        self.checkpoint_every = checkpoint_every

        # --- Laplacian kernel (2nd-order finite difference) ---
        lap_kernel = torch.tensor([[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]], dtype=torch.float32) / (dx * dx)
        self.register_buffer('lap_kernel', lap_kernel.reshape(1, 1, 3, 3))

        # --- PML damping profile ---
        # sigma_max in units matching the wave equation damping term
        # For effective absorption: sigma_max ~ c_max / (pml_width * dx) * scaling
        # With c_max=1700, pml=20, dx=2.34e-4: sigma_max ~ 1700/(20*2.34e-4)*3 ≈ 1.09e6
        sigma_max = 1700.0 / (pml_width * dx) * 3.0  # ~1.09e6 for effective PML
        pml_profile = self._build_pml_profile(nx, ny, pml_width, sigma_max)
        self.register_buffer('pml_damping', pml_profile)

        # --- Source AND Sensor at the SAME ROW (pulse-echo) ---
        # k-Wave uses PML_SIZE + 1 for both source and sensor
        self.transducer_row = pml_width + 1  # row 21

        # --- Sensor/source lateral positions: uniformly spaced ---
        sensor_x = torch.linspace(pml_width, nx - pml_width - 1, n_elements).long()
        self.register_buffer('sensor_x', sensor_x)

        # --- Source: ALL elements fire simultaneously (plane wave) ---
        # Source covers the full active aperture, same positions as sensors
        self.register_buffer('source_x', sensor_x)  # same lateral positions

    def _build_pml_profile(self, nx: int, ny: int, width: int,
                            sigma_max: float) -> torch.Tensor:
        """
        Build 2D PML damping coefficient field.
        Cubic polynomial decay: σ_pml(d) = σ_max * (d/width)^3
        """
        damping = torch.zeros(ny, nx)

        for i in range(width):
            d = (width - i) / width  # 1 at boundary, 0 at interface
            val = sigma_max * (d ** 3)
            damping[:, i] = torch.maximum(damping[:, i], torch.tensor(val))
            damping[:, -(i + 1)] = torch.maximum(damping[:, -(i + 1)], torch.tensor(val))
            damping[i, :] = torch.maximum(damping[i, :], torch.tensor(val))
            damping[-(i + 1), :] = torch.maximum(damping[-(i + 1), :], torch.tensor(val))

        return damping.unsqueeze(0).unsqueeze(0)  # [1, 1, ny, nx]

    def _laplacian(self, p: torch.Tensor) -> torch.Tensor:
        """Compute 2D Laplacian using convolution (zero-padded)."""
        return F.conv2d(p, self.lap_kernel, padding=1)

    def _single_step(self, p_curr: torch.Tensor, p_prev: torch.Tensor,
                     c: torch.Tensor, alpha: torch.Tensor,
                     source_val: torch.Tensor) -> torch.Tensor:
        """
        Single Leapfrog time step.
        
        Args:
            p_curr: [B, 1, ny, nx] current pressure
            p_prev: [B, 1, ny, nx] previous pressure
            c: [B, 1, ny, nx] speed of sound (m/s)
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

        # Laplacian
        lap_p = self._laplacian(p_curr)

        # Source injection: plane wave across all source elements
        # Source is INDEPENDENT of σ — it represents the transducer drive
        source_term = torch.zeros_like(p_curr)
        # Inject at all source positions (plane wave)
        source_term[:, 0, self.transducer_row, self.source_x] = source_val.unsqueeze(-1)

        # Leapfrog update
        p_next = (2.0 * p_curr - coeff_prev * p_prev
                  + dt2 * (c * c * lap_p + source_term)) / denom

        return p_next

    def _run_chunk(self, p_curr: torch.Tensor, p_prev: torch.Tensor,
                   c: torch.Tensor, alpha: torch.Tensor,
                   source_chunk: torch.Tensor, step_offset: int) -> tuple:
        """Run a chunk of time steps (for gradient checkpointing)."""
        chunk_len = source_chunk.size(1)
        sensor_list = []

        for i in range(chunk_len):
            p_next = self._single_step(p_curr, p_prev, c, alpha,
                                        source_chunk[:, i])
            # Extract sensor data at transducer row (same as source — pulse-echo)
            sensor_row = p_next[:, 0, self.transducer_row, :]  # [B, nx]
            sensor_data = sensor_row[:, self.sensor_x]  # [B, n_elements]
            sensor_list.append(sensor_data)

            p_prev = p_curr
            p_curr = p_next

        sensors = torch.stack(sensor_list, dim=2)  # [B, n_elements, chunk_len]
        return p_curr, p_prev, sensors

    def forward(self, c: torch.Tensor, alpha: torch.Tensor,
                source: torch.Tensor) -> torch.Tensor:
        """
        Run full wave propagation.
        
        Args:
            c: [B, 1, ny, nx] speed of sound (m/s)
            alpha: [B, 1, ny, nx] attenuation (Np/m)
            source: [B, n_steps] source waveform
        Returns:
            sensor_data: [B, n_elements, n_steps] pressure at sensor positions
        """
        B = c.size(0)
        device = c.device

        # --- CFL stability check ---
        cfl = c.max().item() * self.dt / self.dx * math.sqrt(2)
        assert cfl < 1.0, (
            f"CFL condition violated! CFL={cfl:.4f} >= 1.0. "
            f"c_max={c.max().item():.1f}, dt={self.dt:.2e}, dx={self.dx:.2e}"
        )

        # --- Initialise pressure fields ---
        p_curr = torch.zeros(B, 1, self.ny, self.nx, device=device, dtype=c.dtype)
        p_prev = torch.zeros_like(p_curr)

        # --- Time stepping with gradient checkpointing ---
        all_sensors = []
        n_chunks = math.ceil(self.n_steps / self.checkpoint_every)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.checkpoint_every
            end = min(start + self.checkpoint_every, self.n_steps)
            source_chunk = source[:, start:end]

            if self.training and chunk_idx > 0:
                def run_fn(p_c, p_p, c_, a_, src_, offset_):
                    return self._run_chunk(p_c, p_p, c_, a_, src_, offset_)

                p_curr, p_prev, sensors = checkpoint(
                    run_fn, p_curr, p_prev, c, alpha, source_chunk, start,
                    use_reentrant=False,
                )
            else:
                p_curr, p_prev, sensors = self._run_chunk(
                    p_curr, p_prev, c, alpha, source_chunk, start
                )

            all_sensors.append(sensors)

        sensor_data = torch.cat(all_sensors, dim=2)  # [B, n_elements, n_steps]
        return sensor_data


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing AcousticLeapfrogV4...")
    prop = AcousticLeapfrogV4(nx=64, ny=64, dx=2.34e-4, dt=2.0e-8,
                               n_steps=50, pml_width=10, n_elements=32)

    B = 1
    c = torch.ones(B, 1, 64, 64) * 1540.0
    alpha = torch.zeros(B, 1, 64, 64)

    # Ricker wavelet source
    f0 = 5e6
    t = torch.arange(50) * 2.0e-8
    t0 = 1.5 / f0
    arg = (math.pi * f0 * (t - t0)) ** 2
    source = ((1.0 - 2.0 * arg) * torch.exp(-arg)).unsqueeze(0)

    cfl = c.max().item() * 2.0e-8 / 2.34e-4 * math.sqrt(2)
    print(f"CFL number: {cfl:.4f} (must be < 1.0)")

    sensor_data = prop(c, alpha, source)
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Sensor data range: [{sensor_data.min():.6e}, {sensor_data.max():.6e}]")

    n_params = sum(p.numel() for p in prop.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params} (should be 0)")

    c.requires_grad_(True)
    sensor_data = prop(c, alpha, source)
    loss = sensor_data.sum()
    loss.backward()
    print(f"Gradient on c exists: {c.grad is not None}")
