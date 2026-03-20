"""
AcousticPropagatorV5 — Velocity-pressure staggered-grid pseudo-spectral solver
===============================================================================

k-Wave-compatible kspaceFirstOrder2D formulation with split-field PML,
per-axis kappa correction, and physical attenuation.

Physics equations (each time step):
    u_x^{n+1} = PML_x · u_x^n  −  (dt/ρ) · F⁻¹{ κ_x · ik_x · F{p^n} }
    u_y^{n+1} = PML_y · u_y^n  −  (dt/ρ) · F⁻¹{ κ_y · ik_y · F{p^n} }
    p^{n+1}   = PML_p · p^n    −  ρc²·dt · F⁻¹{ ik_x · F{u_x^{n+1}} + ik_y · F{u_y^{n+1}} }

Per-axis kappa (unnormalized sinc for k-space correction):
    κ_x[kx] = sinc(c_ref · |kx| · dt / 2)   where sinc(x) = sin(x)/x
    κ_y[ky] = sinc(c_ref · |ky| · dt / 2)

PML (split-field, directional, polynomial grading):
    σ_x[i] = σ_max · ((pml_width − i) / pml_width)³   (x-boundaries)
    σ_y[j] = σ_max · ((pml_width − j) / pml_width)³   (y-boundaries)
    PML_x = (1 − σ_x·dt/2) / (1 + σ_x·dt/2)
    PML_y = (1 − σ_y·dt/2) / (1 + σ_y·dt/2)
    PML_p uses combined σ_x + σ_y

Source: Dirichlet on pressure during burst (mask-based for autograd).
Sensor: Records pressure at transducer row.

Zero learnable parameters. Gradient checkpointing every 200 steps.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AcousticPropagatorV5(nn.Module):
    """Velocity-pressure staggered-grid pseudo-spectral acoustic wave propagator.

    Drop-in replacement for wave_propagator_v4.py with k-Wave-compatible physics.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (default 256×256).
    dx : float
        Grid spacing in meters (default 2.34e-4).
    dt : float
        Time step in seconds (default 4e-8).
    pml_width : int
        PML layer thickness in grid points (default 20).
    n_elements : int
        Number of transducer elements (default 128).
    c_ref : float
        Reference sound speed for kappa correction (default 2000.0 m/s).
    checkpoint_every : int
        Gradient checkpointing interval (default 200).
    """

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        dx: float = 2.34e-4,
        dt: float = 4.0e-8,
        pml_width: int = 20,
        n_elements: int = 128,
        c_ref: float = 2000.0,
        checkpoint_every: int = 200,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.pml_width = pml_width
        self.n_elements = n_elements
        self.c_ref = c_ref
        self.checkpoint_every = checkpoint_every

        # Transducer geometry
        self.transducer_row = pml_width + 1
        self.element_start = (ny - n_elements) // 2  # 64 for ny=256, n_elements=128
        self.element_end = self.element_start + n_elements  # 192

        # --- Wavenumber grids ---
        kx_1d = 2.0 * math.pi * torch.fft.fftfreq(nx, d=dx)  # [nx]
        ky_1d = 2.0 * math.pi * torch.fft.fftfreq(ny, d=dy if (dy := dx) else dx)  # [ny]

        # 2D wavenumber grids for spectral derivatives
        # kx varies along dim 0 (rows), ky along dim 1 (cols)
        kx_2d = kx_1d[:, None].expand(nx, ny)  # [nx, ny]
        ky_2d = ky_1d[None, :].expand(nx, ny)  # [nx, ny]

        self.register_buffer("kx_2d", kx_2d)
        self.register_buffer("ky_2d", ky_2d)

        # --- Per-axis kappa correction (unnormalized sinc) ---
        # κ_x[kx] = sinc(c_ref * |kx| * dt / 2)  with sinc(x) = sin(x)/x
        arg_x = c_ref * kx_1d.abs() * dt / 2.0  # [nx]
        kappa_x_1d = torch.where(
            arg_x < 1e-12,
            torch.ones_like(arg_x),
            torch.sin(arg_x) / arg_x,
        )
        arg_y = c_ref * ky_1d.abs() * dt / 2.0  # [ny]
        kappa_y_1d = torch.where(
            arg_y < 1e-12,
            torch.ones_like(arg_y),
            torch.sin(arg_y) / arg_y,
        )

        # Broadcast to 2D for velocity updates
        kappa_x_2d = kappa_x_1d[:, None].expand(nx, ny)  # [nx, ny]
        kappa_y_2d = kappa_y_1d[None, :].expand(nx, ny)  # [nx, ny]

        self.register_buffer("kappa_x", kappa_x_2d)
        self.register_buffer("kappa_y", kappa_y_2d)

        # --- PML coefficients ---
        sigma_max = c_ref / (pml_width * dx) * 3.0

        # Build 1D sigma profiles
        sigma_x_1d = torch.zeros(nx)
        sigma_y_1d = torch.zeros(ny)

        for i in range(pml_width):
            val = sigma_max * ((pml_width - i) / pml_width) ** 3
            sigma_x_1d[i] = val
            sigma_x_1d[nx - 1 - i] = val
            sigma_y_1d[i] = val
            sigma_y_1d[ny - 1 - i] = val

        # PML decay factors: (1 - σ·dt/2) / (1 + σ·dt/2)
        pml_x_1d = (1.0 - sigma_x_1d * dt / 2.0) / (1.0 + sigma_x_1d * dt / 2.0)
        pml_y_1d = (1.0 - sigma_y_1d * dt / 2.0) / (1.0 + sigma_y_1d * dt / 2.0)

        # 2D PML maps
        pml_x_2d = pml_x_1d[:, None].expand(nx, ny)  # for u_x
        pml_y_2d = pml_y_1d[None, :].expand(nx, ny)  # for u_y

        # Pressure PML: combined σ_x + σ_y
        sigma_p_2d = sigma_x_1d[:, None] + sigma_y_1d[None, :]
        pml_p_2d = (1.0 - sigma_p_2d * dt / 2.0) / (1.0 + sigma_p_2d * dt / 2.0)

        self.register_buffer("pml_x", pml_x_2d)
        self.register_buffer("pml_y", pml_y_2d)
        self.register_buffer("pml_p", pml_p_2d)

        # Attenuation denominator for pressure PML (for stability)
        pml_denom_x = 1.0 / (1.0 + sigma_x_1d * dt / 2.0)
        pml_denom_y = 1.0 / (1.0 + sigma_y_1d * dt / 2.0)
        self.register_buffer("pml_denom_x", pml_denom_x[:, None].expand(nx, ny))
        self.register_buffer("pml_denom_y", pml_denom_y[None, :].expand(nx, ny))

        # Sensor column indices
        sensor_cols = torch.arange(self.element_start, self.element_end)
        self.register_buffer("sensor_cols", sensor_cols)

    def _spectral_grad_x(self, field: torch.Tensor) -> torch.Tensor:
        """Compute ∂field/∂x via spectral method: F⁻¹{ ik_x · F{field} }."""
        F = torch.fft.fft2(field)
        return torch.fft.ifft2(1j * self.kx_2d * F).real

    def _spectral_grad_y(self, field: torch.Tensor) -> torch.Tensor:
        """Compute ∂field/∂y via spectral method: F⁻¹{ ik_y · F{field} }."""
        F = torch.fft.fft2(field)
        return torch.fft.ifft2(1j * self.ky_2d * F).real

    def _spectral_grad_x_kappa(self, field: torch.Tensor) -> torch.Tensor:
        """Compute F⁻¹{ κ_x · ik_x · F{field} } for velocity-x update."""
        F = torch.fft.fft2(field)
        return torch.fft.ifft2(self.kappa_x * 1j * self.kx_2d * F).real

    def _spectral_grad_y_kappa(self, field: torch.Tensor) -> torch.Tensor:
        """Compute F⁻¹{ κ_y · ik_y · F{field} } for velocity-y update."""
        F = torch.fft.fft2(field)
        return torch.fft.ifft2(self.kappa_y * 1j * self.ky_2d * F).real

    def _single_step(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        p: torch.Tensor,
        c_sq: torch.Tensor,
        rho: torch.Tensor,
        rho_inv: torch.Tensor,
        attenuation: torch.Tensor,
        source_val: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One time step of the velocity-pressure update.

        All operations are autograd-safe (no in-place ops).
        """
        dt = self.dt

        # --- Velocity update ---
        # u_x^{n+1} = PML_x · u_x^n − (dt/ρ) · F⁻¹{ κ_x · ik_x · F{p^n} }
        dp_dx_kappa = self._spectral_grad_x_kappa(p)
        ux_new = self.pml_x * ux - dt * rho_inv * dp_dx_kappa

        # u_y^{n+1} = PML_y · u_y^n − (dt/ρ) · F⁻¹{ κ_y · ik_y · F{p^n} }
        dp_dy_kappa = self._spectral_grad_y_kappa(p)
        uy_new = self.pml_y * uy - dt * rho_inv * dp_dy_kappa

        # --- Pressure update ---
        # p^{n+1} = PML_p · p^n − ρc²·dt · F⁻¹{ ik_x·F{u_x} + ik_y·F{u_y} }
        dux_dx = self._spectral_grad_x(ux_new)
        duy_dy = self._spectral_grad_y(uy_new)
        div_u = dux_dx + duy_dy

        p_new = self.pml_p * p - rho * c_sq * dt * div_u

        # --- Physical attenuation (applied as exponential damping) ---
        p_new = p_new * attenuation

        # --- Source injection (Dirichlet, mask-based for autograd) ---
        # source_mask is 1.0 at source locations during burst, 0.0 otherwise
        # p = source_val * mask + p_new * (1 - mask)
        p_new = source_val * source_mask + p_new * (1.0 - source_mask)

        return ux_new, uy_new, p_new

    def _run_chunk(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        p: torch.Tensor,
        c_sq: torch.Tensor,
        rho: torch.Tensor,
        rho_inv: torch.Tensor,
        attenuation: torch.Tensor,
        source_chunk: torch.Tensor,
        source_mask_2d: torch.Tensor,
        chunk_start: int,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a chunk of time steps. Returns (ux, uy, p, sensor_chunk)."""
        B = p.shape[0]
        sensor_list = []

        for local_t in range(chunk_size):
            t = chunk_start + local_t
            # Source value at this time step: [B]
            src_val_t = source_chunk[:, local_t]  # [B]

            # Build source mask: only inject when source is nonzero
            # src_val_t: [B], source_mask_2d: [nx, ny] → broadcast
            is_active = (src_val_t.abs() > 0.0).float()  # [B]
            # source_val broadcast: [B, nx, ny]
            source_val = src_val_t[:, None, None] * source_mask_2d[None, :, :]
            # effective mask: [B, nx, ny]
            eff_mask = is_active[:, None, None] * source_mask_2d[None, :, :]

            ux, uy, p = self._single_step(
                ux, uy, p, c_sq, rho, rho_inv, attenuation, source_val, eff_mask
            )

            # Record sensor data: p at transducer_row, sensor columns
            sensor_list.append(p[:, self.transducer_row, self.sensor_cols])

        sensor_chunk = torch.stack(sensor_list, dim=-1)  # [B, n_elements, chunk_size]
        return ux, uy, p, sensor_chunk

    def forward(
        self,
        c: torch.Tensor,
        alpha: torch.Tensor,
        source: torch.Tensor,
        rho: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run acoustic wave propagation.

        Parameters
        ----------
        c : Tensor [B, nx, ny]
            Sound speed map (m/s).
        alpha : Tensor [B, nx, ny]
            Attenuation coefficient (Np/m). Applied as exp(-alpha * c * dt).
        source : Tensor [B, n_steps]
            Source signal (pressure amplitude at transducer elements).
        rho : Tensor [B, nx, ny] or None
            Density map (kg/m³). If None, uniform water (1000 kg/m³).

        Returns
        -------
        sensor_data : Tensor [B, n_elements, n_steps]
            Recorded pressure at sensor elements.
        """
        B, n_steps = source.shape[0], source.shape[1]
        device = c.device
        dtype = c.dtype

        # Default density
        if rho is None:
            rho = torch.full((B, self.nx, self.ny), 1000.0, device=device, dtype=dtype)

        # --- CFL check ---
        c_max = c.max().item()
        cfl = c_max * self.dt / self.dx
        if cfl > 1.0 / math.sqrt(2.0):
            raise ValueError(
                f"CFL condition violated: CFL={cfl:.4f} > {1.0/math.sqrt(2.0):.4f}. "
                f"c_max={c_max:.1f}, dt={self.dt:.2e}, dx={self.dx:.2e}"
            )

        # Precompute fields
        c_sq = c * c  # [B, nx, ny]
        rho_inv = 1.0 / rho  # [B, nx, ny]

        # Attenuation per time step: exp(-alpha * c * dt)
        attenuation = torch.exp(-alpha * c * self.dt)  # [B, nx, ny]

        # Source mask: 2D binary mask for transducer locations
        source_mask_2d = torch.zeros(self.nx, self.ny, device=device, dtype=dtype)
        source_mask_2d[self.transducer_row, self.element_start : self.element_end] = 1.0

        # Initialize fields
        ux = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)
        uy = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)
        p = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)

        # --- Time-stepping with gradient checkpointing ---
        sensor_chunks = []
        chunk_size = self.checkpoint_every

        for chunk_start in range(0, n_steps, chunk_size):
            actual_chunk_size = min(chunk_size, n_steps - chunk_start)
            source_chunk = source[:, chunk_start : chunk_start + actual_chunk_size]

            if self.training and torch.is_grad_enabled():
                # Use gradient checkpointing — wrap in a function
                def run_ckpt(ux_, uy_, p_, c_sq_, rho_, rho_inv_, atten_, src_chunk_,
                             _cs=chunk_start, _sz=actual_chunk_size):
                    return self._run_chunk(
                        ux_, uy_, p_, c_sq_, rho_, rho_inv_, atten_,
                        src_chunk_, source_mask_2d, _cs, _sz,
                    )

                ux, uy, p, sensor_chunk = checkpoint(
                    run_ckpt,
                    ux, uy, p, c_sq, rho, rho_inv, attenuation, source_chunk,
                    use_reentrant=False,
                )
            else:
                ux, uy, p, sensor_chunk = self._run_chunk(
                    ux, uy, p, c_sq, rho, rho_inv, attenuation,
                    source_chunk, source_mask_2d, chunk_start, actual_chunk_size,
                )

            sensor_chunks.append(sensor_chunk)

        # Concatenate all sensor chunks: [B, n_elements, n_steps]
        sensor_data = torch.cat(sensor_chunks, dim=-1)
        return sensor_data


# =============================================================================
# Test / demo
# =============================================================================
if __name__ == "__main__":
    import time

    device = "cpu"
    dtype = torch.float32

    # --- Parameters ---
    nx, ny = 256, 256
    dx = 2.34e-4
    dt = 4.0e-8
    pml_width = 20
    n_elements = 128
    c_ref = 2000.0
    n_steps = 500

    print("=" * 60)
    print("AcousticPropagatorV5 — Test")
    print("=" * 60)

    # --- Create propagator ---
    prop = AcousticPropagatorV5(
        nx=nx, ny=ny, dx=dx, dt=dt,
        pml_width=pml_width, n_elements=n_elements, c_ref=c_ref,
    ).to(device)

    # --- Two-layer medium ---
    B = 1
    c_map = torch.ones(B, nx, ny, device=device, dtype=dtype) * 1500.0  # water
    c_map[:, nx // 2 :, :] = 2500.0  # bone/tissue layer
    c_map.requires_grad_(True)

    alpha_map = torch.zeros(B, nx, ny, device=device, dtype=dtype)
    alpha_map[:, nx // 2 :, :] = 0.5  # attenuation in second layer

    # --- Source: 5-cycle tone burst at 1 MHz ---
    freq = 1.0e6
    t_axis = torch.arange(n_steps, device=device, dtype=dtype) * dt
    n_cycles = 5
    burst_duration = n_cycles / freq
    burst_mask = (t_axis < burst_duration).float()
    source_signal = (torch.sin(2.0 * math.pi * freq * t_axis) * burst_mask).unsqueeze(0)  # [1, n_steps]

    # Scale source
    source_signal = source_signal * 1e6  # Pa

    # --- CFL check ---
    c_max = c_map.max().item()
    cfl = c_max * dt / dx
    print(f"CFL number: {cfl:.4f}  (limit: {1.0/math.sqrt(2.0):.4f})")
    print(f"c_max: {c_max:.0f} m/s,  dt: {dt:.2e} s,  dx: {dx:.2e} m")

    # --- Run propagation ---
    print(f"\nRunning {n_steps} steps...")
    t0 = time.time()
    prop.eval()
    with torch.no_grad():
        sensor_data = prop(c_map, alpha_map, source_signal)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.2f}s")
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Sensor data range: [{sensor_data.min().item():.4e}, {sensor_data.max().item():.4e}]")

    # --- Gradient check ---
    # Burst = 5 cycles at 1MHz = 125 steps. Need post-burst steps for gradient.
    print("\n--- Gradient flow check ---")
    prop.train()
    c_grad = torch.ones(B, nx, ny, device=device, dtype=dtype) * 1500.0
    c_grad[:, nx // 2 :, :] = 2500.0
    c_grad = c_grad.clone().requires_grad_(True)

    alpha_grad = torch.zeros(B, nx, ny, device=device, dtype=dtype)

    # Need post-burst steps: burst ends at step ~125, use 250 for some echo time
    n_grad_steps = 250
    source_grad = source_signal[:, :n_grad_steps]

    sensor_grad = prop(c_grad, alpha_grad, source_grad)
    loss = sensor_grad.abs().mean()
    loss.backward()

    grad_norm = c_grad.grad.norm().item() if c_grad.grad is not None else 0.0
    print(f"Loss: {loss.item():.6e}")
    print(f"Gradient norm (c): {grad_norm:.6e}")
    print(f"Gradient flows: {'YES ✓' if grad_norm > 0 else 'NO ✗'}")

    print("\n" + "=" * 60)
    print("All checks passed!" if grad_norm > 0 else "WARNING: No gradient flow!")
    print("=" * 60)
