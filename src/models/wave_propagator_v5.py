"""
AcousticPropagatorV5 — k-Wave-compatible kspaceFirstOrder2D solver
===================================================================

Exact reimplementation of k-Wave's kspaceFirstOrder2D algorithm.

State variables: ux_sgx, uy_sgy, rhox, rhoy  (split-field density, NOT p)

Physics equations (each time step):
    # Equation of state
    p = c² · (rhox + rhoy)

    # Velocity update (PML = exp(-σ·dt/2), applied twice multiplicatively)
    dp_dx = F⁻¹{ κ_x · ik_x · F{p} }
    dp_dy = F⁻¹{ κ_y · ik_y · F{p} }
    ux_sgx = pml_x_sgx · (pml_x_sgx · ux_sgx − dt/ρ₀_sgx · dp_dx)
    uy_sgy = pml_y_sgy · (pml_y_sgy · uy_sgy − dt/ρ₀_sgy · dp_dy)

    # Density update (split-field PML, directional)
    dux_dx = F⁻¹{ κ_x · ik_x · F{ux_sgx} }
    duy_dy = F⁻¹{ κ_y · ik_y · F{uy_sgy} }
    rhox = pml_x · (pml_x · rhox − dt · ρ₀ · dux_dx)
    rhoy = pml_y · (pml_y · rhoy − dt · ρ₀ · duy_dy)

    # Source injection (Dirichlet into rhox/rhoy)
    rhox[mask] = source_val / (2·c²)
    rhoy[mask] = source_val / (2·c²)

    # Recompute p and record sensor
    p = c² · (rhox + rhoy)

Key differences from previous V5:
    - Split-field rhox/rhoy instead of monolithic p  (BUG 1 fix)
    - exp(-σ·dt/2) PML applied twice, not Crank-Nicolson  (BUG 2 fix)
    - No combined pressure PML; PML on rhox (x-only) and rhoy (y-only)  (BUG 3 fix)
    - Staggered density interpolation (rho0_sgx, rho0_sgy)  (BUG 4 fix)
    - Source injection into rhox/rhoy scaled by 1/(2c²)  (BUG 5 fix)

Zero learnable parameters. Gradient checkpointing every 200 steps.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AcousticPropagatorV5(nn.Module):
    """k-Wave-compatible kspaceFirstOrder2D acoustic wave propagator.

    Drop-in replacement with corrected split-field PML, staggered density,
    and proper source injection.

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
        ky_1d = 2.0 * math.pi * torch.fft.fftfreq(ny, d=dx)  # [ny]

        # 2D wavenumber grids: kx varies along dim -2 (rows), ky along dim -1 (cols)
        kx_2d = kx_1d[:, None].expand(nx, ny)  # [nx, ny]
        ky_2d = ky_1d[None, :].expand(nx, ny)  # [nx, ny]

        self.register_buffer("kx_2d", kx_2d)
        self.register_buffer("ky_2d", ky_2d)

        # --- Staggered grid shift factors (k-Wave ddx_k_shift_pos/neg) ---
        # k-Wave uses shifted spectral derivatives for staggered grid:
        #   ddx_k_shift_pos = 1j*kx * exp(+1j*kx*dx/2)  (velocity: p→u, forward stagger)
        #   ddx_k_shift_neg = 1j*kx * exp(-1j*kx*dx/2)  (density: u→rho, backward stagger)
        # These are COMPLEX tensors
        shift_pos_x = torch.exp(1j * kx_1d * dx / 2.0)[:, None].expand(nx, ny)  # [nx, ny]
        shift_neg_x = torch.exp(-1j * kx_1d * dx / 2.0)[:, None].expand(nx, ny)
        shift_pos_y = torch.exp(1j * ky_1d * dx / 2.0)[None, :].expand(nx, ny)  # [nx, ny]
        shift_neg_y = torch.exp(-1j * ky_1d * dx / 2.0)[None, :].expand(nx, ny)

        self.register_buffer("shift_pos_x", shift_pos_x)
        self.register_buffer("shift_neg_x", shift_neg_x)
        self.register_buffer("shift_pos_y", shift_pos_y)
        self.register_buffer("shift_neg_y", shift_neg_y)

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

        # Broadcast to 2D
        kappa_x_2d = kappa_x_1d[:, None].expand(nx, ny)  # [nx, ny]
        kappa_y_2d = kappa_y_1d[None, :].expand(nx, ny)  # [nx, ny]

        self.register_buffer("kappa_x", kappa_x_2d)
        self.register_buffer("kappa_y", kappa_y_2d)

        # --- PML coefficients: exp(-sigma * dt / 2) ---
        # k-Wave uses multiplicative PML: field = pml * (pml * field - dt * rhs)
        # where pml = exp(-sigma * dt / 2), applied TWICE per step
        #
        # k-Wave sigma_max formula (from kspaceFirstOrder2D.m):
        #   sigma_max = -(pml_alpha + 1) * c_ref * log(R0) / (2 * pml_size * dx)
        # where pml_alpha=2 (quadratic), R0=default reflection coefficient
        # For R0 = 1e-6 (−120 dB target reflection):
        pml_alpha = 2  # k-Wave default: quadratic polynomial grading
        R0 = 1e-6  # target reflection coefficient
        sigma_max = -(pml_alpha + 1) * c_ref * math.log(R0) / (2.0 * pml_width * dx)

        # Build 1D sigma profiles (quadratic polynomial grading, matching k-Wave)
        # sigma at grid points (for density/pressure updates)
        sigma_x_1d = torch.zeros(nx)
        sigma_y_1d = torch.zeros(ny)

        # sigma at staggered positions (half-cell shifted, for velocity updates)
        sigma_x_sgx_1d = torch.zeros(nx)
        sigma_y_sgy_1d = torch.zeros(ny)

        for i in range(pml_width):
            # Grid-point sigma: k-Wave uses (distance_from_inner_edge / pml_width)^alpha
            # i=0 is outer boundary, i=pml_width-1 is inner boundary
            val = sigma_max * ((pml_width - i) / pml_width) ** pml_alpha
            sigma_x_1d[i] = val
            sigma_x_1d[nx - 1 - i] = val
            sigma_y_1d[i] = val
            sigma_y_1d[ny - 1 - i] = val

            # Staggered-point sigma (half-cell offset)
            val_sg_left = sigma_max * (max(0.0, pml_width - i - 0.5) / pml_width) ** pml_alpha
            sigma_x_sgx_1d[i] = val_sg_left
            sigma_y_sgy_1d[i] = val_sg_left

            val_sg_right = sigma_max * (min(pml_width, pml_width - i + 0.5) / pml_width) ** pml_alpha
            sigma_x_sgx_1d[nx - 1 - i] = val_sg_right
            sigma_y_sgy_1d[ny - 1 - i] = val_sg_right

        # PML factors: exp(-sigma * dt / 2), broadcast to 2D
        # For density updates (grid points)
        pml_x_2d = torch.exp(-sigma_x_1d * dt / 2.0)[:, None].expand(nx, ny)  # [nx, ny]
        pml_y_2d = torch.exp(-sigma_y_1d * dt / 2.0)[None, :].expand(nx, ny)  # [nx, ny]

        # For velocity updates (staggered grid points)
        pml_x_sgx_2d = torch.exp(-sigma_x_sgx_1d * dt / 2.0)[:, None].expand(nx, ny)  # [nx, ny]
        pml_y_sgy_2d = torch.exp(-sigma_y_sgy_1d * dt / 2.0)[None, :].expand(nx, ny)  # [nx, ny]

        self.register_buffer("pml_x", pml_x_2d)
        self.register_buffer("pml_y", pml_y_2d)
        self.register_buffer("pml_x_sgx", pml_x_sgx_2d)
        self.register_buffer("pml_y_sgy", pml_y_sgy_2d)

        # Sensor column indices
        sensor_cols = torch.arange(self.element_start, self.element_end)
        self.register_buffer("sensor_cols", sensor_cols)

    def _single_step(
        self,
        ux_sgx: torch.Tensor,
        uy_sgy: torch.Tensor,
        rhox: torch.Tensor,
        rhoy: torch.Tensor,
        c_sq: torch.Tensor,
        rho0: torch.Tensor,
        rho0_sgx: torch.Tensor,
        rho0_sgy: torch.Tensor,
        source_val: torch.Tensor,
        source_mask: torch.Tensor,
        inv_2c_sq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One time step of the k-Wave kspaceFirstOrder2D algorithm.

        All operations are autograd-safe (no in-place ops).

        Returns (ux_sgx, uy_sgy, rhox, rhoy, p)
        """
        dt = self.dt

        # 1. Equation of state: compute pressure from split density
        p = c_sq * (rhox + rhoy)

        # 2. FFT of pressure
        p_k = torch.fft.fft2(p)

        # 3. Velocity update with kappa correction, staggered shift, and multiplicative PML
        #    k-Wave: dp_dx = ifft2(kappa_x * ddx_k_shift_pos * p_k).real
        #    where ddx_k_shift_pos = 1j*kx * exp(+1j*kx*dx/2)
        dp_dx = torch.fft.ifft2(self.kappa_x * 1j * self.kx_2d * self.shift_pos_x * p_k).real
        dp_dy = torch.fft.ifft2(self.kappa_y * 1j * self.ky_2d * self.shift_pos_y * p_k).real

        ux_sgx = self.pml_x_sgx * (self.pml_x_sgx * ux_sgx - dt / rho0_sgx * dp_dx)
        uy_sgy = self.pml_y_sgy * (self.pml_y_sgy * uy_sgy - dt / rho0_sgy * dp_dy)

        # 4. Density update: split-field with directional PML and backward stagger shift
        #    k-Wave: dux_dx = ifft2(kappa_x * ddx_k_shift_neg * fft2(ux_sgx)).real
        #    where ddx_k_shift_neg = 1j*kx * exp(-1j*kx*dx/2)
        ux_k = torch.fft.fft2(ux_sgx)
        uy_k = torch.fft.fft2(uy_sgy)

        dux_dx = torch.fft.ifft2(self.kappa_x * 1j * self.kx_2d * self.shift_neg_x * ux_k).real
        duy_dy = torch.fft.ifft2(self.kappa_y * 1j * self.ky_2d * self.shift_neg_y * uy_k).real

        rhox = self.pml_x * (self.pml_x * rhox - dt * rho0 * dux_dx)
        rhoy = self.pml_y * (self.pml_y * rhoy - dt * rho0 * duy_dy)

        # 5. Source injection (Dirichlet, mask-based for autograd safety)
        #    Inject into rhox and rhoy so that p = c^2*(rhox+rhoy) = source_val
        #    => rhox = rhoy = source_val / (2 * c^2)
        #    Using mask blending: rhox = src * mask + rhox * (1 - mask)
        src_rho = source_val * inv_2c_sq  # [B, nx, ny] value to inject
        rhox = src_rho * source_mask + rhox * (1.0 - source_mask)
        rhoy = src_rho * source_mask + rhoy * (1.0 - source_mask)

        # 6. Recompute p after source injection
        p = c_sq * (rhox + rhoy)

        return ux_sgx, uy_sgy, rhox, rhoy, p

    def _run_chunk(
        self,
        ux_sgx: torch.Tensor,
        uy_sgy: torch.Tensor,
        rhox: torch.Tensor,
        rhoy: torch.Tensor,
        c_sq: torch.Tensor,
        rho0: torch.Tensor,
        rho0_sgx: torch.Tensor,
        rho0_sgy: torch.Tensor,
        inv_2c_sq: torch.Tensor,
        source_chunk: torch.Tensor,
        source_mask_2d: torch.Tensor,
        chunk_start: int,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a chunk of time steps.

        Returns (ux_sgx, uy_sgy, rhox, rhoy, sensor_chunk).
        """
        sensor_list = []

        for local_t in range(chunk_size):
            # Source value at this time step: [B]
            src_val_t = source_chunk[:, local_t]  # [B]

            # Build effective source mask: only inject when source is nonzero
            is_active = (src_val_t.abs() > 0.0).float()  # [B]
            # source_val broadcast: [B, nx, ny]
            source_val = src_val_t[:, None, None] * source_mask_2d[None, :, :]
            # effective mask: [B, nx, ny]
            eff_mask = is_active[:, None, None] * source_mask_2d[None, :, :]

            ux_sgx, uy_sgy, rhox, rhoy, p = self._single_step(
                ux_sgx, uy_sgy, rhox, rhoy,
                c_sq, rho0, rho0_sgx, rho0_sgy,
                source_val, eff_mask, inv_2c_sq,
            )

            # Record sensor data: p at transducer_row, sensor columns
            sensor_list.append(p[:, self.transducer_row, self.sensor_cols])

        sensor_chunk = torch.stack(sensor_list, dim=-1)  # [B, n_elements, chunk_size]
        return ux_sgx, uy_sgy, rhox, rhoy, sensor_chunk

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
            Attenuation coefficient. Set to 0 for lossless (matching k-Wave).
            Currently unused in the lossless formulation; reserved for future
            absorption models.
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
            rho0 = torch.full((B, self.nx, self.ny), 1000.0, device=device, dtype=dtype)
        else:
            rho0 = rho

        # --- CFL check ---
        c_max = c.max().item()
        cfl = c_max * self.dt / self.dx
        if cfl > 1.0 / math.sqrt(2.0):
            raise ValueError(
                f"CFL condition violated: CFL={cfl:.4f} > {1.0/math.sqrt(2.0):.4f}. "
                f"c_max={c_max:.1f}, dt={self.dt:.2e}, dx={self.dx:.2e}"
            )

        # --- Precompute fields ---
        c_sq = c * c  # [B, nx, ny]
        inv_2c_sq = 1.0 / (2.0 * c_sq)  # [B, nx, ny] — for source injection scaling

        # Staggered density interpolation (k-Wave BUG 4 fix)
        # rho0_sgx = 0.5 * (rho0 + roll(rho0, -1, dim=-2))  half-cell shift in x
        # rho0_sgy = 0.5 * (rho0 + roll(rho0, -1, dim=-1))  half-cell shift in y
        rho0_sgx = 0.5 * (rho0 + torch.roll(rho0, shifts=-1, dims=-2))
        rho0_sgy = 0.5 * (rho0 + torch.roll(rho0, shifts=-1, dims=-1))

        # Source mask: 2D binary mask for transducer locations
        source_mask_2d = torch.zeros(self.nx, self.ny, device=device, dtype=dtype)
        source_mask_2d[self.transducer_row, self.element_start:self.element_end] = 1.0

        # Initialize state variables (split-field)
        ux_sgx = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)
        uy_sgy = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)
        rhox = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)
        rhoy = torch.zeros(B, self.nx, self.ny, device=device, dtype=dtype)

        # --- Time-stepping with gradient checkpointing ---
        sensor_chunks = []
        chunk_size = self.checkpoint_every

        for chunk_start in range(0, n_steps, chunk_size):
            actual_chunk_size = min(chunk_size, n_steps - chunk_start)
            source_chunk = source[:, chunk_start:chunk_start + actual_chunk_size]

            if self.training and torch.is_grad_enabled():
                # Use gradient checkpointing
                def run_ckpt(
                    ux_sgx_, uy_sgy_, rhox_, rhoy_,
                    c_sq_, rho0_, rho0_sgx_, rho0_sgy_, inv_2c_sq_,
                    src_chunk_,
                    _cs=chunk_start, _sz=actual_chunk_size,
                ):
                    return self._run_chunk(
                        ux_sgx_, uy_sgy_, rhox_, rhoy_,
                        c_sq_, rho0_, rho0_sgx_, rho0_sgy_, inv_2c_sq_,
                        src_chunk_, source_mask_2d, _cs, _sz,
                    )

                ux_sgx, uy_sgy, rhox, rhoy, sensor_chunk = checkpoint(
                    run_ckpt,
                    ux_sgx, uy_sgy, rhox, rhoy,
                    c_sq, rho0, rho0_sgx, rho0_sgy, inv_2c_sq,
                    source_chunk,
                    use_reentrant=False,
                )
            else:
                ux_sgx, uy_sgy, rhox, rhoy, sensor_chunk = self._run_chunk(
                    ux_sgx, uy_sgy, rhox, rhoy,
                    c_sq, rho0, rho0_sgx, rho0_sgy, inv_2c_sq,
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

    print("=" * 70)
    print("AcousticPropagatorV5 — k-Wave kspaceFirstOrder2D Test")
    print("=" * 70)

    # --- Create propagator ---
    prop = AcousticPropagatorV5(
        nx=nx, ny=ny, dx=dx, dt=dt,
        pml_width=pml_width, n_elements=n_elements, c_ref=c_ref,
    ).to(device)

    # --- Two-layer medium: c=1500 top, c=1800 bottom ---
    B = 1
    c_map = torch.ones(B, nx, ny, device=device, dtype=dtype) * 1500.0
    c_map[:, nx // 2:, :] = 1800.0  # interface at row 128

    alpha_map = torch.zeros(B, nx, ny, device=device, dtype=dtype)  # lossless

    # --- Source: 2 MHz, 3-cycle burst ---
    freq = 2.0e6
    t_axis = torch.arange(n_steps, device=device, dtype=dtype) * dt
    n_cycles = 3
    burst_duration = n_cycles / freq  # 1.5e-6 s
    burst_mask = (t_axis < burst_duration).float()
    source_signal = (torch.sin(2.0 * math.pi * freq * t_axis) * burst_mask).unsqueeze(0)

    # Scale source to physical pressure
    source_signal = source_signal * 1e6  # Pa

    # --- CFL check ---
    c_max = c_map.max().item()
    cfl = c_max * dt / dx
    print(f"CFL number: {cfl:.4f}  (limit: {1.0 / math.sqrt(2.0):.4f})")
    print(f"c_max: {c_max:.0f} m/s,  dt: {dt:.2e} s,  dx: {dx:.2e} m")
    print(f"Source: {freq / 1e6:.1f} MHz, {n_cycles} cycles, "
          f"burst = {burst_duration * 1e6:.1f} µs = {int(burst_duration / dt)} steps")

    # --- Run propagation (eval mode, no grad) ---
    print(f"\nRunning {n_steps} steps (eval mode, no grad)...")
    t0 = time.time()
    prop.eval()
    with torch.no_grad():
        sensor_data = prop(c_map, alpha_map, source_signal)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.2f}s")
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Sensor data range: [{sensor_data.min().item():.4e}, {sensor_data.max().item():.4e}]")
    print(f"Sensor data mean:  {sensor_data.mean().item():.4e}")
    print(f"Sensor data std:   {sensor_data.std().item():.4e}")

    # Check for non-trivial signal after burst
    burst_steps = int(burst_duration / dt)
    post_burst = sensor_data[:, :, burst_steps:]
    print(f"\nPost-burst signal (steps {burst_steps}-{n_steps}):")
    print(f"  max abs: {post_burst.abs().max().item():.4e}")
    print(f"  mean abs: {post_burst.abs().mean().item():.4e}")
    if post_burst.abs().max().item() > 1e-10:
        print("  ✓ Post-burst echoes detected!")
    else:
        print("  ✗ WARNING: No post-burst signal — possible propagation issue")

    # --- Gradient flow check ---
    print("\n--- Gradient flow check ---")
    prop.train()
    c_grad = torch.ones(B, nx, ny, device=device, dtype=dtype) * 1500.0
    c_grad[:, nx // 2:, :] = 1800.0
    c_grad = c_grad.clone().requires_grad_(True)

    alpha_grad = torch.zeros(B, nx, ny, device=device, dtype=dtype)

    sensor_grad = prop(c_grad, alpha_grad, source_signal)
    loss = sensor_grad.abs().mean()
    loss.backward()

    grad_norm = c_grad.grad.norm().item() if c_grad.grad is not None else 0.0
    grad_max = c_grad.grad.abs().max().item() if c_grad.grad is not None else 0.0
    grad_nonzero = (c_grad.grad.abs() > 0).sum().item() if c_grad.grad is not None else 0

    print(f"Loss: {loss.item():.6e}")
    print(f"Gradient norm (c): {grad_norm:.6e}")
    print(f"Gradient max  (c): {grad_max:.6e}")
    print(f"Gradient nonzero:  {grad_nonzero} / {nx * ny}")
    print(f"Gradient flows: {'YES ✓' if grad_norm > 0 else 'NO ✗'}")

    print("\n" + "=" * 70)
    if grad_norm > 0 and post_burst.abs().max().item() > 1e-10:
        print("ALL CHECKS PASSED ✓")
    else:
        issues = []
        if grad_norm == 0:
            issues.append("no gradient flow")
        if post_burst.abs().max().item() <= 1e-10:
            issues.append("no post-burst echoes")
        print(f"ISSUES DETECTED: {', '.join(issues)}")
    print("=" * 70)
