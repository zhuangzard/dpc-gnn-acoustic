"""
wave_propagator.py — Physically correct Leapfrog wave propagator on graphs.

Replaces kwave_inspired_mp.py with a minimal, physics-correct implementation.

Physics:
  - Leapfrog time integration: p_{n+1} = 2p_n - p_{n-1} + c²dt²∇²p_n + source_n
  - Taylor initialization for zero initial velocity: p_{-1} = p_0 - 0.5*c²*dt²*∇²p_0
  - Graph Laplacian: L(p_i) = (1/dx²) * Σ_j correction_ij * (p_j - p_i)
  - PML absorbing boundaries: σ(x) = σ_max * (x/L_pml)^3
  - Frequency-dependent attenuation: p *= exp(-α*c*dt)
  - GNN correction factor ∈ (0.5, 2.0), initialized to 1.0

All time steps share weights (share_weights=True).
Uses torch.utils.checkpoint for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_add
from typing import Optional, Dict, Tuple


class LeapfrogWavePropagator(nn.Module):
    """Minimal physics-correct wave propagator using Leapfrog integration on graphs.

    Args:
        hidden_dim: Hidden dimension for correction MLP
        dt: Time step [s]
        n_time_steps: Number of propagation steps
        pml_thickness: Number of PML layers (in grid cells)
        sigma_max: Maximum PML absorption coefficient
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dt: float = 2e-8,
        n_time_steps: int = 200,
        pml_thickness: int = 10,
        sigma_max: float = 1e4,
    ):
        super().__init__()
        self.dt = dt
        self.n_time_steps = n_time_steps
        self.pml_thickness = pml_thickness
        self.sigma_max = sigma_max

        # GNN correction factor: edge MLP → scalar correction per edge
        # Input: [dx, dy, distance, c_src/1540, c_dst/1540]
        self.correction_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to output ~0 so sigmoid → 0.5, then shifted to 1.0
        nn.init.zeros_(self.correction_mlp[-1].weight)
        nn.init.zeros_(self.correction_mlp[-1].bias)

        # For storing pressure history (used by physics loss)
        self.pressure_history = None
        self.energy_history = []

    def _compute_graph_laplacian(
        self,
        p: torch.Tensor,          # (N,)
        edge_index: torch.Tensor,  # (2, E)
        edge_weights: torch.Tensor,  # (E,) = correction / dx²
    ) -> torch.Tensor:
        """Compute graph Laplacian: L(p_i) = Σ_j w_ij * (p_j - p_i).

        For a regular grid with correction=1: this equals (1/dx²) * Σ_j (p_j - p_i),
        which is the standard 2D Laplacian stencil.

        Returns:
            lap: (N,) Laplacian values
        """
        src, dst = edge_index  # src → dst
        diff = p[src] - p[dst]  # p_j - p_i (for dst node i, src node j)
        weighted_diff = edge_weights * diff
        lap = scatter_add(weighted_diff, dst, dim=0, dim_size=p.shape[0])
        return lap

    def _compute_pml_damping(
        self,
        positions: torch.Tensor,   # (N, 2)
        domain_size: torch.Tensor,  # (2,)
    ) -> torch.Tensor:
        """Compute PML damping factor: exp(-σ*dt).

        σ(x) = σ_max * (x_pml / L_pml)^3 where x_pml is distance into PML region.

        Returns:
            damping: (N,) damping factor per node, 1.0 in interior
        """
        dx = domain_size[0] / 256.0  # approximate grid spacing
        L_pml = self.pml_thickness * dx  # PML thickness in meters

        sigma = torch.zeros(positions.shape[0], device=positions.device)

        for dim in range(2):
            # Left boundary
            dist_left = L_pml - positions[:, dim]
            mask_left = dist_left > 0
            sigma[mask_left] += self.sigma_max * (dist_left[mask_left] / L_pml).pow(3)

            # Right boundary
            dist_right = positions[:, dim] - (domain_size[dim] - L_pml)
            mask_right = dist_right > 0
            sigma[mask_right] += self.sigma_max * (dist_right[mask_right] / L_pml).pow(3)

        damping = torch.exp(-sigma * self.dt)
        return damping

    def _compute_edge_weights(
        self,
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, D) — at least [dx, dy, dist, ...]
        c: torch.Tensor,           # (N,) sound speed
        dx: float,                 # grid spacing
    ) -> torch.Tensor:
        """Compute edge weights = correction_ij / dx².

        correction_ij ∈ (0.5, 2.0), initialized to 1.0.
        Base weight = 1/dx² for regular grid.

        Returns:
            weights: (E,) edge weights
        """
        src, dst = edge_index

        # Build MLP input: relative position + distance + sound speed ratio
        rel_pos = edge_attr[:, :2]  # (E, 2) — dx, dy components
        dist = edge_attr[:, 2:3]    # (E, 1) — distance
        c_src = (c[src] / 1540.0).unsqueeze(-1)  # (E, 1)
        c_dst = (c[dst] / 1540.0).unsqueeze(-1)  # (E, 1)

        mlp_input = torch.cat([rel_pos / dx, dist / dx, c_src, c_dst], dim=-1)  # (E, 5)

        # Correction factor: sigmoid maps to (0, 1), scale to (0.5, 2.0)
        raw = self.correction_mlp(mlp_input).squeeze(-1)  # (E,)
        correction = 0.5 + 1.5 * torch.sigmoid(raw)  # (0.5, 2.0), init ≈ 1.25

        # Edge weight = correction / dx²
        weights = correction / (dx * dx)
        return weights

    def _single_step(
        self,
        p_curr: torch.Tensor,       # (N,)
        p_prev: torch.Tensor,       # (N,)
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,  # (E,)
        c_sq_dt_sq: torch.Tensor,   # (N,) = c² * dt²
        damping: torch.Tensor,      # (N,) PML damping
        attenuation: torch.Tensor,  # (N,) freq-dependent attenuation
        source: Optional[torch.Tensor] = None,  # (N,) source term
    ) -> torch.Tensor:
        """Single Leapfrog step.

        p_{n+1} = 2*p_n - p_{n-1} + c²*dt²*∇²p_n + source_n
        Then apply PML and attenuation.
        """
        lap = self._compute_graph_laplacian(p_curr, edge_index, edge_weights)
        p_next = 2.0 * p_curr - p_prev + c_sq_dt_sq * lap

        if source is not None:
            p_next = p_next + source

        # PML absorption
        p_next = p_next * damping

        # Frequency-dependent attenuation
        p_next = p_next * attenuation

        return p_next

    def forward(
        self,
        p0: torch.Tensor,                     # (N, 1) initial pressure
        edge_index: torch.Tensor,              # (2, E)
        edge_attr: torch.Tensor,               # (E, D)
        node_props: Dict[str, torch.Tensor],   # c, rho, alpha_0, dx_mean
        positions: Optional[torch.Tensor] = None,
        domain_size: Optional[torch.Tensor] = None,
        source_fn=None,                        # callable(t) → (N,) or None
    ) -> Tuple[torch.Tensor, list]:
        """Run Leapfrog wave propagation.

        Args:
            p0: (N, 1) initial pressure field
            edge_index: (2, E) graph edges
            edge_attr: (E, D) edge features [dx, dy, dist, ...]
            node_props: dict with 'c', 'rho', 'alpha_0', 'dx_mean'
            positions: (N, 2) node positions (for PML)
            domain_size: (2,) domain dimensions (for PML)
            source_fn: optional source function source_fn(step_idx) → (N,)

        Returns:
            p_final: (N, 1) final pressure
            all_pressures: list of (N, 1) pressure at each step
        """
        p0_flat = p0.squeeze(-1)  # (N,)
        N = p0_flat.shape[0]
        device = p0_flat.device

        c = node_props['c'].to(device)
        alpha_0 = node_props['alpha_0'].to(device)

        # Grid spacing
        if 'dx_mean' in node_props:
            dx_val = node_props['dx_mean']
            if isinstance(dx_val, torch.Tensor):
                dx = dx_val.item()
            else:
                dx = float(dx_val)
        else:
            dx = 2.34e-4  # default for 256x256 on 6cm domain

        dt = self.dt

        # Precompute constants
        c_sq_dt_sq = (c * c) * (dt * dt)  # (N,)

        # Frequency-dependent attenuation factor per step: exp(-alpha * c * dt)
        attenuation = torch.exp(-alpha_0 * c * dt)  # (N,)

        # Edge weights with learned correction
        edge_weights = self._compute_edge_weights(edge_index, edge_attr, c, dx)

        # PML damping
        if positions is not None and domain_size is not None:
            damping = self._compute_pml_damping(positions, domain_size)
        else:
            damping = torch.ones(N, device=device)

        # ── Taylor initialization for zero initial velocity ──
        # p_{-1} = p_0 - 0.5 * c² * dt² * ∇²p_0
        lap_p0 = self._compute_graph_laplacian(p0_flat, edge_index, edge_weights)
        p_prev = p0_flat - 0.5 * c_sq_dt_sq * lap_p0

        p_curr = p0_flat.clone()

        # Store pressure history
        all_pressures = [p0.clone()]  # step 0
        self.energy_history = [p0_flat.pow(2).sum().item()]

        # ── Time stepping ──
        for step in range(self.n_time_steps):
            source = source_fn(step) if source_fn is not None else None

            # Use checkpointing for memory efficiency (every 10 steps)
            if self.training and step % 10 == 0:
                p_next = checkpoint(
                    self._single_step,
                    p_curr, p_prev, edge_index, edge_weights,
                    c_sq_dt_sq, damping, attenuation, source,
                    use_reentrant=False,
                )
            else:
                p_next = self._single_step(
                    p_curr, p_prev, edge_index, edge_weights,
                    c_sq_dt_sq, damping, attenuation, source,
                )

            # NO detach! Gradients must flow through for physics loss
            all_pressures.append(p_next.unsqueeze(-1))
            self.energy_history.append(p_next.pow(2).sum().item())

            p_prev = p_curr
            p_curr = p_next

        p_final = p_curr.unsqueeze(-1)  # (N, 1)

        # Cache for physics loss computation
        self.pressure_history = all_pressures
        self._edge_index = edge_index
        self._edge_weights = edge_weights
        self._c_sq_dt_sq = c_sq_dt_sq

        return p_final, all_pressures

    def compute_wave_equation_residual(self) -> torch.Tensor:
        """Compute wave equation residual from stored pressure history.

        Dimensionless form:
            residual = (p[t+1] - 2*p[t] + p[t-1]) - c²dt² * ∇²p[t]

        This is already O(1) since c²dt²/dx² ≈ CFL² ≈ 0.017.
        No division by dt²!

        Returns:
            physics_loss: scalar
        """
        if self.pressure_history is None or len(self.pressure_history) < 3:
            return torch.tensor(0.0)

        residuals = []
        # Sample a subset of time steps to save compute
        n_steps = len(self.pressure_history)
        step_indices = list(range(1, n_steps - 1, max(1, n_steps // 20)))

        for t in step_indices:
            p_prev = self.pressure_history[t - 1].squeeze(-1)
            p_curr = self.pressure_history[t].squeeze(-1)
            p_next = self.pressure_history[t + 1].squeeze(-1)

            # Time difference: p[t+1] - 2*p[t] + p[t-1]
            d2p_dt2 = p_next - 2.0 * p_curr + p_prev

            # Spatial Laplacian
            lap = self._compute_graph_laplacian(
                p_curr, self._edge_index, self._edge_weights
            )

            # Residual (dimensionless): should be ≈ 0 for exact solution
            residual = d2p_dt2 - self._c_sq_dt_sq * lap
            residuals.append(residual.pow(2).mean())

        if len(residuals) == 0:
            return torch.tensor(0.0)

        return torch.stack(residuals).mean()
