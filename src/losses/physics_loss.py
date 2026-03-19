"""
physics_loss.py — Correct wave equation residual loss.

Computes the dimensionless wave equation residual:
    residual = (p[t+1] - 2*p[t] + p[t-1]) - (c*dt/dx)² * ∇²p[t]

Key design decisions:
  - NO division by dt² (this caused numerical explosion in v2)
  - Uses the discrete difference form directly
  - CFL number c*dt/dx ≈ 0.132 keeps the residual O(1)
  - Samples a subset of time steps for efficiency
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from typing import List, Optional


class WaveEquationResidualLoss(nn.Module):
    """Dimensionless wave equation residual loss.

    Given pressure history p[0], p[1], ..., p[T], computes:
        R[t] = (p[t+1] - 2*p[t] + p[t-1]) - (c*dt)² * ∇²p[t] / 1
        loss = mean(R[t]²)

    where ∇² is the graph Laplacian with weights 1/dx².

    The key insight: in the Leapfrog scheme,
        p[t+1] - 2*p[t] + p[t-1] = (c*dt)² * ∇²p[t]
    so the residual is identically zero for an exact solution.
    For the GNN with learned corrections, the residual measures
    how well the correction factors preserve wave physics.

    Args:
        n_sample_steps: Number of time steps to sample for loss computation
    """

    def __init__(self, n_sample_steps: int = 20):
        super().__init__()
        self.n_sample_steps = n_sample_steps

    def forward(
        self,
        pressure_history: List[torch.Tensor],  # list of (N, 1) or (N,)
        edge_index: torch.Tensor,               # (2, E)
        edge_weights: torch.Tensor,             # (E,) = correction / dx²
        c: torch.Tensor,                        # (N,) sound speed
        dt: float,                              # time step
    ) -> torch.Tensor:
        """Compute wave equation residual loss.

        Args:
            pressure_history: list of pressure fields at each time step
            edge_index: graph edges
            edge_weights: edge weights (includes 1/dx² factor)
            c: sound speed per node
            dt: time step

        Returns:
            loss: scalar, mean squared residual
        """
        T = len(pressure_history)
        if T < 3:
            return torch.tensor(0.0, device=c.device)

        # c²*dt² factor
        c_sq_dt_sq = (c * c) * (dt * dt)  # (N,)

        # Sample time steps uniformly
        step_stride = max(1, (T - 2) // self.n_sample_steps)
        step_indices = list(range(1, T - 1, step_stride))

        residuals = []
        for t in step_indices:
            p_prev = pressure_history[t - 1].squeeze(-1)  # (N,)
            p_curr = pressure_history[t].squeeze(-1)
            p_next = pressure_history[t + 1].squeeze(-1)

            # Temporal second derivative (discrete)
            d2p_dt2 = p_next - 2.0 * p_curr + p_prev

            # Spatial Laplacian
            src, dst = edge_index
            diff = p_curr[src] - p_curr[dst]
            weighted_diff = edge_weights * diff
            lap = scatter_add(weighted_diff, dst, dim=0, dim_size=p_curr.shape[0])

            # Residual: should be ≈ 0
            residual = d2p_dt2 - c_sq_dt_sq * lap

            residuals.append(residual.pow(2).mean())

        return torch.stack(residuals).mean()


def compute_physics_loss_from_model(model) -> torch.Tensor:
    """Convenience: extract physics loss from a v3 model's propagator.

    Delegates to the propagator's built-in residual computation.

    Args:
        model: DPCGNNAcousticV3 instance (after forward pass)

    Returns:
        loss: scalar physics loss
    """
    return model.compute_physics_loss()
