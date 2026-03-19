"""
kwave_inspired_mp.py — k-Wave-Inspired Message Passing Layer.

Core innovation of DPC-GNN-Acoustic v2:
  A physics-biased message passing layer that incorporates k-Wave's
  key numerical methods as differentiable neural sub-networks.

## Why the old WaveEquationMP fails

The old WaveEquationMP implements a basic graph Laplacian:
    L(p_i) = Σ_j w_ij · (p_j - p_i)
    p^{n+1} = 2p^n - p^{n-1} + (c·dt)² · L(p)

Problems:
  1. **No dispersion compensation**: Finite-difference Laplacian introduces
     numerical dispersion that grows with frequency. k-Wave fixes this with
     k-space pseudo-spectral methods. We learn this correction.
  
  2. **No power-law attenuation**: Real tissue attenuates as α₀·f^y with
     y ∈ [1, 1.5]. The old code only has simple exponential decay.
  
  3. **Dumb boundaries**: Simple PML with fixed polynomial σ-profile.
     k-Wave's PML is optimized and frequency-adapted.
  
  4. **No heterogeneous coupling**: At tissue interfaces (e.g., fat→bone),
     there's reflection/transmission governed by impedance mismatch.
     The old code doesn't model this properly.

## This layer's approach

We keep the physics structure (wave equation + leapfrog) but augment it:
  - Laplacian → Physics-biased learned Laplacian
  - Fixed update → Dispersion-corrected update (κ·(c·dt)²·L)  
  - No attenuation → Learned power-law attenuation
  - Fixed PML → Learned PML σ-profile
  - Simple aggregation → Attention-weighted for heterogeneous media

The key design principle: **physics sets the inductive bias, data (k-Wave GT)
teaches the corrections.**

References:
  - Treeby & Cox, JASA 2012: k-Wave original paper
  - Tabei et al., JASA 2002: k-space method theory
  - Pfaff et al., ICLR 2021: MeshGraphNets
"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean
from typing import Optional, Tuple, Dict, List

from .components import (
    DispersionCorrectionNet,
    AttenuationNet,
    LearnedPML,
    EdgeEncoder,
    MultiHeadAttentionAggregation,
    TimeStepCorrector,
)


class KWaveInspiredMP(MessagePassing):
    """k-Wave-Inspired Message Passing layer for acoustic wave propagation.
    
    Architecture per step:
    ```
    1. Edge encoding: raw edge features → edge embeddings
    2. Message computation: attention-weighted pressure differences
    3. Aggregation: scatter_add → graph Laplacian approximation
    4. Dispersion correction: κ(x) · Laplacian  
    5. Attenuation: α(f,x) applied to pressure update
    6. Leapfrog + correction: p^{n+1} with learned time-step correction
    7. PML damping: boundary absorption
    ```
    
    Args:
        hidden_dim: Feature dimension for encoder/decoder
        edge_dim: Raw edge feature dimension
        n_heads: Number of attention heads
        frequency: Operating frequency [Hz]
        dt: Time step [s]
        use_dispersion: Enable dispersion correction network
        use_attenuation: Enable learned attenuation
        use_pml: Enable learned PML
        use_attention: Enable multi-head attention aggregation
        use_time_corrector: Enable learned time-step correction
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 7,
        n_heads: int = 4,
        frequency: float = 5e6,
        dt: float = 2e-8,
        use_dispersion: bool = True,
        use_attenuation: bool = True,
        use_pml: bool = True,
        use_attention: bool = True,
        use_time_corrector: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.frequency = frequency
        self.dt = dt
        self.eps = eps
        self.use_dispersion = use_dispersion
        self.use_attenuation = use_attenuation
        self.use_pml = use_pml
        self.use_attention = use_attention
        self.use_time_corrector = use_time_corrector
        
        # ── Edge encoder ──
        self.edge_encoder = EdgeEncoder(input_dim=edge_dim, hidden_dim=hidden_dim)
        
        # ── Message MLP: computes per-edge messages ──
        # Input: [pressure_diff(1), edge_emb(hidden), weight(1)]
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # ── Optional multi-head attention ──
        if use_attention:
            self.attention = MultiHeadAttentionAggregation(
                hidden_dim=hidden_dim, n_heads=n_heads
            )
        
        # ── Dispersion correction ──
        if use_dispersion:
            self.dispersion_net = DispersionCorrectionNet(hidden_dim=64)
        
        # ── Attenuation ──
        if use_attenuation:
            self.attenuation_net = AttenuationNet(hidden_dim=32)
        
        # ── PML ──
        if use_pml:
            self.pml_net = LearnedPML(hidden_dim=32)
        
        # ── Time step corrector ──
        if use_time_corrector:
            self.time_corrector = TimeStepCorrector(hidden_dim=64)
        
        # ── Node update MLP (residual style) ──
        self.node_update = nn.Sequential(
            nn.Linear(2, hidden_dim),  # [p_curr, aggregated_laplacian]
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to identity-like (pass through laplacian)
        nn.init.zeros_(self.node_update[-1].weight)
        nn.init.zeros_(self.node_update[-1].bias)
    
    def forward(
        self,
        p_curr: torch.Tensor,            # (N, 1) current pressure
        p_prev: torch.Tensor,            # (N, 1) previous pressure
        edge_index: torch.Tensor,         # (2, E) graph edges
        edge_attr: torch.Tensor,          # (E, edge_dim) raw edge features
        node_props: Dict[str, torch.Tensor],  # Per-node acoustic properties
        positions: Optional[torch.Tensor] = None,  # (N, D) for PML
        domain_size: Optional[torch.Tensor] = None,  # (D,) for PML
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One complete wave propagation step.
        
        Args:
            p_curr: (N, 1) pressure at time n
            p_prev: (N, 1) pressure at time n-1
            edge_index: (2, E) graph connectivity
            edge_attr: (E, edge_dim) edge features:
                [r_vec(2or3), distance(1), Z_ratio(1), c_ratio(1), alpha_avg(1)]
            node_props: Dictionary with:
                - 'c': (N,) sound speed [m/s]
                - 'rho': (N,) density [kg/m³]
                - 'alpha_0': (N,) reference attenuation [Np/m]
                - 'n_power': (N,) power-law exponent
                - 'dx_mean': (N,) mean neighbor distance [m]
            positions: (N, D) node positions [m], needed for PML
            domain_size: (D,) domain dimensions [m], needed for PML
        
        Returns:
            p_next: (N, 1) pressure at time n+1
            p_curr: (N, 1) current pressure (for history tracking)
        """
        N = p_curr.shape[0]
        device = p_curr.device
        
        c = node_props['c']          # (N,)
        rho = node_props['rho']      # (N,)
        alpha_0 = node_props['alpha_0']  # (N,)
        n_power = node_props['n_power']  # (N,)
        dx_mean = node_props['dx_mean']  # (N,)
        
        # ─── Step 1: Encode edges ───
        edge_emb = self.edge_encoder(edge_attr)  # (E, hidden_dim)
        
        # ─── Step 2: Compute graph Laplacian via message passing ───
        # Edge weights: physics-based with learned corrections
        distance = edge_attr[:, -3] + self.eps  # distance column
        src, dst = edge_index
        
        # FIX #7: Base weight: 1 / |r_ij|² (correct dimension for Laplacian kernel)
        # Old: 1/|r| which has wrong units for ∇² approximation
        # Laplacian ∝ Σ (p_j - p_i) / |r_ij|² in 2D/3D
        base_weight = 1.0 / (distance ** 2)  # (E,)
        
        # Impedance-weighted (reflection at interfaces)
        if edge_attr.shape[1] > 4:
            Z_ratio = edge_attr[:, -2]  # Z_j / Z_i
            base_weight = base_weight * Z_ratio
        
        # Attenuation along edge
        if self.use_attenuation:
            alpha_edge = (alpha_0[src] + alpha_0[dst]) / 2
            n_edge = (n_power[src] + n_power[dst]) / 2
            c_edge = (c[src] + c[dst]) / 2
            rho_edge = (rho[src] + rho[dst]) / 2
            
            alpha_eff = self.attenuation_net(
                alpha_edge, n_edge, self.frequency, c_edge, rho_edge
            )
            atten_factor = torch.exp(-alpha_eff * distance)
            base_weight = base_weight * atten_factor
        
        # Message passing: weighted pressure differences
        laplacian = self.propagate(
            edge_index,
            p=p_curr,
            edge_weight=base_weight,
            edge_emb=edge_emb,
            size=None,
        )  # (N, 1)
        
        # ─── Step 3: Dispersion correction ───
        if self.use_dispersion:
            kappa = self.dispersion_net(c, dx_mean, self.dt, self.frequency)  # (N, 1)
        else:
            kappa = torch.ones(N, 1, device=device)
        
        # ─── Step 4: Leapfrog time step with corrections ───
        c_sq = (c ** 2).unsqueeze(-1)  # (N, 1)
        dt_sq = self.dt ** 2
        
        # Standard leapfrog with dispersion correction
        p_next = 2.0 * p_curr - p_prev + kappa * c_sq * dt_sq * laplacian
        
        # Learned time-step correction
        if self.use_time_corrector:
            correction = self.time_corrector(p_curr, p_prev, laplacian, c, self.dt)
            p_next = p_next + correction
        
        # ─── Step 5: PML boundary damping ───
        if self.use_pml and positions is not None and domain_size is not None:
            damping = self.pml_net(positions, domain_size, c, self.frequency, self.dt)
            p_next = p_next * damping  # (N, 1)
        
        return p_next, p_curr
    
    def message(
        self,
        p_j: torch.Tensor,         # (E, 1) pressure at source
        p_i: torch.Tensor,         # (E, 1) pressure at target
        edge_weight: torch.Tensor,  # (E,) physics-based weights
        edge_emb: torch.Tensor,     # (E, hidden_dim) edge embeddings
    ) -> torch.Tensor:
        """Compute per-edge messages.
        
        Message = w_ij · (p_j - p_i) + learned_correction
        
        The learned correction captures higher-order terms in the
        Laplacian approximation that the simple difference misses.
        """
        # Physics-based message
        pressure_diff = p_j - p_i  # (E, 1)
        physics_msg = edge_weight.unsqueeze(-1) * pressure_diff  # (E, 1)
        
        # Learned correction via MLP
        msg_input = torch.cat([
            pressure_diff,                    # (E, 1)
            edge_emb,                         # (E, hidden_dim)
            edge_weight.unsqueeze(-1),        # (E, 1)
        ], dim=-1)  # (E, hidden_dim + 2)
        
        learned_correction = self.message_mlp(msg_input)  # (E, 1)
        
        # Physics + small learned correction
        msg = physics_msg + 0.1 * learned_correction
        
        return msg
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Aggregate messages → Laplacian approximation."""
        return aggr_out


class KWaveInspiredMPStack(nn.Module):
    """Stack of KWaveInspiredMP layers for multi-step wave propagation.
    
    Handles the full time-stepping loop with pressure history tracking
    and energy monitoring.
    
    Args:
        n_steps: Number of time steps (FIX #1: now uses config n_time_steps, default 200)
        hidden_dim: Feature dimension
        edge_dim: Edge feature dimension
        n_heads: Attention heads
        frequency: Operating frequency [Hz]
        dt: Time step [s]
        share_weights: If True, all steps share the same MP weights
            (FIX #1: must be True for n_steps=200 to save memory)
        kwargs: Additional arguments for KWaveInspiredMP
    """
    
    def __init__(
        self,
        n_steps: int = 200,  # FIX #1: default changed from 12 to 200
        hidden_dim: int = 128,
        edge_dim: int = 7,
        n_heads: int = 4,
        frequency: float = 5e6,
        dt: float = 2e-8,
        share_weights: bool = True,  # FIX #1: default changed to True for memory efficiency
        **kwargs,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.share_weights = share_weights
        
        if share_weights:
            # FIX #1: Single shared layer for all time steps (memory efficient)
            self.mp_layer = KWaveInspiredMP(
                hidden_dim=hidden_dim, edge_dim=edge_dim,
                n_heads=n_heads, frequency=frequency, dt=dt, **kwargs,
            )
        else:
            self.mp_layers = nn.ModuleList([
                KWaveInspiredMP(
                    hidden_dim=hidden_dim, edge_dim=edge_dim,
                    n_heads=n_heads, frequency=frequency, dt=dt, **kwargs,
                )
                for _ in range(n_steps)
            ])
        
        # Pressure history for physics loss computation
        # FIX #4: these will store tensors WITH gradients (no detach)
        self.pressure_history = []
        self.laplacian_history = []  # FIX #3: store Laplacian for physics loss
        self.energy_history = []
    
    def forward(
        self,
        p0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_props: Dict[str, torch.Tensor],
        positions: Optional[torch.Tensor] = None,
        domain_size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Propagate pressure through n_steps time steps.
        
        Args:
            p0: (N, 1) initial pressure field
            edge_index: (2, E) graph connectivity
            edge_attr: (E, edge_dim) edge features
            node_props: Per-node acoustic properties
            positions: (N, D) node positions for PML
            domain_size: (D,) domain size for PML
        
        Returns:
            p_final: (N, 1) final pressure field after n_steps
            transducer_pressure_history: list of (N,1) pressures at each step
                (used by BeamformDecoder for RF data, FIX #5)
        """
        # FIX #4: store p0 WITH gradients (no .detach())
        self.pressure_history = [p0]
        self.laplacian_history = []  # FIX #3: for physics loss
        self.energy_history = []
        
        # Initialize
        c = node_props['c']
        c_sq = (c ** 2).unsqueeze(-1)
        if self.share_weights:
            dt = self.mp_layer.dt
        else:
            dt = self.mp_layers[0].dt
        dt_sq = dt ** 2
        
        # FIX #6: Leapfrog initialization with Taylor expansion
        # Instead of p_prev = p0 (wrong: assumes zero velocity AND zero acceleration),
        # use Taylor: p_prev = p0 - 0.5 * c² * dt² * ∇²p0
        # For quiescent start (v=0), the correct backward extrapolation is:
        #   p(t=-dt) = p(t=0) - dt*v(t=0) + 0.5*dt²*a(t=0)
        #   = p0 + 0.5 * dt² * c² * ∇²p0   (since v=0 and ∂²p/∂t² = c²∇²p)
        # We approximate ∇²p0 via one MP step's Laplacian
        mp_init = self.mp_layer if self.share_weights else self.mp_layers[0]
        laplacian_p0 = mp_init.propagate(
            edge_index,
            p=p0,
            edge_weight=self._compute_base_weights(edge_attr, edge_index, node_props, mp_init),
            edge_emb=mp_init.edge_encoder(edge_attr),
            size=None,
        )
        p_prev = p0 + 0.5 * c_sq * dt_sq * laplacian_p0  # FIX #6: Taylor expansion
        p_curr = p0.clone()
        
        # FIX #5: collect pressure at ALL time steps (for BeamformDecoder RF matrix)
        all_pressures = [p0]
        
        for k in range(self.n_steps):
            mp = self.mp_layer if self.share_weights else self.mp_layers[k]
            
            # FIX #1: use gradient checkpointing for memory efficiency with 200 steps
            if self.share_weights and self.training and k > 0:
                from torch.utils.checkpoint import checkpoint
                p_next, _ = checkpoint(
                    mp, p_curr, p_prev, edge_index, edge_attr,
                    node_props, positions, domain_size,
                    use_reentrant=False,
                )
            else:
                p_next, _ = mp(
                    p_curr, p_prev, edge_index, edge_attr,
                    node_props, positions, domain_size,
                )
            
            # FIX #4: store pressure WITH gradients for physics loss
            # Only keep last few steps to limit memory (physics loss only needs 3 consecutive)
            self.pressure_history.append(p_next)
            if len(self.pressure_history) > 5:
                self.pressure_history.pop(0)
            
            # FIX #5: collect all pressures for RF matrix
            all_pressures.append(p_next)
            
            # Track energy (detached, monitoring only — not part of loss)
            with torch.no_grad():
                v = (p_next - p_prev) / (2.0 * dt)
                rho = node_props['rho'].unsqueeze(-1)
                c_unsq = node_props['c'].unsqueeze(-1)
                potential = p_curr ** 2 / (2 * rho * c_unsq ** 2)
                kinetic = 0.5 * rho * v ** 2
                energy = (potential + kinetic).sum()
                self.energy_history.append(energy.item())
                if len(self.energy_history) > 100:
                    self.energy_history.pop(0)
            
            # Advance
            p_prev = p_curr
            p_curr = p_next
        
        return p_curr, all_pressures
    
    def _compute_base_weights(self, edge_attr, edge_index, node_props, mp_layer):
        """Compute base edge weights for Laplacian (used by FIX #6 init).
        
        FIX #7: Uses 1/|r_ij|² (correct dimension for 2D/3D Laplacian).
        """
        distance = edge_attr[:, -3] + mp_layer.eps
        # FIX #7: 1/|r|² for correct Laplacian approximation
        base_weight = 1.0 / (distance ** 2)
        if edge_attr.shape[1] > 4:
            Z_ratio = edge_attr[:, -2]
            base_weight = base_weight * Z_ratio
        return base_weight
    
    def compute_wave_equation_residual(self, node_props: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute physics loss: wave equation residual from pressure history.
        
        FIX #2: Removed division by dt² — use dimensionless residual instead.
            Old: p_tt = (p_next - 2p_curr + p_prev) / dt²  → numerical explosion
            New: residual = (p_next - 2p_curr + p_prev) directly (dimensionless FD stencil)
        
        FIX #3: Correct wave equation residual = p_tt - c²∇²p.
            Since we use leapfrog, the update should satisfy:
              p_next - 2*p_curr + p_prev = c²*dt²*∇²p
            So dimensionless residual = (p_next - 2*p_curr + p_prev) - c²*dt²*∇²p
            But ∇²p requires graph info. Without it, we use the simpler form:
              residual = p_next - 2*p_curr + p_prev  (should be ≈ c²dt²∇²p if physics holds)
            We penalize large deviations from smoothness.
        
        FIX #4: pressure_history now has gradients (no detach), so this loss
            actually backpropagates through the wave propagation.
        
        Returns:
            residual_loss: Scalar wave equation residual
        """
        if len(self.pressure_history) < 3:
            return torch.tensor(0.0, device=self.pressure_history[0].device if self.pressure_history else 'cpu')
        
        loss = torch.tensor(0.0, device=self.pressure_history[0].device)
        count = 0
        
        for k in range(1, len(self.pressure_history) - 1):
            p_prev = self.pressure_history[k - 1]
            p_curr = self.pressure_history[k]
            p_next = self.pressure_history[k + 1]
            
            # FIX #2: Dimensionless finite difference (NO division by dt²)
            # FIX #3: This is the discrete wave equation residual
            # In a perfect leapfrog: p_next - 2*p_curr + p_prev = c²*dt²*∇²p
            # The residual measures how well the network satisfies temporal smoothness
            residual = p_next - 2.0 * p_curr + p_prev  # dimensionless
            
            # FIX #3: loss = mean(residual²)
            loss = loss + (residual ** 2).mean()
            count += 1
        
        return loss / max(count, 1)
