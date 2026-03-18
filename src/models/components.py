"""
components.py — Sub-networks for KWaveInspiredMP.

Contains:
  1. DispersionCorrectionNet — Learns k-space-like dispersion correction
  2. AttenuationNet — Frequency-dependent power-law attenuation
  3. LearnedPML — Trainable PML absorbing boundary
  4. EdgeEncoder — Encodes edge features (geometry + physics)
  5. NodeEncoder — Encodes node features (medium properties)

Design rationale:
  k-Wave achieves accuracy through k-space pseudo-spectral methods that
  correct for numerical dispersion. We approximate this with learned
  corrections on the message-passing Laplacian. The key insight is that
  k-Wave's corrections are frequency-dependent and spatially varying —
  our networks learn these corrections from k-Wave ground truth.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DispersionCorrectionNet(nn.Module):
    """Learns dispersion correction inspired by k-Wave's k-space method.
    
    In k-Wave, the k-space correction modifies the finite-difference update:
        p^{n+1} = 2p^n - p^{n-1} + κ·(c·dt)²·∇²p
    
    where κ is the k-space correction operator (sinc-like in Fourier domain).
    
    We learn this correction as a function of:
      - Local sound speed c
      - Grid spacing (edge distances)
      - Time step dt
      - CFL number: c·dt/dx
    
    The network outputs a per-node correction factor κ ∈ (0, 2).
    
    Args:
        hidden_dim: Hidden layer dimension
        n_layers: Number of MLP layers
    """
    
    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        
        # Input: [c_local, dx_mean, dt, cfl_number, frequency_normalized]
        input_dim = 5
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize to output ~1.0 (no correction) via bias
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(
        self,
        c_local: torch.Tensor,      # (N,) sound speed at each node
        dx_mean: torch.Tensor,       # (N,) mean neighbor distance
        dt: float,                   # scalar time step
        frequency: float = 5e6,      # operating frequency
    ) -> torch.Tensor:
        """Compute per-node dispersion correction factor.
        
        Args:
            c_local: (N,) local sound speed [m/s]
            dx_mean: (N,) mean distance to neighbors [m]
            dt: Time step [s]
            frequency: Operating frequency [Hz]
        
        Returns:
            kappa: (N, 1) dispersion correction factor, centered at 1.0
        """
        N = c_local.shape[0]
        device = c_local.device
        
        # Compute CFL number per node
        cfl = c_local * dt / (dx_mean + 1e-8)  # (N,)
        
        # Normalize inputs for stable training
        features = torch.stack([
            c_local / 1540.0,                    # normalized sound speed
            dx_mean * 1e3,                        # mm scale
            torch.full((N,), dt * 1e7, device=device),  # ~1.0 scale
            cfl,                                  # CFL number (0-1)
            torch.full((N,), frequency / 5e6, device=device),  # normalized freq
        ], dim=-1)  # (N, 5)
        
        # Network outputs residual correction: κ = 1 + tanh(net(x)) * 0.5
        # This constrains κ ∈ (0.5, 1.5), centered at 1.0
        raw = self.net(features)  # (N, 1)
        kappa = 1.0 + 0.5 * torch.tanh(raw)
        
        return kappa


class AttenuationNet(nn.Module):
    """Frequency-dependent power-law attenuation network.
    
    Models k-Wave's absorption law:
        α(f) = α₀ · (f / f_ref)^y
    
    where y is the power-law exponent (typically 1.0-1.5 for tissue).
    
    The network learns corrections beyond the simple power law, capturing:
      - Non-linear frequency dependence at high frequencies
      - Tissue-specific deviations from power-law model
      - Coupling between attenuation and dispersion (Kramers-Kronig)
    
    Args:
        hidden_dim: Hidden dimension
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        
        # Input: [alpha_0, n_power, frequency_normalized, c_local, rho_local]
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize to small output (physics model dominates initially)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(
        self,
        alpha_0: torch.Tensor,    # (N,) or (E,) reference attenuation [Np/m]
        n_power: torch.Tensor,    # (N,) or (E,) power-law exponent
        frequency: float,         # operating frequency [Hz]
        c: torch.Tensor,          # (N,) or (E,) sound speed
        rho: torch.Tensor,        # (N,) or (E,) density
        f_ref: float = 1e6,
    ) -> torch.Tensor:
        """Compute learned attenuation coefficient.
        
        Returns:
            alpha_effective: (N,) or (E,) effective attenuation [Np/m]
        """
        # Physics-based attenuation (baseline)
        freq_ratio = frequency / f_ref
        alpha_physics = alpha_0 * (freq_ratio ** n_power)
        
        # Learned correction
        features = torch.stack([
            alpha_0 / 10.0,           # normalized
            n_power,
            torch.full_like(alpha_0, freq_ratio),
            c / 1540.0,
            rho / 1000.0,
        ], dim=-1)
        
        # Correction: multiplicative factor centered at 1.0
        correction = 1.0 + 0.3 * torch.tanh(self.net(features).squeeze(-1))
        
        alpha_effective = alpha_physics * correction
        
        return alpha_effective


class LearnedPML(nn.Module):
    """Learned Perfectly Matched Layer for absorbing boundaries.
    
    Instead of fixed polynomial σ-profile, learns the optimal damping
    profile from k-Wave GT data. This captures:
      - Optimal damping profile shape (better than polynomial)
      - Frequency-dependent PML performance
      - Corner reflections mitigation
    
    Args:
        hidden_dim: Hidden dimension
        max_sigma: Maximum damping coefficient
        n_pml_layers: Number of PML grid points
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        max_sigma: float = 2.0,
        n_pml_layers: int = 10,
    ):
        super().__init__()
        self.max_sigma = max_sigma
        self.n_pml_layers = n_pml_layers
        
        # Learns σ-profile as function of normalized distance to boundary
        # Input: [dist_to_boundary_normalized, c_local, frequency_norm]
        self.sigma_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in (0, 1), scaled by max_sigma
        )
        
        # Initialize to approximate polynomial profile
        # σ(d) ≈ σ_max * (1 - d/thickness)^2
        nn.init.xavier_uniform_(self.sigma_net[0].weight, gain=0.1)
    
    def forward(
        self,
        positions: torch.Tensor,     # (N, D) node positions
        domain_size: torch.Tensor,    # (D,) domain dimensions
        c_local: torch.Tensor,        # (N,) sound speed
        frequency: float = 5e6,
        dt: float = 1e-7,
    ) -> torch.Tensor:
        """Compute PML damping coefficient at each node.
        
        Args:
            positions: (N, D) node positions [m]
            domain_size: (D,) domain size [m]
            c_local: (N,) local sound speed [m/s]
            frequency: Operating frequency [Hz]
            dt: Time step [s]
        
        Returns:
            damping: (N, 1) damping factor exp(-σ·dt) ∈ (0, 1]
        """
        N, D = positions.shape
        device = positions.device
        
        # Distance to nearest boundary, per dimension
        dist_low = positions  # (N, D)
        dist_high = domain_size.unsqueeze(0) - positions  # (N, D)
        
        # Minimum distance to any boundary
        dist_min = torch.min(dist_low, dist_high)  # (N, D)
        dist_to_boundary = dist_min.min(dim=-1)[0]  # (N,)
        
        # PML thickness in physical units
        # Approximate from domain size / grid resolution
        pml_physical = domain_size.min().item() * self.n_pml_layers / 256.0  # rough estimate
        
        # Normalized distance: 0 at boundary, 1 at PML interior edge
        dist_normalized = torch.clamp(dist_to_boundary / (pml_physical + 1e-8), 0, 1)
        
        # Features for σ-net
        features = torch.stack([
            1.0 - dist_normalized,        # 1 at boundary, 0 in interior
            c_local / 1540.0,
            torch.full((N,), frequency / 5e6, device=device),
        ], dim=-1)  # (N, 3)
        
        # Compute σ
        sigma = self.sigma_net(features) * self.max_sigma  # (N, 1)
        
        # Only apply in PML region (dist < pml_physical)
        pml_mask = (dist_to_boundary < pml_physical).float().unsqueeze(-1)  # (N, 1)
        sigma = sigma * pml_mask
        
        # Damping factor: exp(-σ·dt)
        damping = torch.exp(-sigma * dt)
        
        return damping


class EdgeEncoder(nn.Module):
    """Encodes edge features combining geometry and physics.
    
    Input features:
      - Relative position vector r_ij (2D or 3D)
      - Distance |r_ij|
      - Impedance ratio Z_j/Z_i
      - Attenuation factor
      - Sound speed ratio c_j/c_i
    
    Output: Encoded edge embedding for message passing.
    
    Args:
        input_dim: Raw edge feature dimension (typically 6-8)
        hidden_dim: Output embedding dimension
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
    
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode edge features.
        
        Args:
            edge_attr: (E, input_dim) raw edge features
        
        Returns:
            edge_emb: (E, hidden_dim) edge embeddings
        """
        return self.net(edge_attr)


class NodeEncoder(nn.Module):
    """Encodes node features (medium properties + pressure state).
    
    Input: [ρ_norm, c_norm, α_norm, HU_norm, p_current, p_velocity]
    Output: Node embedding for message passing.
    
    Args:
        input_dim: Raw node feature dimension
        hidden_dim: Output embedding dimension
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.net(node_features)


class MultiHeadAttentionAggregation(nn.Module):
    """Multi-head attention for heterogeneous medium handling.
    
    In heterogeneous media, messages from different tissue types should
    be weighted differently (e.g., strong reflection at tissue-bone interface).
    Multi-head attention learns these interaction-specific weights.
    
    Args:
        hidden_dim: Feature dimension
        n_heads: Number of attention heads
        dropout: Attention dropout rate
    """
    
    def __init__(self, hidden_dim: int = 128, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x_i: torch.Tensor,      # (E, hidden_dim) target node features
        x_j: torch.Tensor,      # (E, hidden_dim) source node features
        edge_emb: torch.Tensor,  # (E, hidden_dim) edge embeddings
    ) -> torch.Tensor:
        """Compute attention-weighted messages.
        
        Args:
            x_i: (E, hidden_dim) features at target nodes
            x_j: (E, hidden_dim) features at source nodes
            edge_emb: (E, hidden_dim) edge embeddings
        
        Returns:
            messages: (E, hidden_dim) attention-weighted messages
        """
        # Query from target, key from source + edge
        q = self.q_proj(x_i)  # (E, H)
        k = self.k_proj(x_j + edge_emb)  # (E, H)
        v = self.v_proj(x_j)  # (E, H)
        
        # Reshape for multi-head
        E = q.shape[0]
        q = q.view(E, self.n_heads, self.head_dim)
        k = k.view(E, self.n_heads, self.head_dim)
        v = v.view(E, self.n_heads, self.head_dim)
        
        # Per-edge attention score (not softmax across neighbors — done in scatter)
        attn = (q * k).sum(dim=-1, keepdim=True) / self.scale  # (E, n_heads, 1)
        attn = torch.sigmoid(attn)  # Use sigmoid for per-edge gating
        attn = self.dropout(attn)
        
        # Weighted values
        messages = (attn * v).view(E, -1)  # (E, hidden_dim)
        messages = self.out_proj(messages)
        
        return messages


class TimeStepCorrector(nn.Module):
    """Learns corrections to the leapfrog time stepping scheme.
    
    The basic leapfrog scheme has O(dt²) error. This network learns
    a correction term that reduces the temporal discretization error,
    inspired by k-Wave's use of exact time integration in k-space.
    
    p^{n+1} = 2p^n - p^{n-1} + κ·(c·dt)²·L(p) + Δ_correction
    
    Args:
        hidden_dim: Hidden dimension
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # Input: [p_curr, p_prev, laplacian, c_local, dt_norm]
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize to zero correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(
        self,
        p_curr: torch.Tensor,      # (N, 1)
        p_prev: torch.Tensor,      # (N, 1)
        laplacian: torch.Tensor,    # (N, 1)
        c_local: torch.Tensor,     # (N,)
        dt: float,
    ) -> torch.Tensor:
        """Compute time-stepping correction.
        
        Returns:
            correction: (N, 1) additive correction to leapfrog update
        """
        features = torch.cat([
            p_curr,
            p_prev,
            laplacian,
            c_local.unsqueeze(-1) / 1540.0,
            torch.full_like(p_curr, dt * 1e7),
        ], dim=-1)  # (N, 5)
        
        # Small correction: 0.1 * tanh to prevent destabilizing
        correction = 0.1 * torch.tanh(self.net(features))
        
        return correction
