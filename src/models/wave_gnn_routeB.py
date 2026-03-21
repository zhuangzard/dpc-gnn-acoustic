"""
Route B: GNN learns wave propagation dynamics (replacing k-Wave FFT solver).

Architecture:
  GNN learns the SPATIAL DERIVATIVE OPERATOR (replacing FFT Laplacian).
  Time integration (Leapfrog) is a HARD CONSTRAINT with zero learnable parameters.

  Input per step:  [p(t), vx(t), vy(t)] + static [c, rho, alpha] per node
  Output per step: [dp/dt, dvx/dt, dvy/dt] (accelerations)

Pipeline:
  MaterialEncoder: MLP(c, rho, alpha -> 16-dim embedding), runs ONCE per simulation
  StateEncoder:    MLP(p_norm, vx_norm, vy_norm -> 32-dim) with physical normalization
  NodeEmbedder:    Sequential(Linear(32 + 16 + 2 -> 64), GELU())
  AntisymmetricMP: 4 layers (hidden=64, msg=64)
  Decoder:         3 separate MLPs for dp, dvx, dvy with output rescaling
  LeapfrogIntegrator: staggered — velocity update first, then pressure update

Graph: 8-connected grid (Moore neighborhood), ~524K directed edges on 256x256.
Edge features: [dx_ij, dy_ij, |r_ij|, c_avg, rho_avg] (5-dim).
PML damping: applied AFTER GNN as multiplicative mask (not learned).
Source injection: Dirichlet (gradient-preserving replacement, not additive).
Material encoder: cached per simulation (not re-run every step).
Graph: cached (same grid every time).

Parameter count target: ~175K (similar to soft tissue DPC-GNN).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

try:
    from .antisymmetric_mp import AntisymmetricMPLayer
except ImportError:
    from antisymmetric_mp import AntisymmetricMPLayer


# Physical scales for normalization (P1-4) and output rescaling (P0-3)
P_SCALE = 1e4       # typical pressure magnitude [Pa]
V_SCALE = 6.5e-3    # typical velocity magnitude [m/s]


# ---------------------------------------------------------------------------
# Grid graph builder (8-connected Moore neighborhood)
# ---------------------------------------------------------------------------
def build_moore_graph(nx: int, ny: int, dx: float,
                      device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Build 8-connected grid graph for 256x256 domain.

    Returns:
        dict with 'edge_index' (2, E), 'edge_attr' (E, 3),
        'positions' (N, 2).
    """
    N = nx * ny
    src_list = []
    dst_list = []
    dx_list = []
    dy_list = []

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

    for di, dj in offsets:
        # Valid source ranges
        i_start = max(0, -di)
        i_end = min(nx, nx - di)
        j_start = max(0, -dj)
        j_end = min(ny, ny - dj)

        if i_end <= i_start or j_end <= j_start:
            continue

        si = torch.arange(i_start, i_end, device=device)
        sj = torch.arange(j_start, j_end, device=device)
        gi, gj = torch.meshgrid(si, sj, indexing='ij')
        gi = gi.reshape(-1)
        gj = gj.reshape(-1)

        src_idx = gi * ny + gj
        dst_idx = (gi + di) * ny + (gj + dj)

        src_list.append(src_idx)
        dst_list.append(dst_idx)
        dx_list.append(torch.full_like(src_idx, dj * dx, dtype=torch.float32))
        dy_list.append(torch.full_like(src_idx, di * dx, dtype=torch.float32))

    edge_src = torch.cat(src_list)
    edge_dst = torch.cat(dst_list)
    edge_dx = torch.cat(dx_list)
    edge_dy = torch.cat(dy_list)
    edge_dist = torch.sqrt(edge_dx ** 2 + edge_dy ** 2)

    edge_index = torch.stack([edge_src, edge_dst], dim=0)  # (2, E)
    edge_attr = torch.stack([edge_dx, edge_dy, edge_dist], dim=-1)  # (E, 3)

    # Positions (N, 2) in physical coordinates
    ix = torch.arange(nx, device=device, dtype=torch.float32)
    iy = torch.arange(ny, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(ix, iy, indexing='ij')
    positions = torch.stack([gx.reshape(-1) * dx, gy.reshape(-1) * dx], dim=-1)

    return {
        'edge_index': edge_index.long(),
        'edge_attr': edge_attr,
        'positions': positions,
        'N': N,
    }


# ---------------------------------------------------------------------------
# PML mask builder (P0-2: physics-calibrated sigma_max)
# ---------------------------------------------------------------------------
def build_pml_mask(nx: int, ny: int, pml_width: int,
                   dx: float, dt: float, c_ref: float = 1540.0,
                   device: torch.device = None) -> torch.Tensor:
    """
    Build PML damping mask: 1.0 in interior, exponential decay in PML region.

    Physics-calibrated sigma_max:
      steps_per_cell = dx / (c_ref * dt)
      total_steps = pml_width * steps_per_cell
      sigma_max_per_step = ln(100) * 3.0 / total_steps  (~40dB attenuation)

    Returns:
        pml_mask: (N,) multiplicative damping factor per node.
    """
    steps_per_cell = dx / (c_ref * dt)
    total_steps = pml_width * steps_per_cell
    sigma_max = math.log(100.0) * 3.0 / total_steps  # ~0.36 for typical params

    mask_x = torch.ones(nx, device=device)
    mask_y = torch.ones(ny, device=device)

    for k in range(pml_width):
        # Quadratic grading from boundary inward
        frac = (pml_width - k) / pml_width
        sigma = sigma_max * (frac ** 2)
        damping = math.exp(-sigma)
        mask_x[k] = damping
        mask_x[nx - 1 - k] = damping
        mask_y[k] = damping
        mask_y[ny - 1 - k] = damping

    # Outer product -> 2D mask, then flatten
    pml_2d = mask_x.unsqueeze(1) * mask_y.unsqueeze(0)  # (nx, ny)
    return pml_2d.reshape(-1)  # (N,)


# ---------------------------------------------------------------------------
# Material Encoder (runs ONCE per simulation)
# ---------------------------------------------------------------------------
class MaterialEncoder(nn.Module):
    """MLP: (c, rho, alpha) -> 16-dim material embedding."""

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, c: torch.Tensor, rho: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (N,) speed of sound [m/s], normalized internally.
            rho: (N,) density [kg/m^3], normalized internally.
            alpha: (N,) attenuation [Np/m], normalized internally.

        Returns:
            emb: (N, out_dim) material embedding.
        """
        # Normalize to ~O(1) range for stable MLP input
        c_norm = (c - 1500.0) / 200.0
        rho_norm = (rho - 1000.0) / 200.0
        alpha_norm = alpha / 5.0

        x = torch.stack([c_norm, rho_norm, alpha_norm], dim=-1)  # (N, 3)
        return self.net(x)


# ---------------------------------------------------------------------------
# State Encoder (P1-4: physical normalization)
# ---------------------------------------------------------------------------
class StateEncoder(nn.Module):
    """MLP: (p_norm, vx_norm, vy_norm) -> 32-dim state embedding.

    Normalizes inputs by physical scales before encoding:
      p_norm = p / P_SCALE,  vx_norm = vx / V_SCALE,  vy_norm = vy / V_SCALE
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (N, 3) = [p, vx, vy] in PHYSICAL units.

        Returns:
            emb: (N, out_dim) state embedding.
        """
        # P1-4: Normalize to ~O(1) before encoding
        state_norm = state.clone()
        state_norm[:, 0] = state_norm[:, 0] / P_SCALE   # p ~ 1e4 -> O(1)
        state_norm[:, 1] = state_norm[:, 1] / V_SCALE   # vx ~ 6.5e-3 -> O(1)
        state_norm[:, 2] = state_norm[:, 2] / V_SCALE   # vy ~ 6.5e-3 -> O(1)
        return self.net(state_norm)


# ---------------------------------------------------------------------------
# Acceleration Decoders (P0-3: output rescaling to physical units)
# ---------------------------------------------------------------------------
class AccelerationDecoder(nn.Module):
    """Separate MLP heads for dp/dt, dvx/dt, dvy/dt.

    MLP outputs O(1) values, then rescaled to physical units:
      dp_physical  = dp_raw * (P_SCALE / dt)
      dvx_physical = dvx_raw * (V_SCALE / dt)
      dvy_physical = dvy_raw * (V_SCALE / dt)
    """

    def __init__(self, in_dim: int = 64, dt: float = 4.0e-8):
        super().__init__()
        self.dp_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.dvx_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.dvy_head = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, 1))

        # P0-3: Output scales so MLP outputs O(1) values
        self.dp_scale = P_SCALE / dt     # ~2.5e11
        self.dv_scale = V_SCALE / dt     # ~1.625e5

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, in_dim) node features after MP.

        Returns:
            dp: (N,) pressure acceleration [Pa/s].
            dvx: (N,) x-velocity acceleration [m/s^2].
            dvy: (N,) y-velocity acceleration [m/s^2].
        """
        dp_raw = self.dp_head(h).squeeze(-1)
        dvx_raw = self.dvx_head(h).squeeze(-1)
        dvy_raw = self.dvy_head(h).squeeze(-1)

        return (dp_raw * self.dp_scale,
                dvx_raw * self.dv_scale,
                dvy_raw * self.dv_scale)


# ---------------------------------------------------------------------------
# Main Model: WaveGNNRouteB
# ---------------------------------------------------------------------------
class WaveGNNRouteB(nn.Module):
    """
    GNN learns wave propagation dynamics on 256x256 grid.

    GNN predicts spatial derivatives (accelerations) at each time step.
    Leapfrog integrator advances state in time (hard physics constraint).
    PML and source injection applied outside GNN (not learned).

    Args:
        nx, ny: Grid dimensions.
        dx: Grid spacing [m].
        dt: Time step [s].
        pml_width: PML region width [cells].
        hidden_dim: GNN node feature dimension.
        n_mp_layers: Number of antisymmetric MP layers.
        material_dim: Material embedding dimension.
        state_dim: State embedding dimension.
        edge_dim: Edge feature dimension (3 base + 2 material = 5).
        n_elements: Number of transducer elements.
        checkpoint_every: Gradient checkpointing interval (0 = disabled).
        c_ref: Reference speed of sound for PML calibration [m/s].
    """

    def __init__(self, nx: int = 256, ny: int = 256, dx: float = 2.34e-4,
                 dt: float = 4.0e-8, pml_width: int = 20,
                 hidden_dim: int = 80, n_mp_layers: int = 4,
                 material_dim: int = 16, state_dim: int = 32,
                 edge_dim: int = 5, n_elements: int = 128,
                 checkpoint_every: int = 50, c_ref: float = 1540.0):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.pml_width = pml_width
        self.hidden_dim = hidden_dim
        self.n_elements = n_elements
        self.checkpoint_every = checkpoint_every
        self.c_ref = c_ref

        # --- Learnable modules ---
        self.material_encoder = MaterialEncoder(out_dim=material_dim)
        self.state_encoder = StateEncoder(out_dim=state_dim)

        # P2-1: Node embedder with GELU activation
        self.node_embedder = nn.Sequential(
            nn.Linear(state_dim + material_dim + 2, hidden_dim),
            nn.GELU(),
        )

        # Antisymmetric MP layers
        self.mp_layers = nn.ModuleList([
            AntisymmetricMPLayer(node_dim=hidden_dim, edge_dim=edge_dim, msg_dim=hidden_dim)
            for _ in range(n_mp_layers)
        ])

        # P0-3: Acceleration decoder with output rescaling
        self.decoder = AccelerationDecoder(in_dim=hidden_dim, dt=dt)

        # --- Cached (non-learnable) ---
        self._graph = None
        self._pml_mask = None
        self._pos_enc = None
        self._sensor_idx = None

    def _ensure_graph(self, device: torch.device) -> None:
        """Build and cache graph, PML mask, positional encoding, sensor indices."""
        if self._graph is not None and self._graph['edge_index'].device == device:
            return

        self._graph = build_moore_graph(self.nx, self.ny, self.dx, device)
        # P0-2: Physics-calibrated PML
        self._pml_mask = build_pml_mask(
            self.nx, self.ny, self.pml_width,
            dx=self.dx, dt=self.dt, c_ref=self.c_ref, device=device)

        # Positional encoding: normalized grid coordinates
        ix = torch.arange(self.nx, device=device, dtype=torch.float32) / self.nx
        iy = torch.arange(self.ny, device=device, dtype=torch.float32) / self.ny
        gx, gy = torch.meshgrid(ix, iy, indexing='ij')
        self._pos_enc = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (N, 2)

        # Sensor indices: transducer row (pml+1), centered columns
        n_elem = min(self.n_elements, self.ny - 2 * self.pml_width)
        start_col = (self.ny - n_elem) // 2
        sensor_row = self.pml_width + 1
        self._sensor_idx = torch.arange(n_elem, device=device) + sensor_row * self.ny + start_col

    def _augment_edge_attr(self, edge_attr: torch.Tensor,
                           edge_index: torch.Tensor,
                           c: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """
        Augment base edge features (dx, dy, dist) with material averages.

        Returns:
            edge_attr_full: (E, 5) = [dx, dy, dist, c_avg_norm, rho_avg_norm].
        """
        src, dst = edge_index
        c_avg = 0.5 * (c[src] + c[dst])
        rho_avg = 0.5 * (rho[src] + rho[dst])

        # Normalize material features to ~O(1)
        c_avg_norm = (c_avg - 1500.0) / 200.0
        rho_avg_norm = (rho_avg - 1000.0) / 200.0

        return torch.cat([
            edge_attr,                              # (E, 3): dx, dy, dist
            c_avg_norm.unsqueeze(-1),               # (E, 1)
            rho_avg_norm.unsqueeze(-1),             # (E, 1)
        ], dim=-1)  # (E, 5)

    def _gnn_step(self, state: torch.Tensor, material_emb: torch.Tensor,
                  edge_index: torch.Tensor, edge_attr_full: torch.Tensor,
                  pos_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single GNN forward pass: state -> accelerations.

        Args:
            state: (N, 3) = [p, vx, vy] in physical units.
            material_emb: (N, material_dim) cached material embedding.
            edge_index: (2, E).
            edge_attr_full: (E, 5).
            pos_enc: (N, 2).

        Returns:
            dp, dvx, dvy: (N,) each, predicted accelerations in physical units.
        """
        # P1-4: StateEncoder normalizes internally
        state_emb = self.state_encoder(state)  # (N, 32)
        node_feat = torch.cat([state_emb, material_emb, pos_enc], dim=-1)  # (N, 50)
        h = self.node_embedder(node_feat)  # (N, hidden_dim)

        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_attr_full)

        # P0-3: Decoder rescales O(1) MLP output to physical units
        dp, dvx, dvy = self.decoder(h)
        return dp, dvx, dvy

    def _leapfrog_step(self, p: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor,
                       material_emb: torch.Tensor,
                       edge_index: torch.Tensor, edge_attr_full: torch.Tensor,
                       pos_enc: torch.Tensor, dt: float, pml_mask: torch.Tensor,
                       source_val: Optional[torch.Tensor] = None,
                       source_idx: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Staggered Leapfrog time integration (P1-3).

        vx/vy are at half-integer time steps relative to p.

        Step 1: GNN call with (p, vx, vy) -> dvx, dvy
                vx_new = vx + dt * dvx
                vy_new = vy + dt * dvy
        Step 2: GNN call with (p, vx_new, vy_new) -> dp
                p_new  = p  + dt * dp

        Then apply PML damping and source injection (Dirichlet).
        """
        # --- Step 1: velocity update ---
        state_v = torch.stack([p, vx, vy], dim=-1)
        _, dvx, dvy = self._gnn_step(state_v, material_emb,
                                       edge_index, edge_attr_full, pos_enc)
        vx = vx + dt * dvx
        vy = vy + dt * dvy

        # --- Step 2: pressure update using updated velocity ---
        state_p = torch.stack([p, vx, vy], dim=-1)
        dp, _, _ = self._gnn_step(state_p, material_emb,
                                   edge_index, edge_attr_full, pos_enc)
        p = p + dt * dp

        # PML damping (multiplicative, not learned)
        p = p * pml_mask
        vx = vx * pml_mask
        vy = vy * pml_mask

        # P1-5: Source injection — Dirichlet (gradient-preserving replacement)
        if source_val is not None and source_idx is not None:
            mask = torch.zeros_like(p)
            mask[source_idx] = 1.0
            p = p * (1.0 - mask) + source_val * mask

        return p, vx, vy

    def step_from_frames(self, p: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor,
                         c: torch.Tensor, rho: torch.Tensor, alpha: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step prediction for training (P0-1, P1-2).

        Takes current state [p, vx, vy] + material maps -> predicted next state.
        Uses staggered Leapfrog: velocity update then pressure update.
        No PML or source injection (training uses interior frames).

        Args:
            p: (N,) pressure [Pa].
            vx: (N,) x-velocity [m/s].
            vy: (N,) y-velocity [m/s].
            c: (N,) speed of sound [m/s].
            rho: (N,) density [kg/m^3].
            alpha: (N,) attenuation [Np/m].

        Returns:
            p_next, vx_next, vy_next: (N,) each, predicted next state.
        """
        device = p.device
        self._ensure_graph(device)

        edge_index = self._graph['edge_index']
        edge_attr_base = self._graph['edge_attr']
        edge_attr_full = self._augment_edge_attr(edge_attr_base, edge_index, c, rho)
        pos_enc = self._pos_enc

        # Material embedding (caller should cache for multi-step)
        material_emb = self.material_encoder(c, rho, alpha)

        dt = self.dt

        # Staggered Leapfrog (P1-3):
        # Step 1: velocity update
        state_v = torch.stack([p, vx, vy], dim=-1)
        _, dvx, dvy = self._gnn_step(state_v, material_emb,
                                       edge_index, edge_attr_full, pos_enc)
        vx_next = vx + dt * dvx
        vy_next = vy + dt * dvy

        # Step 2: pressure update with updated velocity
        state_p = torch.stack([p, vx_next, vy_next], dim=-1)
        dp, _, _ = self._gnn_step(state_p, material_emb,
                                   edge_index, edge_attr_full, pos_enc)
        p_next = p + dt * dp

        return p_next, vx_next, vy_next

    def forward(self, c: torch.Tensor, rho: torch.Tensor, alpha: torch.Tensor,
                source: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Run full wave simulation.

        Args:
            c: (nx, ny) speed of sound [m/s].
            rho: (nx, ny) density [kg/m^3].
            alpha: (nx, ny) attenuation [Np/m].
            source: (n_steps,) source waveform amplitude.
            n_steps: Number of time steps to simulate.

        Returns:
            sensor_data: (n_elements, n_steps) pressure recorded at sensor row.
        """
        device = c.device
        self._ensure_graph(device)

        N = self.nx * self.ny
        c_flat = c.reshape(-1)        # (N,)
        rho_flat = rho.reshape(-1)    # (N,)
        alpha_flat = alpha.reshape(-1)  # (N,)

        # --- Material embedding (cached, computed ONCE) ---
        material_emb = self.material_encoder(c_flat, rho_flat, alpha_flat)  # (N, 16)

        # --- Augmented edge features (cached per simulation) ---
        edge_index = self._graph['edge_index']
        edge_attr_base = self._graph['edge_attr']
        edge_attr_full = self._augment_edge_attr(edge_attr_base, edge_index,
                                                  c_flat, rho_flat)

        # --- Source indices (transducer row) ---
        n_elem = min(self.n_elements, self.ny - 2 * self.pml_width)
        start_col = (self.ny - n_elem) // 2
        sensor_row = self.pml_width + 1
        source_idx = torch.arange(n_elem, device=device) + sensor_row * self.ny + start_col

        # --- Initialize state ---
        p = torch.zeros(N, device=device, dtype=torch.float32)
        vx = torch.zeros(N, device=device, dtype=torch.float32)
        vy = torch.zeros(N, device=device, dtype=torch.float32)

        pml_mask = self._pml_mask
        pos_enc = self._pos_enc

        # --- Sensor recording ---
        sensor_idx = self._sensor_idx
        n_sensor = sensor_idx.size(0)
        sensor_data = torch.zeros(n_sensor, n_steps, device=device, dtype=torch.float32)

        # --- Time stepping (staggered Leapfrog) ---
        for t in range(n_steps):
            src_val = source[t] if t < source.size(0) else None

            p, vx, vy = self._leapfrog_step(
                p, vx, vy, material_emb,
                edge_index, edge_attr_full, pos_enc,
                self.dt, pml_mask,
                source_val=src_val, source_idx=source_idx)

            # Record at sensor locations
            sensor_data[:, t] = p[sensor_idx]

        return sensor_data

    def count_parameters(self) -> dict:
        """Count learnable parameters by component."""
        mat_params = sum(p.numel() for p in self.material_encoder.parameters())
        state_params = sum(p.numel() for p in self.state_encoder.parameters())
        embed_params = sum(p.numel() for p in self.node_embedder.parameters())
        mp_params = sum(p.numel() for p in self.mp_layers.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total = mat_params + state_params + embed_params + mp_params + dec_params
        return {
            'material_encoder': mat_params,
            'state_encoder': state_params,
            'node_embedder': embed_params,
            'mp_layers': mp_params,
            'decoder': dec_params,
            'total': total,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("WaveGNNRouteB — Smoke Test")
    print("=" * 60)

    model = WaveGNNRouteB(
        nx=64, ny=64,  # Small grid for testing
        dx=2.34e-4, dt=4.0e-8,
        pml_width=10, hidden_dim=80, n_mp_layers=4,
        n_elements=32, checkpoint_every=0,
    )

    params = model.count_parameters()
    print(f"\nParameter count: {params}")
    print(f"Total: {params['total']:,}")

    # Dummy inputs
    c = torch.ones(64, 64) * 1540.0
    rho = torch.ones(64, 64) * 1000.0
    alpha = torch.ones(64, 64) * 2.0
    source = torch.sin(torch.linspace(0, 4 * math.pi, 5))

    print(f"\nRunning forward pass (64x64 grid, 5 steps)...")
    sensor_data = model(c, rho, alpha, source, n_steps=5)
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Sensor data range: [{sensor_data.min():.6f}, {sensor_data.max():.6f}]")

    # Gradient check on forward()
    loss = sensor_data.sum()
    loss.backward()
    grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.parameters() if p.requires_grad)
    print(f"Gradient flows to all parameters (forward): {grad_ok}")

    # Reset grads for step_from_frames test
    model.zero_grad()

    # Test step_from_frames (P0-1, P1-2)
    print(f"\nTesting step_from_frames()...")
    N = 64 * 64
    p_in = torch.randn(N) * P_SCALE * 0.01
    vx_in = torch.randn(N) * V_SCALE * 0.01
    vy_in = torch.randn(N) * V_SCALE * 0.01
    c_flat = torch.ones(N) * 1540.0
    rho_flat = torch.ones(N) * 1000.0
    alpha_flat = torch.ones(N) * 2.0

    p_in.requires_grad_(True)
    p_next, vx_next, vy_next = model.step_from_frames(
        p_in, vx_in, vy_in, c_flat, rho_flat, alpha_flat)
    print(f"  p_next: mean={p_next.mean():.4e}, std={p_next.std():.4e}")
    print(f"  vx_next: mean={vx_next.mean():.4e}, std={vx_next.std():.4e}")

    step_loss = p_next.sum()
    step_loss.backward()
    step_grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
    print(f"  Gradient flows to all parameters (step_from_frames): {step_grad_ok}")

    # --- Comprehensive gradient check (P2-6) ---
    print(f"\n--- Comprehensive Gradient Sanity Check ---")
    model.zero_grad()
    p_test = torch.randn(N) * 100.0
    vx_test = torch.randn(N) * 1e-4
    vy_test = torch.randn(N) * 1e-4

    p_out, vx_out, vy_out = model.step_from_frames(
        p_test, vx_test, vy_test, c_flat, rho_flat, alpha_flat)
    test_loss = p_out.pow(2).mean() + vx_out.pow(2).mean() + vy_out.pow(2).mean()
    test_loss.backward()

    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or param.grad.abs().sum() == 0:
                zero_grad_params.append(name)

    if zero_grad_params:
        print(f"  FAIL: {len(zero_grad_params)} parameters have zero gradient:")
        for name in zero_grad_params:
            print(f"    - {name}")
        raise RuntimeError(
            f"Gradient sanity check failed: {len(zero_grad_params)} parameters "
            f"have zero gradient: {zero_grad_params}")
    else:
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  PASS: All {total_params} learnable parameters have non-zero gradient.")

    # Full 256x256 parameter count
    print(f"\n--- Full 256x256 parameter count ---")
    model_full = WaveGNNRouteB(nx=256, ny=256, dx=2.34e-4, dt=4.0e-8)
    params_full = model_full.count_parameters()
    print(f"Total parameters: {params_full['total']:,}")
    for k, v in params_full.items():
        if k != 'total':
            print(f"  {k}: {v:,}")

    print("\nSmoke test PASSED.")
