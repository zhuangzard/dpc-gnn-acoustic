"""
DPC-GNN-Acoustic V4: Main Model
CT(256×256) → GNN Encoder → (c, α, σ) → Leapfrog → Beamformer → B-mode

Architecture:
  - CNN downsample (256→64) + GNN (5-layer antisymmetric MP, dim=96, k=8)
    + CNN upsample (64→256)
  - Three output fields: speed-of-sound c, attenuation α, reflectivity σ
  - Physics-informed prior: c = c_table(HU) + c_residual
  - Deterministic Leapfrog propagator (no learnable params)
  - Differentiable DAS beamformer (no learnable params)
  - Loss computed on B-mode images vs k-Wave ground truth
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wave_propagator_v4 import AcousticLeapfrogV4
from .beamform_decoder_v4 import DifferentiableBeamformerV4


# ---------------------------------------------------------------------------
# HU → speed-of-sound lookup table (physical prior)
# ---------------------------------------------------------------------------
def hu_to_speed_of_sound(ct_hu: torch.Tensor, c_min: float = 1400.0, c_max: float = 1700.0) -> torch.Tensor:
    """
    Convert normalised CT input to speed-of-sound via affine mapping.
    CT data may be actual speed-of-sound values (normalised to [0,1]).
    Maps [0, 1] → [c_min, c_max] m/s.
    
    For uniform phantoms (ct_hu ≈ constant after normalisation), the GNN residual
    provides all spatial variation. For structured phantoms, ct_hu provides the prior.
    """
    # Affine mapping: [0, 1] → [c_min, c_max]
    c_table = ct_hu * (c_max - c_min) + c_min
    return c_table


# ---------------------------------------------------------------------------
# Antisymmetric Message-Passing Layer
# ---------------------------------------------------------------------------
class AntisymmetricMPLayer(nn.Module):
    """
    Graph message-passing with antisymmetric weight matrix.
    W_anti = W - W^T  →  msg_ij = tanh((h_i - h_j) @ W_anti)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Raw (square) weight; antisymmetric version computed on-the-fly
        self.W = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [N, D] node features
            edge_index: [2, E] source→target indices
        Returns:
            h_new: [N, D]
        """
        src, dst = edge_index  # [E]
        W_anti = self.W - self.W.t()  # antisymmetric

        diff = h[src] - h[dst]  # [E, D]
        msg = torch.tanh(diff @ W_anti)  # [E, D]

        # Aggregate messages (mean)
        agg = torch.zeros_like(h)
        count = torch.zeros(h.size(0), 1, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(dst.size(0), 1, device=h.device))
        count = count.clamp(min=1)
        agg = agg / count

        h_new = self.norm(h + agg + self.bias)
        return h_new


# ---------------------------------------------------------------------------
# GNN Encoder  (CNN↓ → GNN × L → CNN↑)
# ---------------------------------------------------------------------------
class GNNEncoder(nn.Module):
    """
    CNN downsample (256→64) → flatten to graph → k-NN edges →
    L antisymmetric MP layers → reshape → CNN upsample (64→256) →
    3 output heads: c_residual, α, σ
    """

    def __init__(self, hidden_dim: int = 96, n_mp_layers: int = 5,
                 k_local: int = 8, c_min: float = 1400.0, c_max: float = 1700.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_local = k_local
        self.c_min = c_min
        self.c_max = c_max

        # --- CNN Downsample: 1ch 256×256 → hidden_dim ch 64×64 ---
        self.down = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),   # 128
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1), nn.GELU(),  # 64
        )

        # --- GNN message-passing layers ---
        self.mp_layers = nn.ModuleList([
            AntisymmetricMPLayer(hidden_dim) for _ in range(n_mp_layers)
        ])

        # --- CNN Upsample: hidden_dim ch 64×64 → 3ch 256×256 ---
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 4, stride=2, padding=1), nn.GELU(),  # 128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),          # 256
            nn.Conv2d(32, 3, 1),  # 3 output channels: c_residual, α_raw, σ_raw
        )

        # Pre-compute grid-based edge index (cached on first forward)
        self._edge_index = None
        self._grid_size = None

    def _build_grid_knn_edges(self, H: int, W: int, k: int, device: torch.device) -> torch.Tensor:
        """Build k-NN edges on a 2D grid (spatial locality)."""
        # Generate all (y, x) coordinates
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)  # [N, 2]

        N = H * W
        # For efficiency on large grids, use local window instead of full k-NN
        src_list = []
        dst_list = []
        radius = int(math.ceil(math.sqrt(k))) + 1

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                dist_sq = dy * dy + dx * dx
                if dist_sq > radius * radius:
                    continue
                # Shift indices
                src_y = torch.arange(max(0, -dy), min(H, H - dy), device=device)
                src_x = torch.arange(max(0, -dx), min(W, W - dx), device=device)
                if len(src_y) == 0 or len(src_x) == 0:
                    continue
                sy, sx = torch.meshgrid(src_y, src_x, indexing='ij')
                src_idx = sy.reshape(-1) * W + sx.reshape(-1)
                dst_idx = (sy.reshape(-1) + dy) * W + (sx.reshape(-1) + dx)
                src_list.append(src_idx)
                dst_list.append(dst_idx)

        edge_src = torch.cat(src_list)
        edge_dst = torch.cat(dst_list)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, E]
        return edge_index

    def forward(self, ct: torch.Tensor) -> tuple:
        """
        Args:
            ct: [B, 1, 256, 256] normalised CT image
        Returns:
            c: [B, 1, 256, 256] speed of sound in m/s
            alpha: [B, 1, 256, 256] attenuation in Np/m
            sigma: [B, 1, 256, 256] reflectivity in [0, 1]
        """
        B = ct.size(0)

        # --- CNN downsample ---
        feat = self.down(ct)  # [B, D, 64, 64]
        _, D, H, W = feat.shape

        # --- Build edges (cached) ---
        if self._edge_index is None or self._grid_size != (H, W):
            self._edge_index = self._build_grid_knn_edges(H, W, self.k_local, ct.device)
            self._grid_size = (H, W)
        edge_index = self._edge_index.to(ct.device)

        # --- GNN on each sample ---
        feat_out = []
        for b in range(B):
            h = feat[b].reshape(D, -1).t()  # [N, D]  where N = H*W
            for mp in self.mp_layers:
                h = mp(h, edge_index)
            feat_out.append(h.t().reshape(D, H, W))
        feat = torch.stack(feat_out, dim=0)  # [B, D, 64, 64]

        # --- CNN upsample ---
        out = self.up(feat)  # [B, 3, 256, 256]

        c_residual = out[:, 0:1]  # residual added to physics prior
        alpha_raw = out[:, 1:2]
        sigma_raw = out[:, 2:3]

        # --- Physics prior for speed-of-sound ---
        c_table = hu_to_speed_of_sound(ct, self.c_min, self.c_max)  # [B, 1, 256, 256]
        # c_residual scaled to ±150 m/s (full range coverage for uniform phantoms)
        c = c_table + torch.tanh(c_residual) * 150.0
        c = c.clamp(self.c_min, self.c_max)

        # --- Activation for attenuation ---
        alpha = F.softplus(alpha_raw) * 10.0  # [0, ~50] Np/m

        # --- Activation for reflectivity ---
        sigma = torch.sigmoid(sigma_raw)  # [0, 1]

        return c, alpha, sigma


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class DPCGNNAcousticV4(nn.Module):
    """
    Full differentiable pipeline:
    CT → GNN Encoder → (c, α, σ) → Leapfrog Propagator → DAS Beamformer → B-mode
    
    Learnable parameters: ~150K (all in GNN encoder)
    Physics modules: zero learnable parameters
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get('model', {})
        physics_cfg = config.get('physics', {})

        # --- GNN Encoder (only learnable component) ---
        self.encoder = GNNEncoder(
            hidden_dim=model_cfg.get('hidden_dim', 96),
            n_mp_layers=model_cfg.get('n_mp_layers', 5),
            k_local=model_cfg.get('k_local', 8),
            c_min=model_cfg.get('c_min', 1400.0),
            c_max=model_cfg.get('c_max', 1700.0),
        )

        # --- Deterministic Leapfrog Propagator (no learnable params) ---
        self.propagator = AcousticLeapfrogV4(
            nx=config.get('data', {}).get('grid_resolution', 256),
            ny=config.get('data', {}).get('grid_resolution', 256),
            dx=physics_cfg.get('dx', 2.34e-4),
            dt=physics_cfg.get('dt', 2.0e-8),
            n_steps=physics_cfg.get('n_time_steps', 200),
            pml_width=physics_cfg.get('pml_width', 20),
        )

        # --- Deterministic DAS Beamformer (no learnable params) ---
        self.beamformer = DifferentiableBeamformerV4(
            n_elements=128,
            c_ref=physics_cfg.get('c0', 1540.0),
            image_size=(128, 128),
            dx=physics_cfg.get('dx', 2.34e-4),
            dt=physics_cfg.get('dt', 2.0e-8),
        )

    def forward(self, ct: torch.Tensor, source: torch.Tensor = None) -> dict:
        """
        Args:
            ct: [B, 1, 256, 256] normalised CT slice
            source: [B, n_steps] source waveform (optional, generated if None)
        Returns:
            dict with keys:
                'bmode': [B, 1, 128, 128] predicted B-mode image
                'c': [B, 1, 256, 256] speed-of-sound field
                'alpha': [B, 1, 256, 256] attenuation field
                'sigma': [B, 1, 256, 256] reflectivity field
                'sensor_data': [B, n_elements, n_steps] raw sensor data
        """
        # 1. GNN encoder: CT → tissue properties
        c, alpha, sigma = self.encoder(ct)

        # 2. Generate source if not provided
        if source is None:
            source = self._default_source(ct.device, ct.size(0))

        # 3. Leapfrog propagation: (c, α, σ, source) → sensor data
        sensor_data = self.propagator(c, alpha, sigma, source)

        # 4. DAS beamforming: sensor data → B-mode image
        bmode = self.beamformer(sensor_data)

        return {
            'bmode': bmode,
            'c': c,
            'alpha': alpha,
            'sigma': sigma,
            'sensor_data': sensor_data,
        }

    def _default_source(self, device: torch.device, batch_size: int) -> torch.Tensor:
        """Generate a default ultrasound pulse (Ricker wavelet, 5 MHz centre freq)."""
        n_steps = self.propagator.n_steps
        dt = self.propagator.dt
        f0 = 5.0e6  # 5 MHz centre frequency

        t = torch.arange(n_steps, device=device, dtype=torch.float32) * dt
        t0 = 1.5 / f0  # delay to avoid non-causal start
        arg = (math.pi * f0 * (t - t0)) ** 2
        wavelet = (1.0 - 2.0 * arg) * torch.exp(-arg)

        # Normalise
        wavelet = wavelet / (wavelet.abs().max() + 1e-12)
        return wavelet.unsqueeze(0).expand(batch_size, -1)  # [B, n_steps]

    def count_parameters(self) -> dict:
        """Count learnable parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        propagator_params = sum(p.numel() for p in self.propagator.parameters() if p.requires_grad)
        beamformer_params = sum(p.numel() for p in self.beamformer.parameters() if p.requires_grad)
        return {
            'encoder': encoder_params,
            'propagator': propagator_params,  # should be 0
            'beamformer': beamformer_params,  # should be 0
            'total': encoder_params + propagator_params + beamformer_params,
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    config = {
        'model': {'hidden_dim': 96, 'n_mp_layers': 5, 'k_local': 8,
                  'c_min': 1400, 'c_max': 1700},
        'physics': {'dx': 2.34e-4, 'dt': 2.0e-8, 'n_time_steps': 50,
                    'pml_width': 20, 'c0': 1540.0},
        'data': {'grid_resolution': 256},
    }
    model = DPCGNNAcousticV4(config)
    params = model.count_parameters()
    print(f"Parameter count: {params}")

    ct = torch.randn(1, 1, 256, 256).sigmoid()  # fake normalised CT
    out = model(ct)
    print(f"B-mode shape: {out['bmode'].shape}")
    print(f"c range: [{out['c'].min():.1f}, {out['c'].max():.1f}] m/s")
    print(f"alpha range: [{out['alpha'].min():.2f}, {out['alpha'].max():.2f}] Np/m")
    print(f"sigma range: [{out['sigma'].min():.4f}, {out['sigma'].max():.4f}]")

    # Check gradient flow
    loss = out['bmode'].sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)
    print(f"Gradient flows to encoder: {grad_ok}")
