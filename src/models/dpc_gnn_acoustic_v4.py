"""
DPC-GNN-Acoustic V4: Main Model
CT(256×256) → GNN Encoder → (c, α) → Leapfrog → Beamformer → B-mode

Architecture:
  - CNN downsample (256→64) + GNN (5-layer antisymmetric MP, dim=96, k=8)
    + CNN upsample (64→256)
  - Two output fields: speed-of-sound c, attenuation α
  - Physics-informed prior: c = c_table(CT) + c_residual (linear mapping)
  - Deterministic Leapfrog propagator (no learnable params)
  - Differentiable DAS beamformer (no learnable params)
  - Loss computed on B-mode images vs k-Wave ground truth

V4 Fixes (2026-03-20):
  - σ removed: source injection is physics-driven, not GNN-controlled
  - Unified coordinate system: propagator and beamformer share same dx/grid
  - Source & sensor co-located at transducer_row = pml+1
  - PML sigma_max physically correct (~1e6)
  - c_table uses direct linear mapping (no HU conversion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wave_propagator_v4 import AcousticLeapfrogV4
from .beamform_decoder_v4 import DifferentiableBeamformerV4


# ---------------------------------------------------------------------------
# CT → speed-of-sound lookup table (linear prior)
# ---------------------------------------------------------------------------
def ct_to_speed_of_sound(ct_norm: torch.Tensor, c_min: float = 1400.0,
                          c_max: float = 1700.0) -> torch.Tensor:
    """
    Direct linear mapping from normalised CT [0, 1] to speed of sound [c_min, c_max].
    No intermediate HU conversion — avoids information loss from non-invertible mapping.
    """
    return ct_norm * (c_max - c_min) + c_min


# ---------------------------------------------------------------------------
# Antisymmetric Message-Passing Layer
# ---------------------------------------------------------------------------
class AntisymmetricMPLayer(nn.Module):
    """
    Graph message-passing with antisymmetric weight matrix.
    W_anti = W - W^T  →  msg_ij = tanh((h_i - h_j) @ W_anti)
    Guarantees F_ij = -F_ji by construction.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        W_anti = self.W - self.W.t()

        diff = h[src] - h[dst]
        msg = torch.tanh(diff @ W_anti)

        agg = torch.zeros_like(h)
        count = torch.zeros(h.size(0), 1, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(-1),
                           torch.ones(dst.size(0), 1, device=h.device))
        count = count.clamp(min=1)
        agg = agg / count

        h_new = self.norm(h + agg + self.bias)
        return h_new


# ---------------------------------------------------------------------------
# Symmetric MP Layer (for ablation)
# ---------------------------------------------------------------------------
class SymmetricMPLayer(nn.Module):
    """Standard (non-antisymmetric) message passing for ablation."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):
        src, dst = edge_index
        diff = h[src] - h[dst]
        msg = torch.tanh(diff @ self.W)
        agg = torch.zeros_like(h)
        count = torch.zeros(h.size(0), 1, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(-1),
                           torch.ones(dst.size(0), 1, device=h.device))
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
    2 output heads: c_residual, α
    
    σ is removed — source injection is physics-driven.
    """

    def __init__(self, hidden_dim: int = 96, n_mp_layers: int = 5,
                 k_local: int = 8, c_min: float = 1400.0, c_max: float = 1700.0,
                 use_residual: bool = True, antisymmetric: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_local = k_local
        self.c_min = c_min
        self.c_max = c_max
        self.use_residual = use_residual

        # --- CNN Downsample: 1ch 256×256 → hidden_dim ch 64×64 ---
        self.down = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),   # 128
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1), nn.GELU(),  # 64
        )

        # --- GNN message-passing layers ---
        MPLayerClass = AntisymmetricMPLayer if antisymmetric else SymmetricMPLayer
        self.mp_layers = nn.ModuleList([
            MPLayerClass(hidden_dim) for _ in range(n_mp_layers)
        ])

        # --- CNN Upsample: hidden_dim ch 64×64 → 2ch 256×256 ---
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 4, stride=2, padding=1), nn.GELU(),  # 128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),          # 256
            nn.Conv2d(32, 2, 1),  # 2 output channels: c_residual, α_raw
        )

        self._edge_index = None
        self._grid_size = None

    def _build_grid_knn_edges(self, H: int, W: int, k: int,
                               device: torch.device) -> torch.Tensor:
        """Build k-NN edges on a 2D grid (spatial locality)."""
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
        return torch.stack([edge_src, edge_dst], dim=0)

    def forward(self, ct: torch.Tensor) -> tuple:
        """
        Args:
            ct: [B, 1, 256, 256] normalised CT image
        Returns:
            c: [B, 1, 256, 256] speed of sound in m/s
            alpha: [B, 1, 256, 256] attenuation in Np/m
        """
        B = ct.size(0)

        # --- CNN downsample ---
        feat = self.down(ct)  # [B, D, 64, 64]
        _, D, H, W = feat.shape

        # --- Build edges (cached) ---
        if self._edge_index is None or self._grid_size != (H, W):
            self._edge_index = self._build_grid_knn_edges(H, W, self.k_local,
                                                           ct.device)
            self._grid_size = (H, W)
        edge_index = self._edge_index.to(ct.device)

        # --- GNN on each sample ---
        feat_out = []
        for b in range(B):
            h = feat[b].reshape(D, -1).t()  # [N, D]
            for mp in self.mp_layers:
                h = mp(h, edge_index)
            feat_out.append(h.t().reshape(D, H, W))
        feat = torch.stack(feat_out, dim=0)  # [B, D, 64, 64]

        # --- CNN upsample ---
        out = self.up(feat)  # [B, 2, 256, 256]

        c_residual = out[:, 0:1]
        alpha_raw = out[:, 1:2]

        # --- Speed of sound: c_table + residual ---
        if self.use_residual:
            c_table = ct_to_speed_of_sound(ct, self.c_min, self.c_max)
            c = c_table + torch.tanh(c_residual) * 150.0
        else:
            c = torch.sigmoid(c_residual) * (self.c_max - self.c_min) + self.c_min
        c = c.clamp(self.c_min, self.c_max)

        # --- Attenuation ---
        alpha = F.softplus(alpha_raw) * 10.0  # [0, ~50] Np/m

        return c, alpha


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class DPCGNNAcousticV4(nn.Module):
    """
    Full differentiable pipeline:
    CT → GNN Encoder → (c, α) → Leapfrog Propagator → DAS Beamformer → B-mode
    
    Learnable parameters: ~170K (all in GNN encoder, 2 heads instead of 3)
    Physics modules: zero learnable parameters
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get('model', {})
        physics_cfg = config.get('physics', {})

        # --- Ablation flag: fix alpha to zero ---
        self.fix_alpha_zero = model_cfg.get('fix_alpha_zero', False)

        dx = physics_cfg.get('dx', 2.34e-4)
        dt = physics_cfg.get('dt', 2.0e-8)
        grid_size = config.get('data', {}).get('grid_resolution', 256)
        pml_width = physics_cfg.get('pml_width', 20)

        # --- GNN Encoder (only learnable component) ---
        self.encoder = GNNEncoder(
            hidden_dim=model_cfg.get('hidden_dim', 96),
            n_mp_layers=model_cfg.get('n_mp_layers', 5),
            k_local=model_cfg.get('k_local', 8),
            c_min=model_cfg.get('c_min', 1400.0),
            c_max=model_cfg.get('c_max', 1700.0),
            use_residual=model_cfg.get('use_residual', True),
            antisymmetric=model_cfg.get('antisymmetric', True),
        )

        # --- Deterministic Leapfrog Propagator (no learnable params) ---
        self.propagator = AcousticLeapfrogV4(
            nx=grid_size,
            ny=grid_size,
            dx=dx,
            dt=dt,
            n_steps=physics_cfg.get('n_time_steps', 200),
            pml_width=pml_width,
        )

        # --- Deterministic DAS Beamformer (no learnable params) ---
        # Element positions in V4 coordinates, pixel grid aligned to GT FOV
        self.beamformer = DifferentiableBeamformerV4(
            n_elements=128,
            c_ref=physics_cfg.get('c0', 1540.0),
            image_size=(128, 128),
            dx=dx,
            dt=dt,
            grid_size=grid_size,
            pml_size=pml_width,
            gt_dx=physics_cfg.get('gt_dx', 4.69e-4),
            gt_grid_size=physics_cfg.get('gt_grid_size', 128),
            gt_pml=physics_cfg.get('gt_pml', 20),
        )

    def forward(self, ct: torch.Tensor, source: torch.Tensor = None) -> dict:
        """
        Args:
            ct: [B, 1, 256, 256] normalised CT slice
            source: [B, n_steps] source waveform (optional)
        Returns:
            dict with keys:
                'bmode': [B, 1, 128, 128] predicted B-mode image
                'c': [B, 1, 256, 256] speed-of-sound field
                'alpha': [B, 1, 256, 256] attenuation field
                'sensor_data': [B, n_elements, n_steps] raw sensor data
        """
        # 1. GNN encoder: CT → tissue properties (c and α only)
        c, alpha = self.encoder(ct)

        # Ablation: zero out attenuation if configured
        if self.fix_alpha_zero:
            alpha = torch.zeros_like(alpha)

        # 2. Generate source if not provided
        if source is None:
            source = self._default_source(ct.device, ct.size(0))

        # 3. Leapfrog propagation: (c, α, source) → sensor data
        sensor_data = self.propagator(c, alpha, source)

        # 4. DAS beamforming: sensor data → B-mode image
        bmode = self.beamformer(sensor_data)

        return {
            'bmode': bmode,
            'c': c,
            'alpha': alpha,
            'sensor_data': sensor_data,
        }

    def _default_source(self, device: torch.device, batch_size: int) -> torch.Tensor:
        """Generate a default ultrasound pulse (Ricker wavelet, 5 MHz)."""
        n_steps = self.propagator.n_steps
        dt = self.propagator.dt
        f0 = 5.0e6

        t = torch.arange(n_steps, device=device, dtype=torch.float32) * dt
        t0 = 1.5 / f0
        arg = (math.pi * f0 * (t - t0)) ** 2
        wavelet = (1.0 - 2.0 * arg) * torch.exp(-arg)

        wavelet = wavelet / (wavelet.abs().max() + 1e-12)
        wavelet = wavelet * 1e10  # Scale to overcome dt²=4e-16
        return wavelet.unsqueeze(0).expand(batch_size, -1)

    def count_parameters(self) -> dict:
        """Count learnable parameters by component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters()
                             if p.requires_grad)
        propagator_params = sum(p.numel() for p in self.propagator.parameters()
                                if p.requires_grad)
        beamformer_params = sum(p.numel() for p in self.beamformer.parameters()
                                if p.requires_grad)
        return {
            'encoder': encoder_params,
            'propagator': propagator_params,
            'beamformer': beamformer_params,
            'total': encoder_params + propagator_params + beamformer_params,
        }


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

    ct = torch.randn(1, 1, 256, 256).sigmoid()
    out = model(ct)
    print(f"B-mode shape: {out['bmode'].shape}")
    print(f"c range: [{out['c'].min():.1f}, {out['c'].max():.1f}] m/s")
    print(f"alpha range: [{out['alpha'].min():.2f}, {out['alpha'].max():.2f}] Np/m")

    loss = out['bmode'].sum()
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.encoder.parameters()
                  if p.requires_grad)
    print(f"Gradient flows to encoder: {grad_ok}")
