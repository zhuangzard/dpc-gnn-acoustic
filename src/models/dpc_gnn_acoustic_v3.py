"""
dpc_gnn_acoustic_v3.py — Minimal physics-correct DPC-GNN-Acoustic model.

Replaces v2 with a clean architecture:
  CT HU → NodeEncoder → p0 → LeapfrogWavePropagator → DASBeamformDecoder → B-mode

All physics bugs from v1/v2 are fixed:
  - Correct Taylor initialization (zero initial velocity)
  - Correct graph Laplacian (1/dx², not 1/r)
  - Full pressure history with gradients (no detach)
  - Physical DAS beamforming (not learned-only)
  - Dimensionless physics loss (no dt² division)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .wave_propagator import LeapfrogWavePropagator
from .beamform_decoder import DASBeamformDecoder
from .components import NodeEncoder


class DPCGNNAcousticV3(nn.Module):
    """DPC-GNN-Acoustic v3: Minimal physics-correct CT → US simulation.

    Architecture:
      CT HU (N,) → [NodeEncoder] → h (N, hidden)
                  → [p0_generator] → p0 (N, 1)
                  → [LeapfrogWavePropagator] → pressure_history (T, N)
                  → [DASBeamformDecoder] → B-mode (H, W)

    Args:
        hidden_dim: Feature dimension for encoder
        dt: Time step [s]
        n_time_steps: Number of wave propagation steps
        n_elements: Number of transducer elements
        image_h: B-mode image height (lateral)
        image_w: B-mode image width (axial/depth)
        pml_thickness: PML layer count
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dt: float = 2e-8,
        n_time_steps: int = 200,
        n_elements: int = 128,
        image_h: int = 256,
        image_w: int = 512,
        pml_thickness: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.n_time_steps = n_time_steps

        # ── Node encoder: (ρ, c, α, n, HU, is_transducer) → hidden ──
        self.node_encoder = NodeEncoder(input_dim=6, hidden_dim=hidden_dim)

        # ── Initial pressure generator ──
        self.p0_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Small init → small p0
        nn.init.uniform_(self.p0_generator[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.p0_generator[-1].bias)

        # ── Wave propagator (physics-correct Leapfrog) ──
        self.propagator = LeapfrogWavePropagator(
            hidden_dim=64,
            dt=dt,
            n_time_steps=n_time_steps,
            pml_thickness=pml_thickness,
        )

        # ── Beamform decoder (DAS) ──
        self.decoder = DASBeamformDecoder(
            n_elements=n_elements,
            image_h=image_h,
            image_w=image_w,
            dt=dt,
        )

    def forward(
        self,
        hu: torch.Tensor,                     # (N,) CT HU values
        edge_index: torch.Tensor,              # (2, E) graph edges
        edge_attr: torch.Tensor,               # (E, D) edge features
        node_props: Dict[str, torch.Tensor],   # c, rho, alpha_0, n_power, dx_mean
        transducer_idx: torch.Tensor,          # (M,) transducer node indices
        positions: Optional[torch.Tensor] = None,
        domain_size: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: CT → B-mode image.

        Args:
            hu: (N,) CT Hounsfield Units
            edge_index: (2, E) graph connectivity
            edge_attr: (E, D) edge features
            node_props: Dict with 'c', 'rho', 'alpha_0', 'n_power', 'dx_mean'
            transducer_idx: (M,) indices of transducer element nodes
            positions: (N, 2) node positions
            domain_size: (2,) domain size

        Returns:
            outputs: Dictionary with:
                - 'bmode': (H, W) B-mode image
                - 'pressure_field': (N, 1) final pressure
                - 'initial_pressure': (N, 1) initial pressure
                - 'energy_history': list of energy values
        """
        N = hu.shape[0]
        device = hu.device

        c = node_props['c'].to(device)
        rho = node_props['rho'].to(device)
        alpha_0 = node_props['alpha_0'].to(device)
        n_power = node_props['n_power'].to(device)

        # ── Encode nodes ──
        is_transducer = torch.zeros(N, device=device)
        is_transducer[transducer_idx] = 1.0

        node_features = torch.stack([
            rho / 1000.0,
            c / 1540.0,
            alpha_0 / 10.0,
            n_power,
            hu / 1000.0,
            is_transducer,
        ], dim=-1)  # (N, 6)

        h = self.node_encoder(node_features)  # (N, hidden_dim)

        # ── Generate initial pressure ──
        p0 = self.p0_generator(h)  # (N, 1)

        # Focus at transducer elements only
        transducer_mask = torch.zeros(N, 1, device=device)
        transducer_mask[transducer_idx] = 1.0
        p0 = p0 * transducer_mask

        # ── Wave propagation ──
        p_final, all_pressures = self.propagator(
            p0, edge_index, edge_attr,
            node_props, positions, domain_size,
        )

        # ── Beamform to B-mode ──
        # Get transducer positions for DAS
        transducer_positions = None
        if positions is not None:
            transducer_positions = positions[transducer_idx]

        # Compute mean sound speed for DAS delay calculation
        c_mean = c.mean().item()

        bmode = self.decoder(
            all_pressures,
            transducer_idx,
            transducer_positions=transducer_positions,
            c_mean=c_mean,
            domain_size=domain_size,
        )

        return {
            'bmode': bmode,
            'pressure_field': p_final,
            'initial_pressure': p0,
            'energy_history': self.propagator.energy_history,
            'all_pressures': all_pressures,
        }

    def compute_physics_loss(self) -> torch.Tensor:
        """Compute wave equation residual loss from propagation history."""
        return self.propagator.compute_wave_equation_residual()

    def count_parameters(self) -> Dict[str, dict]:
        """Count parameters by component."""
        components = {
            'node_encoder': self.node_encoder,
            'p0_generator': self.p0_generator,
            'propagator': self.propagator,
            'decoder': self.decoder,
        }
        counts = {}
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {'total': total, 'trainable': trainable}

        counts['model_total'] = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return counts


def create_model_v3(config: dict, device: str = 'cpu') -> DPCGNNAcousticV3:
    """Factory function to create v3 model from config dict.

    Args:
        config: Configuration dictionary (from YAML)
        device: Target device

    Returns:
        model: Initialized DPCGNNAcousticV3
    """
    model_cfg = config.get('model', {})
    physics_cfg = config.get('physics', {})
    probe_cfg = config.get('probe', {})
    data_cfg = config.get('data', {})

    model = DPCGNNAcousticV3(
        hidden_dim=model_cfg.get('hidden_dim', 128),
        dt=float(physics_cfg.get('dt', 2e-8)),
        n_time_steps=int(physics_cfg.get('n_time_steps', 200)),
        n_elements=int(probe_cfg.get('n_elements', 128)),
        image_h=data_cfg.get('image_size', 512) // 2,
        image_w=data_cfg.get('image_size', 512),
        pml_thickness=int(physics_cfg.get('pml_thickness', 10)),
    ).to(device)

    # Print model summary
    counts = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"DPC-GNN-Acoustic v3 — Model Summary")
    print(f"{'='*60}")
    for name, c in counts.items():
        if name == 'model_total':
            print(f"  {'TOTAL':20s}: {c['trainable']:>10,} trainable / {c['total']:>10,} total")
        else:
            print(f"  {name:20s}: {c['trainable']:>10,} params")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    return model
