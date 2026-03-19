"""
dpc_gnn_acoustic.py — Full DPC-GNN-Acoustic v2 Model.

Complete encoder-processor-decoder architecture for CT → Ultrasound simulation.

Pipeline:
  CT (HU) → AcousticPropertyMapper → Graph → KWaveInspiredMP stack → BeamformDecoder → B-mode

This is the top-level model that orchestrates all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .kwave_inspired_mp import KWaveInspiredMP, KWaveInspiredMPStack
from .components import NodeEncoder


class BeamformDecoder(nn.Module):
    """Converts pressure time-series at transducer elements to B-mode image.
    
    FIX #5: Now uses full RF matrix (n_elements × n_time_steps) instead of
    only the last time step. This is essential for beamforming — you need the
    full time-domain signal to apply delays.
    
    Pipeline:
      1. Extract pressure at transducer positions at ALL time steps → RF matrix
      2. Learned beamforming (delay-and-sum approximation)
      3. Envelope detection (learned)
      4. Log compression → B-mode image
    
    Args:
        n_elements: Number of transducer elements
        n_lines: Number of scanlines in output image
        n_samples: Number of depth samples per scanline
        n_time_steps: Number of time steps in RF data
        hidden_dim: Hidden dimension for the decoder MLP
    """
    
    def __init__(
        self,
        n_elements: int = 128,
        n_lines: int = 256,
        n_samples: int = 512,
        n_time_steps: int = 200,  # FIX #5: need time dimension
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_lines = n_lines
        self.n_samples = n_samples
        self.n_time_steps = n_time_steps
        
        # FIX #5: RF signal processor — operates on (n_elements, n_time_steps) matrix
        # 1D conv along time axis for each element (shared weights)
        self.rf_processor = nn.Sequential(
            nn.Conv1d(n_elements, hidden_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        
        # FIX #5: Learned beamforming weights (approximates delay-and-sum)
        # For each scanline, learn which element-time combinations to weight
        self.beamform_weights = nn.Parameter(
            torch.randn(n_lines, n_elements) * 0.01
        )
        
        # FIX #5+#6: Two separate depth mappers for the two signal paths
        # Path 1: processed RF features (hidden_dim) → depth samples
        self.processed_to_depth = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_samples),
        )
        # Path 2: beamformed RF (n_time_steps) → depth samples
        self.beamform_to_depth = nn.Sequential(
            nn.Linear(n_time_steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_samples),
        )
        
        # Log compression parameters
        self.log_gain = nn.Parameter(torch.tensor(1.0))
        self.dynamic_range = 60.0  # dB
    
    def forward(
        self,
        all_pressures: list,           # FIX #5: list of (N, 1) at each time step
        transducer_idx: torch.Tensor,  # (M,) indices of transducer nodes
    ) -> torch.Tensor:
        """Decode pressure time-series to B-mode image.
        
        FIX #5: Uses full pressure history at transducer positions,
        not just the final time step.
        
        Args:
            all_pressures: list of (N, 1) pressure fields at each time step
            transducer_idx: (M,) indices of transducer element nodes
        
        Returns:
            bmode: (n_lines, n_samples) B-mode image
        """
        # FIX #5: Build RF matrix (n_elements, n_time_steps)
        # Extract pressure at transducer positions for each time step
        n_steps = len(all_pressures)
        M = transducer_idx.shape[0]
        
        # Stack transducer pressures across time: (M, n_steps)
        rf_signals = torch.stack([
            p[transducer_idx].squeeze(-1) for p in all_pressures
        ], dim=-1)  # (M, n_steps)
        
        # Pad/truncate elements to n_elements
        if M < self.n_elements:
            rf_signals = F.pad(rf_signals, (0, 0, 0, self.n_elements - M))
        else:
            rf_signals = rf_signals[:self.n_elements]
        
        # Pad/truncate time to expected n_time_steps
        if n_steps < self.n_time_steps:
            rf_signals = F.pad(rf_signals, (0, self.n_time_steps - n_steps))
        else:
            rf_signals = rf_signals[:, :self.n_time_steps]
        
        # Process RF signals with 1D conv: (n_elements, n_time_steps) → (hidden, n_time_steps)
        # Add batch dim for conv1d
        rf_processed = self.rf_processor(rf_signals.unsqueeze(0))  # (1, hidden, T)
        rf_processed = rf_processed.squeeze(0)  # (hidden, T)
        
        # Beamforming: weighted combination across elements for each scanline
        weights = F.softmax(self.beamform_weights, dim=-1)  # (n_lines, n_elements)
        
        # Apply beamforming: for each line, weight the RF signals
        # rf_signals: (n_elements, T), weights: (n_lines, n_elements)
        rf_beamformed = weights @ rf_signals  # (n_lines, T)
        
        # FIX #6: Convert time samples to depth samples via two separate paths
        # Path 1: processed RF features → mean over time → (n_lines, hidden_dim) → depth
        processed_signal = rf_processed.T.unsqueeze(0).expand(self.n_lines, -1, -1).mean(dim=1)
        bmode = self.processed_to_depth(processed_signal)  # (n_lines, n_samples)
        # Path 2: beamformed RF → (n_lines, n_time_steps) → depth
        bmode = bmode + self.beamform_to_depth(rf_beamformed)  # (n_lines, n_samples)
        
        # Envelope detection (absolute value as simple approximation)
        envelope = torch.abs(bmode)
        
        # Log compression
        eps = 1e-8
        envelope_norm = envelope / (envelope.max() + eps)
        bmode_log = torch.log1p(self.log_gain.abs() * envelope_norm)
        
        # Normalize to [0, 1]
        bmode_out = (bmode_log - bmode_log.min()) / (bmode_log.max() - bmode_log.min() + eps)
        
        return bmode_out


class DPCGNNAcousticV2(nn.Module):
    """DPC-GNN-Acoustic v2: Full CT→US simulation model.
    
    Architecture:
    ```
    CT HU (N,)
        ↓
    [AcousticPropertyMapper] → (ρ, c, α₀, n) per node
        ↓
    [NodeEncoder] → h⁰ (N, hidden_dim)
        ↓
    [InitialPressureGenerator] → p⁰ (N, 1) from probe excitation
        ↓
    [KWaveInspiredMPStack × K steps] → p^T (N, 1)
        ↓
    [BeamformDecoder] → B-mode (n_lines, n_samples)
    ```
    
    Args:
        hidden_dim: Feature dimension
        n_mp_steps: Number of message passing time steps
        edge_dim: Raw edge feature dimension
        n_heads: Attention heads in MP
        frequency: Ultrasound frequency [Hz]
        dt: Time step [s]
        n_elements: Transducer elements
        image_size: Output B-mode image size (height)
        share_mp_weights: Share weights across MP steps
        use_dispersion: Enable dispersion correction
        use_attenuation: Enable learned attenuation
        use_pml: Enable learned PML
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_mp_steps: int = 12,
        edge_dim: int = 7,
        n_heads: int = 4,
        frequency: float = 5e6,
        dt: float = 2e-8,
        n_elements: int = 128,
        image_size: int = 512,
        share_mp_weights: bool = False,
        use_dispersion: bool = True,
        use_attenuation: bool = True,
        use_pml: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.frequency = frequency
        self.dt = dt
        
        # ── Node encoder: (ρ, c, α, n, HU, is_transducer) → hidden ──
        self.node_encoder = NodeEncoder(input_dim=6, hidden_dim=hidden_dim)
        
        # ── Initial pressure generator ──
        # Generates initial pressure from node embedding + probe config
        self.p0_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to small values
        nn.init.uniform_(self.p0_generator[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.p0_generator[-1].bias)
        
        # ── Wave propagation (k-Wave-inspired MP stack) ──
        # FIX #1: share_weights=True for 200 time steps
        self.propagator = KWaveInspiredMPStack(
            n_steps=n_mp_steps,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            n_heads=n_heads,
            frequency=frequency,
            dt=dt,
            share_weights=True if n_mp_steps > 50 else share_mp_weights,  # FIX #1
            use_dispersion=use_dispersion,
            use_attenuation=use_attenuation,
            use_pml=use_pml,
        )
        
        # ── Beamform decoder ──
        # FIX #5: pass n_time_steps for RF matrix construction
        self.decoder = BeamformDecoder(
            n_elements=n_elements,
            n_lines=image_size // 2,
            n_samples=image_size,
            n_time_steps=n_mp_steps + 1,  # FIX #5: +1 for initial p0
            hidden_dim=hidden_dim,
        )
    
    def forward(
        self,
        hu: torch.Tensor,                  # (N,) CT HU values
        edge_index: torch.Tensor,           # (2, E) graph edges
        edge_attr: torch.Tensor,            # (E, edge_dim) edge features
        node_props: Dict[str, torch.Tensor],  # Acoustic properties
        transducer_idx: torch.Tensor,        # (M,) transducer node indices
        positions: Optional[torch.Tensor] = None,
        domain_size: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: CT → B-mode image.
        
        Args:
            hu: (N,) CT Hounsfield Units
            edge_index: (2, E) graph connectivity
            edge_attr: (E, edge_dim) edge features
            node_props: Dict with 'c', 'rho', 'alpha_0', 'n_power', 'dx_mean'
            transducer_idx: (M,) indices of transducer element nodes
            positions: (N, D) node positions for PML
            domain_size: (D,) domain size for PML
        
        Returns:
            outputs: Dictionary with:
                - 'bmode': (n_lines, n_samples) B-mode image
                - 'pressure_field': (N, 1) final pressure
                - 'initial_pressure': (N, 1) initial pressure
                - 'energy_history': list of energy values
        """
        N = hu.shape[0]
        device = hu.device
        
        # ── Encode nodes ──
        c = node_props['c']
        rho = node_props['rho']
        alpha_0 = node_props['alpha_0']
        n_power = node_props['n_power']
        
        # Build node feature vector
        is_transducer = torch.zeros(N, device=device)
        is_transducer[transducer_idx] = 1.0
        
        node_features = torch.stack([
            rho / 1000.0,        # normalized density
            c / 1540.0,          # normalized sound speed
            alpha_0 / 10.0,      # normalized attenuation
            n_power,             # power law exponent
            hu / 1000.0,         # normalized HU
            is_transducer,       # transducer indicator
        ], dim=-1)  # (N, 6)
        
        h = self.node_encoder(node_features)  # (N, hidden_dim)
        
        # ── Generate initial pressure ──
        p0 = self.p0_generator(h)  # (N, 1)
        
        # Focus initial pressure at transducer elements
        transducer_mask = torch.zeros(N, 1, device=device)
        transducer_mask[transducer_idx] = 1.0
        p0 = p0 * transducer_mask  # Only excite at transducer
        
        # ── Propagate ──
        # FIX #5: propagator now returns (p_final, all_pressures)
        p_final, all_pressures = self.propagator(
            p0, edge_index, edge_attr,
            node_props, positions, domain_size,
        )
        
        # ── Decode to B-mode ──
        # FIX #5: pass full pressure history for RF matrix construction
        bmode = self.decoder(all_pressures, transducer_idx)
        
        return {
            'bmode': bmode,
            'pressure_field': p_final,
            'initial_pressure': p0,
            'energy_history': self.propagator.energy_history,
            'all_pressures': all_pressures,  # FIX #5: for external use
        }
    
    def compute_physics_loss(self) -> torch.Tensor:
        """Compute wave equation residual loss from propagation history."""
        return self.propagator.compute_wave_equation_residual()
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        
        components = {
            'node_encoder': self.node_encoder,
            'p0_generator': self.p0_generator,
            'propagator': self.propagator,
            'decoder': self.decoder,
        }
        
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {'total': total, 'trainable': trainable}
        
        counts['model_total'] = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        return counts


def create_model(config: dict, device: str = 'cpu') -> DPCGNNAcousticV2:
    """Factory function to create model from config dict.
    
    Args:
        config: Configuration dictionary (from YAML)
        device: Target device
    
    Returns:
        model: Initialized DPCGNNAcousticV2
    """
    model_cfg = config.get('model', {})
    physics_cfg = config.get('physics', {})
    probe_cfg = config.get('probe', {})
    data_cfg = config.get('data', {})
    
    model = DPCGNNAcousticV2(
        hidden_dim=model_cfg.get('hidden_dim', 128),
        # FIX #1: Use n_time_steps from physics config (200) instead of n_mp_layers (12)
        n_mp_steps=physics_cfg.get('n_time_steps', model_cfg.get('n_mp_layers', 200)),
        edge_dim=7,  # [r_vec(2), distance(1), Z_ratio(1), c_ratio(1), alpha_avg(1), n_avg(1)]
        n_heads=model_cfg.get('n_heads', 4),
        frequency=physics_cfg.get('frequency', 5e6),
        dt=physics_cfg.get('dt', 2e-8),
        n_elements=probe_cfg.get('n_elements', 128),
        image_size=data_cfg.get('image_size', 512),
        share_mp_weights=False,
        use_dispersion=model_cfg.get('use_dispersion_net', True),
        use_attenuation=model_cfg.get('use_attenuation_net', True),
        use_pml=model_cfg.get('use_pml_net', True),
    ).to(device)
    
    # Print model summary
    counts = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"DPC-GNN-Acoustic v2 — Model Summary")
    print(f"{'='*60}")
    for name, c in counts.items():
        if name == 'model_total':
            print(f"  {'TOTAL':20s}: {c['trainable']:>10,} trainable / {c['total']:>10,} total")
        else:
            print(f"  {name:20s}: {c['trainable']:>10,} params")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    return model
