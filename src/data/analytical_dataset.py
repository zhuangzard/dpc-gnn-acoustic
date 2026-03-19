"""
analytical_dataset.py — Analytical solution dataset for wave propagation validation.

Provides 3 scenarios with exact analytical solutions as ground truth:
  1. Point source (2D cylindrical wave, Green's function)
  2. Plane wave propagation
  3. Interface reflection/transmission (two-layer medium)

Each scenario generates pressure fields p(x,y,t) that satisfy the 2D wave equation
exactly, providing a reliable GT for validating GNN wave propagation.
"""

import os
import sys
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.graph_builder import (
    build_2d_grid_graph,
    augment_edge_attr_with_physics,
    create_transducer_indices,
)


# ──────────────────────────────────────────────────────────
# Analytical solutions (all operate on numpy for precision)
# ──────────────────────────────────────────────────────────

def _gaussian_modulated_sinusoid(t: np.ndarray, f0: float, t0: float, sigma: float) -> np.ndarray:
    """Gaussian-windowed sinusoidal pulse.
    
    f(t) = sin(2π f0 t) * exp(-((t - t0) / σ)²)
    """
    return np.sin(2.0 * np.pi * f0 * t) * np.exp(-((t - t0) / sigma) ** 2)


def analytical_point_source(
    nx: int, ny: int,
    Lx: float, Ly: float,
    x0: float, y0: float,
    c0: float, f0: float,
    n_steps: int, dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D point source (cylindrical wave) analytical solution.
    
    In 2D, the Green's function for the wave equation gives:
        p(r, t) = Re[ H0^(2)(k r) ] * source(t)  (frequency domain)
    
    For the time domain with a narrowband pulse, the far-field approximation:
        p(r, t) ≈ f(t - r/c0) / sqrt(max(r, r_min))
    
    where f is the source wavelet and 1/sqrt(r) is the 2D cylindrical spreading.
    
    Args:
        nx, ny: Grid resolution
        Lx, Ly: Physical domain size [m]
        x0, y0: Source position [m]
        c0: Sound speed [m/s]
        f0: Center frequency [Hz]
        n_steps: Number of time steps
        dt: Time step [s]
        
    Returns:
        c_field: (ny, nx) sound speed field (uniform)
        pressure_history: (n_steps, ny, nx) pressure at each time step
    """
    dx = Lx / nx
    dy = Ly / ny
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    
    # Distance from source
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    r_min = dx * 0.5  # Regularize singularity at source
    R_safe = np.maximum(R, r_min)
    
    # Cylindrical spreading factor: 1/sqrt(r) for 2D
    spreading = 1.0 / np.sqrt(R_safe)
    
    # Source wavelet parameters
    period = 1.0 / f0
    sigma = 1.5 * period  # Width of Gaussian envelope
    t0 = 3.0 * sigma      # Delay so pulse starts near zero
    
    # Compute pressure at each time step
    pressure_history = np.zeros((n_steps, ny, nx), dtype=np.float64)
    
    for step in range(n_steps):
        t = step * dt
        # Retarded time: t_ret = t - r/c0
        t_ret = t - R_safe / c0
        # Source wavelet at retarded time
        wavelet = _gaussian_modulated_sinusoid(t_ret, f0, t0, sigma)
        # Pressure = spreading * wavelet
        pressure_history[step] = spreading * wavelet
    
    # Normalize peak amplitude to 1.0
    peak = np.max(np.abs(pressure_history))
    if peak > 0:
        pressure_history /= peak
    
    c_field = np.full((ny, nx), c0, dtype=np.float64)
    return c_field, pressure_history


def analytical_plane_wave(
    nx: int, ny: int,
    Lx: float, Ly: float,
    c0: float, f0: float,
    theta: float,
    n_steps: int, dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plane wave propagation at angle theta from y-axis.
    
    p(x, y, t) = f(t - (x sin θ + y cos θ) / c0)
    
    where f is a Gaussian-modulated sinusoid. θ=0 → propagation along +y.
    
    Args:
        nx, ny: Grid resolution
        Lx, Ly: Physical domain size [m]
        c0: Sound speed [m/s]
        f0: Center frequency [Hz]
        theta: Propagation angle from y-axis [radians]
        n_steps: Number of time steps
        dt: Time step [s]
        
    Returns:
        c_field: (ny, nx) sound speed field (uniform)
        pressure_history: (n_steps, ny, nx)
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    
    # Projection along propagation direction
    proj = X * np.sin(theta) + Y * np.cos(theta)
    
    # Source wavelet parameters
    period = 1.0 / f0
    sigma = 1.5 * period
    t0 = 3.0 * sigma
    
    pressure_history = np.zeros((n_steps, ny, nx), dtype=np.float64)
    
    for step in range(n_steps):
        t = step * dt
        t_ret = t - proj / c0
        pressure_history[step] = _gaussian_modulated_sinusoid(t_ret, f0, t0, sigma)
    
    peak = np.max(np.abs(pressure_history))
    if peak > 0:
        pressure_history /= peak
    
    c_field = np.full((ny, nx), c0, dtype=np.float64)
    return c_field, pressure_history


def analytical_interface_reflection(
    nx: int, ny: int,
    Lx: float, Ly: float,
    c1: float, c2: float,
    rho1: float, rho2: float,
    interface_y: float,
    f0: float,
    n_steps: int, dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plane wave reflection/transmission at a flat interface (normal incidence).
    
    Medium 1 (y < interface_y): c1, rho1
    Medium 2 (y >= interface_y): c2, rho2
    
    Incident plane wave traveling in +y direction.
    
    Reflection coefficient: R = (Z2 - Z1) / (Z2 + Z1)
    Transmission coefficient: T = 2*Z2 / (Z2 + Z1)
    where Z = rho * c (acoustic impedance)
    
    In medium 1: p = p_inc(t - y/c1) + R * p_inc(t - (2*y_if - y)/c1)
       (incident + reflected, reflected wave travels in -y with path via interface)
    In medium 2: p = T * p_inc(t - y_if/c1 - (y - y_if)/c2)  (scaled by pressure T)
    
    Note: Pressure transmission coefficient T_p = 2*Z2/(Z2+Z1) for pressure.
    Actually for normal incidence:
        R_p = (Z2 - Z1)/(Z2 + Z1)    (pressure reflection)
        T_p = 2*Z2/(Z2 + Z1)         (pressure transmission)
    
    Args:
        nx, ny: Grid resolution
        Lx, Ly: Physical domain size [m]
        c1, c2: Sound speeds [m/s]
        rho1, rho2: Densities [kg/m³]
        interface_y: Interface position [m]
        f0: Center frequency [Hz]
        n_steps: Number of time steps
        dt: Time step [s]
        
    Returns:
        c_field: (ny, nx) sound speed field (two-layer)
        rho_field: (ny, nx) density field (two-layer)
        pressure_history: (n_steps, ny, nx)
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)  # (ny, nx)
    
    # Acoustic impedances
    Z1 = rho1 * c1
    Z2 = rho2 * c2
    
    # Reflection and transmission coefficients (pressure)
    R_p = (Z2 - Z1) / (Z2 + Z1)
    T_p = 2.0 * Z2 / (Z2 + Z1)
    
    # Source wavelet parameters
    period = 1.0 / f0
    sigma = 1.5 * period
    t0 = 3.0 * sigma
    
    # Masks
    mask1 = Y < interface_y   # Medium 1
    mask2 = ~mask1             # Medium 2
    
    pressure_history = np.zeros((n_steps, ny, nx), dtype=np.float64)
    
    for step in range(n_steps):
        t = step * dt
        p = np.zeros((ny, nx), dtype=np.float64)
        
        # Medium 1: incident wave + reflected wave
        # Incident: f(t - y/c1)
        t_inc = t - Y / c1
        p_inc = _gaussian_modulated_sinusoid(t_inc, f0, t0, sigma)
        
        # Reflected: R * f(t - (2*y_if - y)/c1)
        # The reflected wave at position y has traveled: y (down to interface) + (y_if - y) back up
        # Total path in medium 1 = 2*y_if - y  (for y < y_if)
        t_ref = t - (2.0 * interface_y - Y) / c1
        p_ref = R_p * _gaussian_modulated_sinusoid(t_ref, f0, t0, sigma)
        
        p[mask1] = (p_inc + p_ref)[mask1]
        
        # Medium 2: transmitted wave
        # Travel time: y_if/c1 + (y - y_if)/c2
        t_trans = t - interface_y / c1 - (Y - interface_y) / c2
        p_trans = T_p * _gaussian_modulated_sinusoid(t_trans, f0, t0, sigma)
        p[mask2] = p_trans[mask2]
        
        pressure_history[step] = p
    
    # Normalize
    peak = np.max(np.abs(pressure_history))
    if peak > 0:
        pressure_history /= peak
    
    # Build fields
    c_field = np.where(Y < interface_y, c1, c2)
    rho_field = np.where(Y < interface_y, rho1, rho2)
    
    return c_field, rho_field, pressure_history


# ──────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────

class AnalyticalDataset(Dataset):
    """Dataset with analytical solutions for wave propagation validation.
    
    Generates 3 types of scenarios (100 samples each = 300 total):
      - Point source (cylindrical wave): samples 0-99
      - Plane wave: samples 100-199
      - Interface reflection: samples 200-299
    
    Each sample returns a dict compatible with KWaveDataset format plus
    'pressure_gt' for direct pressure comparison.
    
    Args:
        split: 'train', 'val', or 'test'
        grid_resolution: Grid size (nx = ny)
        domain_size: Physical domain [m]
        n_time_steps: Number of time steps
        dt: Time step [s]
        n_elements: Number of transducer elements
        samples_per_scenario: Samples per scenario type
    """
    
    def __init__(
        self,
        split: str = 'train',
        grid_resolution: int = 256,
        domain_size: Tuple[float, float] = (0.06, 0.06),
        n_time_steps: int = 200,
        dt: float = 2e-8,
        n_elements: int = 128,
        k_local: int = 8,
        samples_per_scenario: int = 100,
        frequency: float = 5e6,  # ignored, each sample has its own
    ):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.domain_size = domain_size
        self.n_time_steps = n_time_steps
        self.dt = dt
        self.n_elements = n_elements
        self.k_local = k_local
        self.samples_per_scenario = samples_per_scenario
        
        total = 3 * samples_per_scenario  # 300 total
        
        # Split: 80% train, 10% val, 10% test
        indices = list(range(total))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        
        n_train = int(total * 0.8)
        n_val = int(total * 0.1)
        
        if split == 'train':
            self.indices = sorted(indices[:n_train])
        elif split == 'val':
            self.indices = sorted(indices[n_train:n_train + n_val])
        else:
            self.indices = sorted(indices[n_train + n_val:])
        
        print(f"[AnalyticalDataset] {split}: {len(self.indices)} samples "
              f"({samples_per_scenario}×3 scenarios)")
        
        # Cache graph (shared structure)
        self._graph_cache = None
    
    def _get_graph(self) -> Dict[str, torch.Tensor]:
        if self._graph_cache is None:
            self._graph_cache = build_2d_grid_graph(
                nx=self.grid_resolution,
                ny=self.grid_resolution,
                domain_size=self.domain_size,
                k_local=self.k_local,
            )
        return self._graph_cache
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _sample_params(self, global_idx: int) -> Dict:
        """Generate deterministic random parameters for a given sample index."""
        rng = np.random.RandomState(global_idx * 137 + 7)
        
        scenario_type = global_idx // self.samples_per_scenario  # 0, 1, 2
        
        Lx, Ly = self.domain_size
        nx = ny = self.grid_resolution
        
        # Frequency: 1-3 MHz
        f0 = rng.uniform(1e6, 3e6)
        
        if scenario_type == 0:
            # Point source
            # Source position: away from boundaries (10%-90% of domain)
            x0 = rng.uniform(0.1 * Lx, 0.9 * Lx)
            y0 = rng.uniform(0.1 * Ly, 0.9 * Ly)
            # Sound speed: typical tissue range
            c0 = rng.uniform(1400.0, 1600.0)
            return {
                'type': 'point_source',
                'f0': f0, 'c0': c0,
                'x0': x0, 'y0': y0,
            }
        
        elif scenario_type == 1:
            # Plane wave
            c0 = rng.uniform(1400.0, 1600.0)
            # Angle: -30° to +30° from y-axis
            theta = rng.uniform(-np.pi / 6, np.pi / 6)
            return {
                'type': 'plane_wave',
                'f0': f0, 'c0': c0, 'theta': theta,
            }
        
        else:
            # Interface reflection
            c1 = rng.uniform(1400.0, 1600.0)
            # c2 different enough to produce visible reflection
            c2 = c1 * rng.uniform(0.7, 1.5)
            c2 = np.clip(c2, 1000.0, 4000.0)
            rho1 = rng.uniform(950.0, 1100.0)
            rho2 = rng.uniform(950.0, 2000.0)
            # Interface position: 30%-70% of domain height
            interface_y = rng.uniform(0.3 * Ly, 0.7 * Ly)
            return {
                'type': 'interface',
                'f0': f0,
                'c1': c1, 'c2': c2,
                'rho1': rho1, 'rho2': rho2,
                'interface_y': interface_y,
            }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = self.indices[idx]
        params = self._sample_params(global_idx)
        
        nx = ny = self.grid_resolution
        Lx, Ly = self.domain_size
        N = nx * ny
        
        # Generate analytical solution
        if params['type'] == 'point_source':
            c_field, pressure_history = analytical_point_source(
                nx, ny, Lx, Ly,
                params['x0'], params['y0'],
                params['c0'], params['f0'],
                self.n_time_steps, self.dt,
            )
            rho_field = np.full((ny, nx), 1020.0)  # Uniform density
            
        elif params['type'] == 'plane_wave':
            c_field, pressure_history = analytical_plane_wave(
                nx, ny, Lx, Ly,
                params['c0'], params['f0'],
                params['theta'],
                self.n_time_steps, self.dt,
            )
            rho_field = np.full((ny, nx), 1020.0)
            
        else:  # interface
            c_field, rho_field, pressure_history = analytical_interface_reflection(
                nx, ny, Lx, Ly,
                params['c1'], params['c2'],
                params['rho1'], params['rho2'],
                params['interface_y'],
                params['f0'],
                self.n_time_steps, self.dt,
            )
        
        # Convert to tensors
        # pressure_history: (T, ny, nx) → (T, N) for graph format
        pressure_gt = torch.from_numpy(
            pressure_history.reshape(self.n_time_steps, -1)
        ).float()  # (T, N)
        
        # Sound speed and density as flat node arrays
        c_flat = torch.from_numpy(c_field.flatten()).float()   # (N,)
        rho_flat = torch.from_numpy(rho_field.flatten()).float()  # (N,)
        
        # Build graph
        graph = self._get_graph()
        
        # Node properties (matching kwave_dataset format)
        node_props = {
            'c': c_flat,
            'rho': rho_flat,
            'alpha_0': torch.zeros(N),   # No attenuation in analytical solutions
            'n_power': torch.ones(N),
            'dx_mean': graph['dx_mean'],
        }
        
        # Augment edge attributes with physics
        edge_attr_full = augment_edge_attr_with_physics(
            graph['edge_attr'], graph['edge_index'],
            node_props['c'], node_props['rho'],
            node_props['alpha_0'], node_props['n_power'],
        )
        
        # Transducer indices
        transducer_idx = create_transducer_indices(
            graph['positions'], n_elements=self.n_elements,
            domain_size=self.domain_size,
        )
        
        # HU values: derived from c (for node encoder compatibility)
        # Invert the simplified mapping: soft tissue c≈1540 → HU≈40
        # Linear approximation: HU ≈ (c - 1540) * 0.5 + 40
        hu = ((c_flat - 1540.0) * 0.5 + 40.0)
        
        # Generate a "B-mode" GT from pressure history for backward compatibility
        # (envelope of pressure at transducer elements over time)
        # This is a simplified beamform — but the real GT is pressure_gt
        bmode_gt = self._pressure_to_simple_bmode(pressure_gt, transducer_idx)
        
        return {
            'hu': hu,
            'edge_index': graph['edge_index'],
            'edge_attr': edge_attr_full,
            'bmode_gt': bmode_gt,
            'transducer_idx': transducer_idx,
            'positions': graph['positions'],
            'domain_size': graph['domain_size'],
            'node_props': node_props,
            # ── Analytical GT (new keys) ──
            'pressure_gt': pressure_gt,                          # (T, N) full history
            'pressure_final_gt': pressure_gt[-1],                # (N,) final step
            'scenario_type': torch.tensor(global_idx // self.samples_per_scenario),
        }
    
    def _pressure_to_simple_bmode(
        self, pressure_gt: torch.Tensor, transducer_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Convert pressure history to a simple B-mode-like image.
        
        Takes transducer element signals and creates a time-depth image.
        Shape: (n_elements, n_time_steps) → acts as a pseudo B-mode.
        Then resize to (256, 512) to match KWaveDataset format.
        """
        T, N = pressure_gt.shape
        M = transducer_idx.shape[0]
        
        # Extract signals at transducer elements: (T, M)
        rf_data = pressure_gt[:, transducer_idx]  # (T, M)
        
        # Hilbert envelope (using absolute value as simple approximation)
        envelope = torch.abs(rf_data)  # (T, M)
        
        # Transpose to (M, T) → treat as (lateral, axial)
        bmode = envelope.T  # (M, T)
        
        # Log compression
        bmode = 20.0 * torch.log10(bmode / (bmode.max() + 1e-10) + 1e-6)
        bmode = torch.clamp(bmode, min=-60.0)
        bmode = (bmode + 60.0) / 60.0  # Normalize to [0, 1]
        
        # Resize to (256, 512)
        import torch.nn.functional as F
        bmode = F.interpolate(
            bmode.unsqueeze(0).unsqueeze(0),
            size=(256, 512),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)
        
        return bmode


def create_analytical_dataloader(
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for analytical dataset."""
    dataset = AnalyticalDataset(split=split, **kwargs)
    
    def collate_fn(batch):
        if len(batch) == 1:
            return batch[0]
        return torch.utils.data.default_collate(batch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
