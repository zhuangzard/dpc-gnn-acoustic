"""
graph_builder.py — Multi-scale graph construction for acoustic simulation.

Builds a graph from CT/phantom data with:
  1. Local k-NN edges (for Laplacian approximation)
  2. Long-range edges (for fast wave propagation)
  3. Edge attributes including physics features
  4. Node property computation (dx_mean, etc.)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


def build_2d_grid_graph(
    nx: int = 256,
    ny: int = 256,
    dx: float = 0.0,  # Will be computed from domain_size
    domain_size: Tuple[float, float] = (0.06, 0.06),
    k_local: int = 8,
    k_long_range: int = 0,
    long_range_radius: float = 0.02,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    """Build 2D grid graph for acoustic simulation.
    
    Args:
        nx, ny: Grid resolution
        domain_size: (Lx, Ly) physical domain size [m]
        k_local: Local k-NN neighbors
        k_long_range: Additional long-range connections
        long_range_radius: Radius for long-range edges [m]
        device: Target device
    
    Returns:
        graph: Dictionary with positions, edge_index, edge_attr, etc.
    """
    Lx, Ly = domain_size
    dx_val = Lx / nx
    dy_val = Ly / ny
    
    # Create regular 2D grid positions
    x = torch.linspace(0, Lx, nx, device=device)
    y = torch.linspace(0, Ly, ny, device=device)
    gx, gy = torch.meshgrid(x, y, indexing='ij')
    positions = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (N, 2)
    N = positions.shape[0]
    
    # ── Build local k-NN edges ──
    try:
        from torch_cluster import knn_graph
        edge_index_local = knn_graph(positions, k=k_local, loop=False)
    except ImportError:
        # Fallback: structured grid connectivity (8-connected)
        edge_index_local = _build_grid_edges_2d(nx, ny, device)
    
    # ── Build long-range edges (optional) ──
    if k_long_range > 0:
        try:
            from torch_cluster import knn_graph
            # Subsample for long-range: every 4th node
            stride = max(1, int(np.sqrt(N / 100)))
            idx_subsample = torch.arange(0, N, stride, device=device)
            pos_sub = positions[idx_subsample]
            
            # k-NN on subsampled (gives long-range connections)
            ei_sub = knn_graph(pos_sub, k=k_long_range, loop=False)
            
            # Map back to full indices
            edge_index_long = idx_subsample[ei_sub]
            
            # Combine
            edge_index = torch.cat([edge_index_local, edge_index_long], dim=1)
        except ImportError:
            edge_index = edge_index_local
    else:
        edge_index = edge_index_local
    
    # Remove duplicate edges
    edge_index = _remove_duplicate_edges(edge_index)
    
    # ── Compute edge attributes ──
    src, dst = edge_index
    r_vec = positions[dst] - positions[src]  # (E, 2) relative position
    distance = torch.norm(r_vec, dim=-1, keepdim=True)  # (E, 1)
    
    # Basic edge_attr: [r_x, r_y, distance]
    edge_attr = torch.cat([r_vec, distance], dim=-1)  # (E, 3)
    
    # ── Compute per-node dx_mean ──
    dx_mean = _compute_dx_mean(positions, edge_index)  # (N,)
    
    return {
        'positions': positions,       # (N, 2)
        'edge_index': edge_index,     # (2, E)
        'edge_attr': edge_attr,       # (E, 3)
        'dx_mean': dx_mean,           # (N,)
        'N': N,
        'nx': nx,
        'ny': ny,
        'dx': dx_val,
        'dy': dy_val,
        'domain_size': torch.tensor(domain_size, device=device),
    }


def augment_edge_attr_with_physics(
    edge_attr: torch.Tensor,     # (E, 3) [r_vec, distance]
    edge_index: torch.Tensor,    # (2, E)
    c: torch.Tensor,             # (N,) sound speed
    rho: torch.Tensor,           # (N,) density
    alpha_0: torch.Tensor,       # (N,) reference attenuation
    n_power: torch.Tensor,       # (N,) power law exponent
) -> torch.Tensor:
    """Add physics features to edge attributes.
    
    Output edge_attr: [r_x, r_y, distance, Z_ratio, c_ratio, alpha_avg, n_avg]
    
    Args:
        edge_attr: (E, 3) basic edge features
        edge_index: (2, E) graph edges
        c, rho, alpha_0, n_power: (N,) per-node acoustic properties
    
    Returns:
        edge_attr_full: (E, 7) augmented edge features
    """
    src, dst = edge_index
    
    # Impedance ratio: Z_dst / Z_src
    Z = rho * c  # (N,)
    Z_ratio = Z[dst] / (Z[src] + 1e-8)  # (E,)
    
    # Sound speed ratio
    c_ratio = c[dst] / (c[src] + 1e-8)  # (E,)
    
    # Average attenuation along edge
    alpha_avg = (alpha_0[src] + alpha_0[dst]) / 2  # (E,)
    
    # Average power law exponent
    n_avg = (n_power[src] + n_power[dst]) / 2  # (E,)
    
    edge_attr_full = torch.cat([
        edge_attr,                      # (E, 3): [r_vec(2), distance(1)]
        Z_ratio.unsqueeze(-1),          # (E, 1)
        c_ratio.unsqueeze(-1),          # (E, 1)
        alpha_avg.unsqueeze(-1),        # (E, 1)
        n_avg.unsqueeze(-1),            # (E, 1)
    ], dim=-1)  # (E, 7)
    
    return edge_attr_full


def _build_grid_edges_2d(nx: int, ny: int, device: str = 'cpu') -> torch.Tensor:
    """Build 8-connected grid edges for 2D grid (fallback without torch_cluster)."""
    edges_src = []
    edges_dst = []
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            # 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        nidx = ni * ny + nj
                        edges_src.append(idx)
                        edges_dst.append(nidx)
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
    return edge_index


def _remove_duplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove duplicate edges from edge_index."""
    # Pack edges into single tensor for unique
    E = edge_index.shape[1]
    packed = edge_index[0] * (edge_index.max() + 1) + edge_index[1]
    _, unique_idx = torch.unique(packed, return_inverse=True)
    _, first_idx = torch.unique(unique_idx, return_inverse=True)
    mask = torch.zeros(E, dtype=torch.bool, device=edge_index.device)
    seen = set()
    keep = []
    for i in range(E):
        key = packed[i].item()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = torch.tensor(keep, dtype=torch.long, device=edge_index.device)
    return edge_index[:, keep]


def _compute_dx_mean(positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute mean neighbor distance per node."""
    N = positions.shape[0]
    src, dst = edge_index
    
    distances = torch.norm(positions[dst] - positions[src], dim=-1)  # (E,)
    
    # Scatter mean to get per-node average
    try:
        from torch_scatter import scatter_mean
        dx_mean = scatter_mean(distances, src, dim=0, dim_size=N)
    except ImportError:
        # Fallback
        dx_mean = torch.zeros(N, device=positions.device)
        counts = torch.zeros(N, device=positions.device)
        dx_mean.scatter_add_(0, src, distances)
        counts.scatter_add_(0, src, torch.ones_like(distances))
        dx_mean = dx_mean / (counts + 1e-8)
    
    return dx_mean


def create_transducer_indices(
    positions: torch.Tensor,      # (N, 2) node positions
    probe_type: str = 'linear',
    n_elements: int = 128,
    probe_position: str = 'top',   # 'top', 'bottom', 'left', 'right'
    domain_size: Tuple[float, float] = (0.06, 0.06),
) -> torch.Tensor:
    """Identify graph nodes corresponding to transducer elements.
    
    Args:
        positions: (N, 2) node positions [m]
        probe_type: 'linear' or 'convex'
        n_elements: Number of transducer elements
        probe_position: Side of domain where probe is placed
        domain_size: (Lx, Ly) domain size [m]
    
    Returns:
        transducer_idx: (M,) indices of transducer nodes (M ≤ n_elements)
    """
    Lx, Ly = domain_size
    
    if probe_position == 'top':
        # Top edge: y ≈ 0 (or close to it)
        boundary_mask = positions[:, 1] < (Ly * 0.01)
        # Center along x-axis
        x_center = Lx / 2
        probe_width = n_elements * 3e-4  # pitch ≈ 0.3mm
    elif probe_position == 'bottom':
        boundary_mask = positions[:, 1] > (Ly * 0.99)
        x_center = Lx / 2
        probe_width = n_elements * 3e-4
    elif probe_position == 'left':
        boundary_mask = positions[:, 0] < (Lx * 0.01)
        x_center = Ly / 2
        probe_width = n_elements * 3e-4
    else:  # right
        boundary_mask = positions[:, 0] > (Lx * 0.99)
        x_center = Ly / 2
        probe_width = n_elements * 3e-4
    
    # Get boundary nodes
    boundary_idx = torch.where(boundary_mask)[0]
    
    if len(boundary_idx) == 0:
        # Fallback: first n_elements nodes
        return torch.arange(min(n_elements, positions.shape[0]), device=positions.device)
    
    # Select n_elements evenly spaced along boundary
    if len(boundary_idx) >= n_elements:
        step = len(boundary_idx) // n_elements
        transducer_idx = boundary_idx[::step][:n_elements]
    else:
        transducer_idx = boundary_idx
    
    return transducer_idx
