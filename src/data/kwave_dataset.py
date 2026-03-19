"""
kwave_dataset.py — PyTorch Dataset for k-Wave ground truth data.

Loads pre-computed k-Wave simulation results and builds graphs for training.

Data format (HDF5):
  Each sample contains:
    - ct_slice: (H, W) CT image in HU
    - bmode_gt: (H_out, W_out) k-Wave simulated B-mode image
    - probe_config: dict with probe parameters
    - medium_props: dict with acoustic properties
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple, List
import json

from .graph_builder import (
    build_2d_grid_graph,
    augment_edge_attr_with_physics,
    create_transducer_indices,
)

# Import acoustic property mapper from physics module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class KWaveDataset(Dataset):
    """Dataset for k-Wave ground truth ultrasound simulations.
    
    Expected directory structure:
    ```
    data_dir/
      sample_000/
        ct_slice.npy       # (H, W) CT in HU
        bmode_gt.npy       # (H_out, W_out) k-Wave B-mode
        metadata.json      # probe config, medium info
      sample_001/
        ...
    ```
    
    Or HDF5 format:
    ```
    data_dir/
      kwave_dataset.h5     # Contains all samples
    ```
    
    Args:
        data_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        grid_resolution: Graph grid resolution
        k_local: k-NN neighbors for local graph
        domain_size: Physical domain size [m]
        frequency: Ultrasound frequency [Hz]
        transform: Optional data augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        grid_resolution: int = 256,
        k_local: int = 8,
        domain_size: Tuple[float, float] = (0.06, 0.06),
        frequency: float = 5e6,
        n_elements: int = 128,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.grid_resolution = grid_resolution
        self.k_local = k_local
        self.domain_size = domain_size
        self.frequency = float(frequency)  # 确保是float类型
        self.n_elements = n_elements
        self.transform = transform
        
        # Discover samples
        self.samples = self._discover_samples()
        
        # Build graph structure (shared across samples for same grid)
        self._graph_cache = None
    
    def _discover_samples(self) -> List[str]:
        """Find all sample directories or entries."""
        samples = []
        
        # Check for directory-based format
        if os.path.isdir(self.data_dir):
            for name in sorted(os.listdir(self.data_dir)):
                sample_dir = os.path.join(self.data_dir, name)
                if os.path.isdir(sample_dir):
                    ct_path = os.path.join(sample_dir, 'ct_slice.npy')
                    gt_path = os.path.join(sample_dir, 'bmode_gt.npy')
                    if os.path.exists(ct_path) and os.path.exists(gt_path):
                        samples.append(sample_dir)
        
        # Split
        n = len(samples)
        if n == 0:
            print(f"⚠️ No samples found in {self.data_dir}")
            return []
        
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        if self.split == 'train':
            samples = samples[:train_end]
        elif self.split == 'val':
            samples = samples[train_end:val_end]
        else:
            samples = samples[val_end:]
        
        print(f"[KWaveDataset] {self.split}: {len(samples)} samples")
        return samples
    
    def _get_graph(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Get or build graph structure (cached for efficiency)."""
        if self._graph_cache is None:
            self._graph_cache = build_2d_grid_graph(
                nx=self.grid_resolution,
                ny=self.grid_resolution,
                domain_size=self.domain_size,
                k_local=self.k_local,
                device=device,
            )
        return self._graph_cache
    
    def __len__(self) -> int:
        return max(len(self.samples), 1)  # At least 1 for synthetic
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.
        
        Returns:
            sample: Dictionary with:
                - 'hu': (N,) CT HU values on graph nodes
                - 'edge_index': (2, E) graph edges
                - 'edge_attr': (E, 7) edge features with physics
                - 'bmode_gt': (H, W) ground truth B-mode
                - 'transducer_idx': (M,) transducer node indices
                - 'positions': (N, 2) node positions
                - 'domain_size': (2,) domain dimensions
                - 'node_props': dict with c, rho, alpha_0, n_power, dx_mean
        """
        if len(self.samples) > 0 and idx < len(self.samples):
            return self._load_real_sample(idx)
        else:
            return self._generate_synthetic_sample(idx)
    
    def _load_real_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a real k-Wave GT sample."""
        sample_dir = self.samples[idx]
        
        # Load CT and GT
        ct_slice = np.load(os.path.join(sample_dir, 'ct_slice.npy'))
        bmode_gt = np.load(os.path.join(sample_dir, 'bmode_gt.npy'))
        
        # Load metadata
        meta_path = os.path.join(sample_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Resize CT to grid resolution
        from scipy.ndimage import zoom
        scale_x = self.grid_resolution / ct_slice.shape[0]
        scale_y = self.grid_resolution / ct_slice.shape[1]
        ct_resized = zoom(ct_slice, (scale_x, scale_y), order=1)
        
        # Flatten to node values
        hu = torch.from_numpy(ct_resized.flatten()).float()
        bmode_gt = torch.from_numpy(bmode_gt).float()
        
        # Build graph
        graph = self._get_graph()
        
        # Map HU to acoustic properties
        node_props = self._hu_to_properties(hu)
        node_props['dx_mean'] = graph['dx_mean']
        
        # Augment edge attributes
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
        
        return {
            'hu': hu,
            'edge_index': graph['edge_index'],
            'edge_attr': edge_attr_full,
            'bmode_gt': bmode_gt,
            'transducer_idx': transducer_idx,
            'positions': graph['positions'],
            'domain_size': graph['domain_size'],
            'node_props': node_props,
        }
    
    def _generate_synthetic_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a synthetic sample (for testing without k-Wave data)."""
        nx = ny = self.grid_resolution
        N = nx * ny
        
        # Build graph
        graph = self._get_graph()
        
        # Synthetic phantom: random circular inclusions
        positions = graph['positions']
        rng = torch.Generator().manual_seed(idx)
        
        # Background: soft tissue (HU ~ 40)
        hu = torch.full((N,), 40.0)
        
        # Add 1-3 random inclusions
        n_inclusions = torch.randint(1, 4, (1,), generator=rng).item()
        for _ in range(n_inclusions):
            cx = torch.rand(1, generator=rng).item() * self.domain_size[0]
            cy = torch.rand(1, generator=rng).item() * self.domain_size[1]
            radius = (torch.rand(1, generator=rng).item() * 0.005 + 0.003)  # 3-8mm
            hu_val = torch.randint(-100, 1000, (1,), generator=rng).item()
            
            dist = torch.norm(positions - torch.tensor([cx, cy]), dim=-1)
            mask = dist < radius
            hu[mask] = hu_val
        
        # Map to properties
        node_props = self._hu_to_properties(hu)
        node_props['dx_mean'] = graph['dx_mean']
        
        # Augment edges
        edge_attr_full = augment_edge_attr_with_physics(
            graph['edge_attr'], graph['edge_index'],
            node_props['c'], node_props['rho'],
            node_props['alpha_0'], node_props['n_power'],
        )
        
        # Transducer
        transducer_idx = create_transducer_indices(
            positions, n_elements=self.n_elements,
            domain_size=self.domain_size,
        )
        
        # Synthetic GT: simple geometric pattern
        bmode_gt = torch.zeros(256, 512)
        
        return {
            'hu': hu,
            'edge_index': graph['edge_index'],
            'edge_attr': edge_attr_full,
            'bmode_gt': bmode_gt,
            'transducer_idx': transducer_idx,
            'positions': positions,
            'domain_size': graph['domain_size'],
            'node_props': node_props,
        }
    
    def _hu_to_properties(self, hu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map HU to acoustic properties using piecewise linear model.
        
        Simple but differentiable mapping without external dependencies.
        """
        N = hu.shape[0]
        
        # Piecewise linear mapping (simplified from acoustic_properties.py)
        # HU ranges and corresponding tissue properties
        # [rho, c, alpha_0_db_per_cm, n_power]
        tissues = [
            (-2000, -900, 1.2, 343.0, 41.0, 2.0),     # air
            (-900, -100, 400.0, 650.0, 40.0, 1.0),     # lung
            (-100, -50, 920.0, 1450.0, 0.63, 1.0),     # fat
            (-50, 20, 1000.0, 1480.0, 0.002, 2.0),     # water
            (20, 60, 1020.0, 1540.0, 0.5, 1.1),        # soft tissue
            (60, 100, 1060.0, 1550.0, 0.94, 1.1),      # liver
            (100, 300, 1050.0, 1580.0, 1.09, 1.0),     # muscle
            (300, 3000, 1900.0, 3500.0, 20.0, 1.0),    # bone
        ]
        
        rho = torch.full((N,), 1020.0)
        c = torch.full((N,), 1540.0)
        alpha_0 = torch.full((N,), 0.5)
        n_power = torch.full((N,), 1.1)
        
        for hu_min, hu_max, r, s, a, n in tissues:
            mask = (hu >= hu_min) & (hu < hu_max)
            rho[mask] = r
            c[mask] = s
            alpha_0[mask] = a
            n_power[mask] = n
        
        # Convert alpha from dB/cm to Np/m and apply frequency dependence
        alpha_np = alpha_0 * 11.51  # dB/cm → Np/m
        freq = float(self.frequency) if not isinstance(self.frequency, (int, float)) else self.frequency
        freq_ratio = freq / 1e6
        alpha_freq = alpha_np * (freq_ratio ** n_power)
        
        return {
            'c': c,
            'rho': rho,
            'alpha_0': alpha_freq,
            'n_power': n_power,
        }


def create_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for k-Wave dataset.
    
    Note: batch_size > 1 requires custom collation for graphs.
    For simplicity, we use batch_size=1 with graph-level batching.
    """
    dataset = KWaveDataset(data_dir, split=split, **kwargs)
    
    # Custom collate_fn to remove batch dimension [1, ...] -> [...]
    def collate_fn(batch):
        if len(batch) == 1:
            return batch[0]  # Return single sample directly
        # For batch_size > 1, use default collate
        return torch.utils.data.default_collate(batch)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def _collate_fn(batch: List[Dict]) -> Dict:
    """Custom collation for graph batches."""
    # For now, just return first item (batch_size=1 recommended)
    if len(batch) == 1:
        return batch[0]
    
    # Stack what we can, keep graphs separate
    return {
        'batch': batch,
        'batch_size': len(batch),
    }
