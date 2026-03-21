"""
Wavefield Dataset for Route B training.

Loads k-Wave full-field wavefield snapshots from FLAT HDF5 files.
Each file contains pressure frames p(x, y, t), velocity frames vx/vy,
and static material maps.

Data format: flat directory of HDF5 files (D_0000.h5, E_0001.h5, etc.).
Each h5 has: p_frames (T,H,W), c_map (H,W), rho_map (H,W), alpha_map (H,W).
Optionally: vx_frames (T,H,W), vy_frames (T,H,W).

Returns per __getitem__:
    dict with keys:
        'p_curr':    (H, W) current pressure frame
        'vx_curr':   (H, W) current x-velocity frame
        'vy_curr':   (H, W) current y-velocity frame
        'p_next':    (H, W) next pressure frame (target)
        'vx_next':   (H, W) next x-velocity frame (target)
        'vy_next':   (H, W) next y-velocity frame (target)
        'c_map':     (H, W) speed of sound
        'rho_map':   (H, W) density
        'alpha_map': (H, W) attenuation
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class WavefieldDataset(Dataset):
    """
    Loads k-Wave wavefield snapshots from flat HDF5 files for Route B training.

    Directory structure:
        data_dir/
            D_0000.h5
            E_0001.h5
            ...

    Each h5 file contains:
        - p_frames:  (T, H, W) pressure snapshots
        - c_map:     (H, W)   speed of sound
        - rho_map:   (H, W)   density
        - alpha_map: (H, W)   attenuation
        - vx_frames: (T, H, W) x-velocity (optional, zeros if absent)
        - vy_frames: (T, H, W) y-velocity (optional, zeros if absent)

    Args:
        data_dir: Path to directory containing .h5 files.
        rollout_len: Number of consecutive target frames per sample (default 1).
        frame_stride: Stride between sampled frame pairs (default 1).
    """

    def __init__(self, data_dir: str, rollout_len: int = 1,
                 frame_stride: int = 1, normalize: bool = False,
                 max_frames_per_file: int = 0,
                 file_indices: Optional[List[int]] = None):
        self.data_dir = Path(data_dir)
        self.rollout_len = rollout_len
        self.frame_stride = frame_stride
        self.normalize = normalize
        self.max_frames_per_file = max_frames_per_file  # 0=use all frames

        self.h5_files: List[Path] = []
        self.file_info: List[Dict] = []  # n_frames, grid_shape per file
        self._index: List[Tuple[int, int]] = []  # (file_idx, frame_idx)

        # P1-1: Glob *.h5 files directly from flat directory
        all_h5_files: List[Path] = []
        if self.data_dir.exists():
            all_h5_files = sorted(self.data_dir.glob('*.h5'))

        # Filter by file_indices if provided (for CT-level train/val split)
        if file_indices is not None:
            self.h5_files = [all_h5_files[i] for i in file_indices
                             if i < len(all_h5_files)]
        else:
            self.h5_files = all_h5_files

        # Probe each file for frame count
        if self.h5_files:
            import h5py
            for fi, h5_path in enumerate(self.h5_files):
                try:
                    with h5py.File(str(h5_path), 'r') as f:
                        p_shape = f['p_frames'].shape  # (T, H, W)
                        n_frames = p_shape[0]
                        grid_shape = p_shape[1:]
                        self.file_info.append({
                            'n_frames': n_frames,
                            'grid_shape': grid_shape,
                        })
                        # Build index: each valid (file, frame) pair
                        max_start = n_frames - self.rollout_len
                        file_frames = []
                        for t in range(0, max(1, max_start), self.frame_stride):
                            if t + self.rollout_len < n_frames:
                                file_frames.append((fi, t))
                        # Subsample if max_frames_per_file is set
                        if self.max_frames_per_file > 0 and len(file_frames) > self.max_frames_per_file:
                            import numpy as np
                            step = len(file_frames) / self.max_frames_per_file
                            file_frames = [file_frames[int(i * step)] for i in range(self.max_frames_per_file)]
                        self._index.extend(file_frames)
                except Exception as e:
                    print(f"WARNING: Skipping {h5_path}: {e}")

        if len(self._index) == 0 and len(self.h5_files) == 0:
            print(f"WARNING: No .h5 files found in {data_dir}.")

    @staticmethod
    def count_h5_files(data_dir: str) -> int:
        """Return the number of .h5 files in data_dir."""
        d = Path(data_dir)
        return len(sorted(d.glob('*.h5'))) if d.exists() else 0

    def __len__(self) -> int:
        return max(len(self._index), 1)  # At least 1 for dummy mode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if len(self._index) == 0:
            # Dummy data for testing without real data
            H, W = 256, 256
            return {
                'p_curr': torch.randn(H, W) * 1e4,
                'vx_curr': torch.randn(H, W) * 6.5e-3,
                'vy_curr': torch.randn(H, W) * 6.5e-3,
                'p_next': torch.randn(H, W) * 1e4,
                'vx_next': torch.randn(H, W) * 6.5e-3,
                'vy_next': torch.randn(H, W) * 6.5e-3,
                'c_map': torch.ones(H, W) * 1540.0,
                'rho_map': torch.ones(H, W) * 1000.0,
                'alpha_map': torch.ones(H, W) * 2.0,
            }

        file_idx, frame_idx = self._index[idx % len(self._index)]
        h5_path = self.h5_files[file_idx]

        import h5py
        with h5py.File(str(h5_path), 'r') as f:
            p_curr = f['p_frames'][frame_idx]                      # (H, W)
            p_next = f['p_frames'][frame_idx + 1]                  # (H, W)
            c_map = f['c_map'][:]                                  # (H, W)
            rho_map = f['rho_map'][:] if 'rho_map' in f else np.ones_like(c_map) * 1000.0
            alpha_map = f['alpha_map'][:] if 'alpha_map' in f else np.ones_like(c_map) * 2.0

            if 'vx_frames' in f:
                vx_curr = f['vx_frames'][frame_idx]
                vx_next = f['vx_frames'][frame_idx + 1]
            else:
                vx_curr = np.zeros_like(p_curr)
                vx_next = np.zeros_like(p_curr)

            if 'vy_frames' in f:
                vy_curr = f['vy_frames'][frame_idx]
                vy_next = f['vy_frames'][frame_idx + 1]
            else:
                vy_curr = np.zeros_like(p_curr)
                vy_next = np.zeros_like(p_curr)

        return {
            'p_curr': torch.from_numpy(p_curr.astype(np.float32)),
            'vx_curr': torch.from_numpy(vx_curr.astype(np.float32)),
            'vy_curr': torch.from_numpy(vy_curr.astype(np.float32)),
            'p_next': torch.from_numpy(p_next.astype(np.float32)),
            'vx_next': torch.from_numpy(vx_next.astype(np.float32)),
            'vy_next': torch.from_numpy(vy_next.astype(np.float32)),
            'c_map': torch.from_numpy(c_map.astype(np.float32)),
            'rho_map': torch.from_numpy(rho_map.astype(np.float32)),
            'alpha_map': torch.from_numpy(alpha_map.astype(np.float32)),
        }


if __name__ == '__main__':
    print("WavefieldDataset — Smoke Test")

    # Test with dummy data (no real files needed)
    ds = WavefieldDataset("nonexistent_dir")
    print(f"  Length: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    print("  Import OK")
