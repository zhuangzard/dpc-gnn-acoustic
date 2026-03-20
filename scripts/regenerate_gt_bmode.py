#!/usr/bin/env python3
"""
Regenerate bmode_gt.npy from existing sensor_data.npy using DAS beamforming.

This fixes the GT domain mismatch: old bmode_gt was sensor-domain envelope (128, Nt),
new bmode_gt is image-domain B-mode (128, 128) via Delay-and-Sum beamforming.

No need to re-run k-Wave simulations — just post-processes existing sensor_data.

Usage:
    python scripts/regenerate_gt_bmode.py --data_dir data/kwave_gt
    python scripts/regenerate_gt_bmode.py --data_dir data/kwave_gt --output_size 128
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import hilbert


def das_beamform_numpy(sensor_data, metadata, output_size=128, c_ref=1540.0):
    """
    DAS beamforming: sensor_data (n_elements, Nt) → B-mode image (output_size, output_size).
    
    Correctly handles:
    - Scenario A (p0 source): one-way delay (wave travels from target to sensor)
    - Scenario B/C (driven source, pulse-echo): round-trip delay (2 × distance)
    - Pixel grid starts from sensor depth (not from 0)
    - Coordinate convention: k-Wave dim0=row=axial(depth), dim1=col=lateral
    
    Args:
        sensor_data: (n_elements, n_samples) raw RF data
        metadata: dict with dx, pml_size, grid_size, scenario, etc.
        output_size: output image resolution (square)
        c_ref: reference speed of sound for beamforming
    
    Returns:
        bmode: (output_size, output_size) B-mode image in [0, 1]
    """
    n_elements, n_samples = sensor_data.shape
    
    # Physical parameters from metadata
    grid_nx = metadata.get('grid_size', [128, 128])[0]  # rows = axial/depth
    grid_ny = metadata.get('grid_size', [128, 128])[1]  # cols = lateral
    dx = metadata.get('dx', 4.69e-4)
    pml = metadata.get('pml_size', 20)
    dt_sim = metadata.get('dt', None)
    
    # Determine scenario for delay model
    scenario = metadata.get('scenario', 'A_homogeneous_point')
    is_pulse_echo = not scenario.startswith('A')  # B/C are pulse-echo
    
    # Compute dt from k-Wave: dt = kgrid.dt (stored in metadata)
    if dt_sim is None:
        c_max = 1700.0
        dt_sim = 0.4 * dx / (c_max * np.sqrt(2))
    
    # Element positions
    # k-Wave: smask[PML_SIZE+1, start_col:start_col+n_elem] = True
    # Row PML_SIZE+1 = axial position (depth), cols = lateral positions
    active_width = grid_ny - 2 * pml  # lateral active region
    n_elem_actual = min(n_elements, active_width)
    start_col = (grid_ny - n_elem_actual) // 2
    
    # elem_lateral = lateral positions of each element (k-Wave col direction)
    elem_lateral = np.array([(start_col + i) * dx for i in range(n_elem_actual)])
    if n_elements > n_elem_actual:
        # More requested elements than grid allows — use actual count
        elem_lateral = np.linspace(start_col * dx, (start_col + n_elem_actual - 1) * dx, n_elements)
    
    # elem_axial = axial (depth) position of sensor row
    sensor_row = pml + 1
    elem_axial = sensor_row * dx  # scalar, all elements at same depth
    
    # Pixel grid — starts from sensor depth, not from 0
    # Lateral: same range as sensor array
    px_lateral = np.linspace(start_col * dx, (start_col + n_elem_actual - 1) * dx, output_size)
    
    # Axial: from sensor depth to bottom of active region (excluding bottom PML)
    axial_start = sensor_row * dx
    axial_end = (grid_nx - pml) * dx
    py_axial = np.linspace(axial_start, axial_end, output_size)
    
    # Meshgrid: grid_axial[i,j] = py_axial[i], grid_lateral[i,j] = px_lateral[j]
    grid_axial, grid_lateral = np.meshgrid(py_axial, px_lateral, indexing='ij')
    
    # DAS: compute delays and sum
    rf_image = np.zeros((output_size, output_size), dtype=np.float64)
    
    for e in range(n_elements):
        # Distance from each pixel to this element
        d_lateral = grid_lateral - elem_lateral[e]
        d_axial = grid_axial - elem_axial
        dist = np.sqrt(d_lateral**2 + d_axial**2 + 1e-12)
        
        if is_pulse_echo:
            # Scenario B/C: round-trip (source=sensor, same position)
            # Total time = TX (source→target) + RX (target→sensor) = 2×dist
            delay_samples = 2.0 * dist / (c_ref * dt_sim)
        else:
            # Scenario A: p0 initial pressure, one-way (target→sensor)
            delay_samples = dist / (c_ref * dt_sim)
        
        # Clamp to valid range
        delay_samples = np.clip(delay_samples, 0, n_samples - 2)
        
        # Linear interpolation
        idx_lo = delay_samples.astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, n_samples - 1)
        frac = delay_samples - idx_lo
        
        val = sensor_data[e, idx_lo] * (1.0 - frac) + sensor_data[e, idx_hi] * frac
        rf_image += val
    
    # Hilbert envelope along axial/depth dimension (axis=0)
    analytic = hilbert(rf_image, axis=0)
    envelope = np.abs(analytic)
    
    # Log compression: log(env + eps) + min-max normalize to [0,1]
    log_env = np.log(envelope + 1e-6)
    log_min = log_env.min()
    log_max = log_env.max()
    if log_max - log_min > 1e-8:
        bmode = (log_env - log_min) / (log_max - log_min)
    else:
        bmode = np.zeros_like(log_env)
    
    return bmode.astype(np.float32)


def process_sample(sample_dir, output_size=128, c_ref=1540.0, backup=True):
    """Process one sample directory."""
    sample_dir = Path(sample_dir)
    sensor_path = sample_dir / 'sensor_data.npy'
    bmode_path = sample_dir / 'bmode_gt.npy'
    meta_path = sample_dir / 'metadata.json'
    
    if not sensor_path.exists():
        print(f"  SKIP (no sensor_data.npy): {sample_dir}")
        return False
    
    # Load sensor data
    sensor_data = np.load(str(sensor_path))
    
    # Load metadata
    metadata = {}
    if meta_path.exists():
        with open(str(meta_path)) as f:
            metadata = json.load(f)
    
    # Backup old bmode
    if backup and bmode_path.exists():
        backup_path = sample_dir / 'bmode_gt_old_sensor_domain.npy'
        if not backup_path.exists():
            os.rename(str(bmode_path), str(backup_path))
    
    # Generate new DAS B-mode
    bmode = das_beamform_numpy(sensor_data, metadata, output_size, c_ref)
    
    # Save
    np.save(str(bmode_path), bmode)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate bmode_gt.npy with DAS beamforming")
    parser.add_argument('--data_dir', type=str, default='data/kwave_gt',
                        help='Directory containing sample_XXXX subdirs')
    parser.add_argument('--output_size', type=int, default=128,
                        help='Output B-mode image size (square)')
    parser.add_argument('--c_ref', type=float, default=1540.0,
                        help='Reference speed of sound for beamforming')
    parser.add_argument('--no_backup', action='store_true',
                        help='Do not backup old bmode_gt.npy')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    samples = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(samples)} samples in {data_dir}")
    print(f"Output size: {args.output_size}×{args.output_size}")
    print(f"c_ref: {args.c_ref} m/s")
    print(f"Backup old GT: {not args.no_backup}")
    print()
    
    n_ok = 0
    n_fail = 0
    
    for i, sample_dir in enumerate(samples):
        ok = process_sample(sample_dir, args.output_size, args.c_ref, 
                           backup=not args.no_backup)
        if ok:
            n_ok += 1
            if (i + 1) % 10 == 0 or i == 0:
                # Quick check
                bmode = np.load(str(sample_dir / 'bmode_gt.npy'))
                print(f"  [{i+1}/{len(samples)}] {sample_dir.name}: "
                      f"shape={bmode.shape}, range=[{bmode.min():.4f}, {bmode.max():.4f}]")
        else:
            n_fail += 1
    
    print(f"\nDone: {n_ok} OK, {n_fail} failed")
    print(f"New bmode_gt.npy shape: ({args.output_size}, {args.output_size})")
    print("Old GT backed up as bmode_gt_old_sensor_domain.npy")


if __name__ == '__main__':
    main()
