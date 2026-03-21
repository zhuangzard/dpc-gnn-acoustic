"""
Generate Route B full-field wavefield data from EXISTING CT slices.

Reads 200 real CT slices from data/kwave_gt/sample_XXXX/ct_slice.npy,
converts to acoustic parameters (c, rho, alpha), runs k-Wave simulation,
and saves full-field pressure snapshots as HDF5.

CT-to-acoustic mapping:
  c   = 1400 + (ct - ct_min)/(ct_max - ct_min) * 600  [m/s]  (range [1400, 2000])
  rho = 1000.0 + (c - 1540.0) * 0.5  [kg/m³]
  alpha ~ 0.75 * (1 + 0.3*(c-1500)/200)  [Np/(MHz^y * m)]

Usage:
    python scripts/generate_routeB_gt_ct.py
    python scripts/generate_routeB_gt_ct.py --n_samples 10 --workers 2
    python scripts/generate_routeB_gt_ct.py --ct_data_dir data/kwave_gt --output_dir data/routeB_gt_ct
"""

import sys
import os
import argparse
import json
import time
import traceback
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
from kwave.kgrid import kWaveGrid
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC

# ---------------------------------------------------------------------------
# Grid & physics constants — MUST match generate_complex_gt.py / generate_routeB_gt.py
# ---------------------------------------------------------------------------
NX, NY = 256, 256
DX = 2.34e-4              # m
DOMAIN_SIZE = NX * DX      # ~60 mm
PML_SIZE = 20
FREQ = 2.0e6               # Hz
N_CYCLES = 3
N_ELEMENTS = 128
C_TISSUE = 1540.0           # m/s
RHO_TISSUE = 1000.0         # kg/m³

# Frame extraction — same as generate_routeB_gt.py
FRAME_STRIDE = 5
FRAME_START = 30
FRAME_END = 1400

# Attenuation model — same as generate_routeB_gt.py
ALPHA_COEFF = 0.75
ALPHA_POWER = 1.5


# ---------------------------------------------------------------------------
# CT → acoustic property mapping
# ---------------------------------------------------------------------------
def ct_to_c(ct_slice: np.ndarray) -> np.ndarray:
    """Convert CT slice to speed of sound [m/s] using CALIBRATED inverse mapping.

    Exact inverse of c_to_hu(c) = (c - 1400) / 300 * 400 from generate_complex_gt.py.
    ct_slice values ~20-320 -> c ~1415-1640 m/s (soft tissue range).
    Bone augmentation is applied separately via add_synthetic_bone().
    """
    c = ct_slice * 0.75 + 1400.0
    return np.clip(c, 1400.0, 2000.0)


def add_synthetic_bone(c_map: np.ndarray, rho_map: np.ndarray,
                       alpha_map: np.ndarray, rng: np.random.Generator,
                       prob: float = 0.3) -> tuple:
    """Add synthetic bone regions to ~30% of samples for high-contrast diversity.

    Adds 1-3 thin bone-like strips (ribs/spine) with c=1800-2200, rho=1500-1900.
    This extends training coverage beyond the soft-tissue-only CT data.
    """
    if rng.random() > prob:
        return c_map, rho_map, alpha_map
    c_map = c_map.copy()
    rho_map = rho_map.copy()
    alpha_map = alpha_map.copy()
    n_bones = rng.integers(1, 4)
    for _ in range(n_bones):
        row = rng.integers(30, c_map.shape[0] - 30)
        width = rng.integers(3, 8)  # 3-8 pixels (~0.7-1.9 mm)
        c_bone = rng.uniform(1800.0, 2200.0)
        rho_bone = rng.uniform(1500.0, 1900.0)
        alpha_bone = rng.uniform(3.0, 8.0)  # bone attenuation much higher
        c_map[row:row+width, :] = c_bone
        rho_map[row:row+width, :] = rho_bone
        alpha_map[row:row+width, :] = alpha_bone
    return np.clip(c_map, 1400.0, 2500.0), rho_map, alpha_map


def rho_from_c(c):
    """Empirical c → ρ mapping for soft tissue (from generate_complex_gt.py)."""
    return 1000.0 + (c - 1540.0) * 0.5


def alpha_from_c(c_map: np.ndarray) -> np.ndarray:
    """Estimate attenuation from speed of sound (from generate_routeB_gt.py)."""
    alpha_map = ALPHA_COEFF * np.ones_like(c_map)
    alpha_map = alpha_map * (1.0 + 0.3 * (c_map - 1500.0) / 200.0)
    alpha_map = np.clip(alpha_map, 0.3, 3.0)
    return alpha_map


# ---------------------------------------------------------------------------
# Source setup — copied from generate_complex_gt.py
# ---------------------------------------------------------------------------
def make_tone_burst(freq, n_cycles, dt, Nt):
    """Generate Gaussian-windowed tone burst."""
    t = np.arange(Nt) * dt
    burst_len = n_cycles / freq
    n_burst = int(burst_len / dt)
    signal = np.zeros(Nt)
    tw = t[:n_burst]
    window = np.exp(-0.5 * ((tw - burst_len / 2) / (burst_len / 6)) ** 2)
    signal[:n_burst] = np.sin(2 * np.pi * freq * tw) * window
    signal /= np.abs(signal).max() + 1e-12
    return signal


def make_array_source(Nt, dt):
    """Create linear array source (matching generate_complex_gt.py)."""
    source = kSource()
    smask = np.zeros((NX, NY), dtype=bool)
    n_elem = min(N_ELEMENTS, NY - 2 * PML_SIZE)
    start_y = (NY - n_elem) // 2
    smask[PML_SIZE + 1, start_y:start_y + n_elem] = True
    source.p_mask = smask
    n_active = int(smask.sum())
    signal = make_tone_burst(FREQ, N_CYCLES, dt, Nt)
    # Trimmed signal — Dirichlet stops after source.p ends
    burst_steps = int(np.ceil(N_CYCLES / FREQ / dt)) + 1
    signal_trimmed = signal[:burst_steps]
    source.p = np.tile(signal_trimmed, (n_active, 1))
    source.p_mode = "dirichlet"
    return source


# ---------------------------------------------------------------------------
# k-Wave full-field simulation — copied from generate_routeB_gt.py
# ---------------------------------------------------------------------------
def run_fullfield_sim(c_map: np.ndarray, rho_map: np.ndarray,
                      alpha_map: np.ndarray = None,
                      frame_stride: int = FRAME_STRIDE,
                      frame_start: int = FRAME_START,
                      frame_end: int = FRAME_END,
                      data_path: str = None,
                      threads: int = 32) -> tuple:
    """Run k-Wave simulation recording full pressure field.

    Returns (p_frames, sim_meta) — identical logic to generate_routeB_gt.py.
    """
    kgrid = kWaveGrid([NX, NY], [DX, DX])
    kgrid.makeTime(float(c_map.max()))
    Nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    min_time = 2.0 * DOMAIN_SIZE / C_TISSUE
    if Nt * dt < min_time:
        Nt = int(np.ceil(min_time / dt))
    kgrid.setTime(Nt, dt)

    actual_frame_end = min(frame_end, Nt)

    medium_kwargs = dict(
        sound_speed=c_map.astype(np.float64),
        density=rho_map.astype(np.float64),
    )
    if alpha_map is not None:
        medium_kwargs['alpha_coeff'] = alpha_map.astype(np.float64)
        medium_kwargs['alpha_power'] = ALPHA_POWER
    medium = kWaveMedium(**medium_kwargs)

    source = make_array_source(Nt, dt)

    sensor = kSensor()
    sensor.mask = np.ones((NX, NY), dtype=bool)

    record_steps = list(range(frame_start, actual_frame_end, frame_stride))
    if len(record_steps) == 0:
        record_steps = list(range(0, Nt, frame_stride))

    sim_opts_kwargs = dict(
        pml_size=[PML_SIZE, PML_SIZE],
        smooth_p0=False,
        smooth_c0=False, smooth_rho0=False,
        save_to_disk=True,
        data_cast='single',
    )
    if data_path:
        sim_opts_kwargs['data_path'] = data_path
    sim_options = SimulationOptions(**sim_opts_kwargs)
    exec_options = SimulationExecutionOptions(
        is_gpu_simulation=False, delete_data=True,
        show_sim_log=False, num_threads=threads,
    )

    result = kspaceFirstOrder2DC(kgrid, source, sensor, medium,
                                  sim_options, exec_options)

    p_all = np.array(result["p"]).T
    if p_all.ndim == 2:
        # p_all is (n_sensors, n_time_steps) = (NX*NY, Nt)
        frame_indices = [t for t in record_steps if t < p_all.shape[1]]
        p_frames = p_all[:, frame_indices]       # (NX*NY, n_frames)
        p_frames = p_frames.T.reshape(len(frame_indices), NX, NY)
    else:
        raise ValueError(f"Unexpected p shape: {p_all.shape}")

    sim_meta = {
        'Nt': Nt,
        'dt': dt,
        'frame_stride': frame_stride,
        'frame_start': frame_start,
        'frame_end': actual_frame_end,
        'n_frames': p_frames.shape[0],
        'record_steps': frame_indices,
    }
    return p_frames, sim_meta


# ---------------------------------------------------------------------------
# Discover CT samples
# ---------------------------------------------------------------------------
def discover_ct_samples(ct_data_dir: str) -> list:
    """Find all sample_XXXX dirs containing ct_slice.npy, return sorted list of (idx, path)."""
    samples = []
    for entry in sorted(os.listdir(ct_data_dir)):
        m = re.match(r'sample_(\d+)', entry)
        if m is None:
            continue
        ct_path = os.path.join(ct_data_dir, entry, 'ct_slice.npy')
        if os.path.isfile(ct_path):
            samples.append((int(m.group(1)), ct_path))
    return samples


# ---------------------------------------------------------------------------
# Process one CT sample
# ---------------------------------------------------------------------------
def process_one(sample_idx: int, ct_path: str, output_dir: str,
                kwave_tmpdir: str, threads: int = 32) -> dict:
    """Load CT slice, convert to acoustic maps, run k-Wave, save HDF5."""
    t0 = time.time()

    # Ensure temp dir exists
    if kwave_tmpdir:
        os.makedirs(kwave_tmpdir, exist_ok=True)

    # Load CT slice
    ct_slice = np.load(ct_path).astype(np.float32)
    assert ct_slice.shape == (NX, NY), \
        f"CT slice shape {ct_slice.shape} != expected ({NX}, {NY})"

    # Convert to acoustic maps (calibrated inverse mapping)
    c_map = ct_to_c(ct_slice)
    rho_map = rho_from_c(c_map)
    alpha_map = alpha_from_c(c_map)

    # Add synthetic bone to ~30% of samples for high-contrast diversity
    rng = np.random.default_rng(seed=sample_idx)
    c_map, rho_map, alpha_map = add_synthetic_bone(c_map, rho_map, alpha_map, rng)

    # Run full-field simulation
    p_frames, sim_meta = run_fullfield_sim(
        c_map, rho_map, alpha_map=alpha_map,
        data_path=kwave_tmpdir, threads=threads,
    )

    # Save as HDF5
    out_path = os.path.join(output_dir, f"CT_{sample_idx:04d}.h5")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('p_frames', data=p_frames.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('c_map', data=c_map.astype(np.float32))
        f.create_dataset('rho_map', data=rho_map.astype(np.float32))
        f.create_dataset('alpha_map', data=alpha_map.astype(np.float32))
        # Store metadata as attributes
        for k, v in sim_meta.items():
            if isinstance(v, list):
                f.attrs[k] = np.array(v)
            else:
                f.attrs[k] = v
        f.attrs['source_ct_path'] = ct_path
        f.attrs['sample_idx'] = sample_idx
        f.attrs['frequency'] = FREQ
        f.attrs['n_elements'] = N_ELEMENTS
        f.attrs['domain_size'] = DOMAIN_SIZE
        f.attrs['grid_size'] = [NX, NY]
        f.attrs['dx'] = DX
        f.attrs['pml_size'] = PML_SIZE

    elapsed = time.time() - t0
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    return {
        'sample_idx': sample_idx,
        'out_path': out_path,
        'n_frames': p_frames.shape[0],
        'elapsed': elapsed,
        'size_mb': sz_mb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Route B full-field data from existing CT slices")
    parser.add_argument('--ct_data_dir', type=str, default='data/kwave_gt',
                        help='Path to kwave_gt dir containing sample_XXXX/')
    parser.add_argument('--output_dir', type=str, default='data/routeB_gt_ct',
                        help='Output directory for HDF5 files')
    parser.add_argument('--n_samples', type=int, default=0,
                        help='Number of CT slices to process (0 = all)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting sample index (skip earlier samples)')
    parser.add_argument('--kwave_tmpdir', type=str, default='/tmp/kw_rB_ct',
                        help='Temp directory for k-Wave')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel k-Wave processes')
    parser.add_argument('--threads', type=int, default=32,
                        help='OMP threads per k-Wave process')
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover samples
    samples = discover_ct_samples(args.ct_data_dir)
    if not samples:
        print(f"ERROR: No sample_XXXX/ct_slice.npy found in {args.ct_data_dir}")
        sys.exit(1)

    # Filter by start_idx
    samples = [(idx, p) for idx, p in samples if idx >= args.start_idx]

    # Limit count
    if args.n_samples > 0:
        samples = samples[:args.n_samples]

    print(f"Found {len(samples)} CT samples to process")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}, Threads/worker: {args.threads}")
    print(f"Grid: {NX}x{NY}, dx={DX*1e3:.3f}mm, PML={PML_SIZE}")
    print(f"Frame extraction: stride={FRAME_STRIDE}, start={FRAME_START}, end={FRAME_END}")
    print()

    t_total = time.time()
    n_ok, n_fail = 0, 0

    if args.workers <= 1:
        # Sequential
        for i, (sample_idx, ct_path) in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] sample_{sample_idx:04d} ...", flush=True)
            try:
                info = process_one(sample_idx, ct_path, args.output_dir,
                                   args.kwave_tmpdir, threads=args.threads)
                n_ok += 1
                print(f"  -> CT_{sample_idx:04d}.h5  "
                      f"{info['n_frames']} frames, "
                      f"{info['size_mb']:.1f}MB, "
                      f"{info['elapsed']:.1f}s")
            except Exception as e:
                n_fail += 1
                print(f"  ERROR: {e}")
                traceback.print_exc()
    else:
        # Parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for sample_idx, ct_path in samples:
                # Each worker gets its own tmp subdir
                tmp = os.path.join(args.kwave_tmpdir, f"w{sample_idx}")
                fut = pool.submit(process_one, sample_idx, ct_path,
                                  args.output_dir, tmp, args.threads)
                futures[fut] = sample_idx

            for fut in as_completed(futures):
                sample_idx = futures[fut]
                try:
                    info = fut.result()
                    n_ok += 1
                    print(f"  CT_{sample_idx:04d}.h5  "
                          f"{info['n_frames']} frames, "
                          f"{info['size_mb']:.1f}MB, "
                          f"{info['elapsed']:.1f}s")
                except Exception as e:
                    n_fail += 1
                    print(f"  CT_{sample_idx:04d} ERROR: {e}")
                    traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Done! {n_ok} OK, {n_fail} failed, {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
