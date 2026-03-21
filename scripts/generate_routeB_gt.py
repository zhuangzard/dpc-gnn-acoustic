"""
Generate training data for Route B: full wavefield snapshots.

Uses existing k-Wave pipeline but records FULL FIELD pressure p(x,y,t)
instead of just sensor_data.

Changes from generate_complex_gt.py:
  - sensor.mask = ones(Nx, Ny)  # Record everywhere, not just sensor row
  - Save every 5th timestep in window [30, 1400]
  - Output: p_frames.npy, c_map.npy, rho_map.npy, alpha_map.npy, metadata.json
  - Storage: ~45MB per sample (compressed)

Reuses existing phantom generators (scenarios D-H).

Usage:
    python scripts/generate_routeB_gt.py --scenario D --n_per_type 50
    python scripts/generate_routeB_gt.py --scenario all --n_per_type 100
"""

import sys
import os
import argparse
import json
import time
import traceback
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC

# Import phantom generators from existing script
from scripts.generate_complex_gt import (
    NX, NY, DX, DOMAIN_SIZE, PML_SIZE, FREQ, N_CYCLES, N_ELEMENTS,
    C_TISSUE, RHO_TISSUE, rho_from_c, make_tone_burst, make_array_source,
    gen_D, gen_E, gen_F, gen_G, gen_H,
)


# ---------------------------------------------------------------------------
# Constants for Route B data generation
# ---------------------------------------------------------------------------
FRAME_STRIDE = 5        # Save every 5th k-Wave step
FRAME_START = 30        # Skip initial transient
FRAME_END = 1400        # Stop before reflections from boundary
ALPHA_COEFF = 0.75      # Default attenuation coefficient [Np/(MHz^y * m)]
ALPHA_POWER = 1.5       # Frequency power law exponent


def run_fullfield_sim(c_map: np.ndarray, rho_map: np.ndarray,
                      alpha_map: np.ndarray = None,
                      frame_stride: int = FRAME_STRIDE,
                      frame_start: int = FRAME_START,
                      frame_end: int = FRAME_END) -> tuple:
    """
    Run k-Wave simulation recording full pressure field.

    Args:
        c_map: (NX, NY) speed of sound [m/s].
        rho_map: (NX, NY) density [kg/m^3].
        alpha_map: (NX, NY) attenuation [Np/m] (optional).
        frame_stride: Save every N-th time step.
        frame_start: First time step to record.
        frame_end: Last time step to record.

    Returns:
        p_frames: (T, NX, NY) pressure snapshots.
        sim_meta: dict with simulation metadata.
    """
    kgrid = kWaveGrid([NX, NY], [DX, DX])
    kgrid.makeTime(float(c_map.max()))
    Nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    min_time = 2.0 * DOMAIN_SIZE / C_TISSUE
    if Nt * dt < min_time:
        Nt = int(np.ceil(min_time / dt))
    kgrid.setTime(Nt, dt)

    # Adjust frame_end to not exceed Nt
    actual_frame_end = min(frame_end, Nt)

    medium = kWaveMedium(
        sound_speed=c_map.astype(np.float64),
        density=rho_map.astype(np.float64),
    )

    # Source: same as generate_complex_gt.py
    source = make_array_source(Nt, dt)

    # Sensor: ENTIRE FIELD (record pressure everywhere)
    sensor = kSensor()
    sensor.mask = np.ones((NX, NY), dtype=bool)  # All nodes
    # Record specific time steps only
    record_steps = list(range(frame_start, actual_frame_end, frame_stride))
    if len(record_steps) == 0:
        record_steps = list(range(0, Nt, frame_stride))

    sim_options = SimulationOptions(
        pml_size=[PML_SIZE, PML_SIZE],
        smooth_p0=False,
        smooth_c0=False, smooth_rho0=False,
        save_to_disk=True,
        data_cast='single',
    )
    exec_options = SimulationExecutionOptions(
        is_gpu_simulation=False, delete_data=True,
        show_sim_log=False, num_threads=32,
    )

    result = kspaceFirstOrder2DC(kgrid, source, sensor, medium,
                                  sim_options, exec_options)

    # result["p"] shape depends on sensor.mask = all -> (N_nodes, Nt)
    p_all = np.array(result["p"]).T  # (Nt, N_nodes) -> need to reshape
    # Actually k-Wave with full mask returns (n_active_sensors, Nt)
    # where n_active_sensors = NX*NY, ordered row-major
    if p_all.ndim == 2:
        # p_all is (n_sensors, n_time_steps) = (NX*NY, Nt)
        # Select recorded frames and reshape
        frame_indices = [t for t in record_steps if t < p_all.shape[1]]
        p_frames = p_all[:, frame_indices]  # (NX*NY, n_frames)
        p_frames = p_frames.T.reshape(len(frame_indices), NX, NY)  # (T, NX, NY)
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


def generate_phantom(scenario: str, rng: np.random.Generator) -> tuple:
    """
    Generate phantom maps for a given scenario.

    Returns:
        c_map: (NX, NY) speed of sound.
        rho_map: (NX, NY) density.
        alpha_map: (NX, NY) attenuation (estimated from c).
        scenario_meta: dict with scenario-specific metadata.
    """
    # Reuse existing generators to get c_map and rho_map
    # The generators also run sim, but we only need the maps
    # We'll extract maps then run our own full-field sim

    if scenario == 'D':
        # Multi-layer
        n_layers = rng.integers(3, 6)
        interfaces = sorted(rng.integers(NX // 6, 5 * NX // 6, size=n_layers - 1))
        c_values = [C_TISSUE]
        for _ in range(n_layers - 1):
            c_values.append(rng.uniform(1400, 1700))

        c_map = np.full((NX, NY), c_values[0])
        rho_map = np.full((NX, NY), rho_from_c(c_values[0]))
        for i, iface in enumerate(interfaces):
            c_map[iface:, :] = c_values[i + 1]
            rho_map[iface:, :] = rho_from_c(c_values[i + 1])

        meta = {'scenario': 'D_multi_layer', 'n_layers': n_layers,
                'c_values': [float(c) for c in c_values]}

    elif scenario == 'E':
        # Random scatterers
        c_map = np.full((NX, NY), C_TISSUE)
        rho_map = np.full((NX, NY), RHO_TISSUE)
        xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        margin = PML_SIZE + 15
        n_scat = rng.integers(10, 31)
        for _ in range(n_scat):
            cx, cy = rng.integers(margin, NX - margin), rng.integers(margin, NY - margin)
            r = rng.uniform(2, 8)
            c_inc = rng.uniform(1450, 1680)
            mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= r
            c_map[mask] = c_inc
            rho_map[mask] = rho_from_c(c_inc)
        meta = {'scenario': 'E_scatterers', 'n_scatterers': int(n_scat)}

    elif scenario == 'F':
        # Smooth gradient
        c_top = rng.uniform(1450, 1550)
        c_bottom = rng.uniform(1600, 1700)
        row_coords = np.linspace(0, 1, NX)
        c_profile = c_top + (c_bottom - c_top) * row_coords
        col_coords = np.linspace(0, 1, NY)
        n_waves = rng.integers(1, 4)
        amp = rng.uniform(10, 50)
        c_map = np.zeros((NX, NY))
        for i in range(NX):
            c_map[i, :] = c_profile[i] + amp * np.sin(2 * np.pi * n_waves * col_coords)
        c_map = np.clip(c_map, 1400, 1700)
        rho_map = np.vectorize(rho_from_c)(c_map)
        meta = {'scenario': 'F_gradient', 'c_top': float(c_top), 'c_bottom': float(c_bottom)}

    elif scenario == 'G':
        # Multi-layer + scatterers
        n_layers = rng.integers(2, 4)
        interfaces = sorted(rng.integers(NX // 4, 3 * NX // 4, size=n_layers - 1))
        c_values = [C_TISSUE]
        for _ in range(n_layers - 1):
            c_values.append(rng.uniform(1480, 1680))
        c_map = np.full((NX, NY), c_values[0])
        rho_map = np.full((NX, NY), rho_from_c(c_values[0]))
        for i, iface in enumerate(interfaces):
            c_map[iface:, :] = c_values[i + 1]
            rho_map[iface:, :] = rho_from_c(c_values[i + 1])
        xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        margin = PML_SIZE + 15
        n_scat = rng.integers(5, 20)
        for _ in range(n_scat):
            cx, cy = rng.integers(margin, NX - margin), rng.integers(margin, NY - margin)
            r = rng.uniform(2, 6)
            c_inc = np.clip(c_map[cx, cy] + rng.uniform(-100, 100), 1400, 1700)
            mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= r
            c_map[mask] = c_inc
            rho_map[mask] = rho_from_c(c_inc)
        meta = {'scenario': 'G_layered_scatterers'}

    elif scenario == 'H':
        # Realistic abdominal
        fat_depth = rng.integers(30, 50)
        muscle_depth = rng.integers(20, 40)
        c_fat, c_muscle, c_organ = rng.uniform(1430, 1470), rng.uniform(1560, 1600), rng.uniform(1540, 1580)
        c_map = np.full((NX, NY), c_fat)
        rho_map = np.full((NX, NY), rho_from_c(c_fat))
        c_map[fat_depth:fat_depth + muscle_depth, :] = c_muscle
        rho_map[fat_depth:fat_depth + muscle_depth, :] = rho_from_c(c_muscle)
        c_map[fat_depth + muscle_depth:, :] = c_organ
        rho_map[fat_depth + muscle_depth:, :] = rho_from_c(c_organ)
        speckle = rng.normal(0, rng.uniform(10, 30), size=(NX, NY))
        from scipy.ndimage import uniform_filter
        speckle = uniform_filter(speckle, size=3)
        c_map += speckle
        c_map = np.clip(c_map, 1400, 1700)
        rho_map = np.vectorize(rho_from_c)(c_map)
        meta = {'scenario': 'H_abdominal'}
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Estimate attenuation from speed of sound (simple model)
    # Higher c -> lower attenuation (bone is exception, but we're in soft tissue range)
    alpha_map = ALPHA_COEFF * np.ones_like(c_map)
    # Vary slightly with c: denser tissue has slightly higher attenuation
    alpha_map = alpha_map * (1.0 + 0.3 * (c_map - 1500.0) / 200.0)
    alpha_map = np.clip(alpha_map, 0.3, 3.0)

    return c_map, rho_map, alpha_map, meta


def save_routeB_sample(out_dir: str, idx: int,
                       p_frames: np.ndarray, c_map: np.ndarray,
                       rho_map: np.ndarray, alpha_map: np.ndarray,
                       meta: dict) -> None:
    """Save a Route B training sample."""
    d = os.path.join(out_dir, f"sample_{idx:04d}")
    os.makedirs(d, exist_ok=True)

    np.save(os.path.join(d, "p_frames.npy"), p_frames.astype(np.float32))
    np.save(os.path.join(d, "c_map.npy"), c_map.astype(np.float32))
    np.save(os.path.join(d, "rho_map.npy"), rho_map.astype(np.float32))
    np.save(os.path.join(d, "alpha_map.npy"), alpha_map.astype(np.float32))

    meta.update({
        'frequency': FREQ,
        'n_elements': N_ELEMENTS,
        'domain_size': [DOMAIN_SIZE, DOMAIN_SIZE],
        'grid_size': [NX, NY],
        'dx': DX,
        'pml_size': PML_SIZE,
    })

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, cls=NpEncoder)

    sz = sum(os.path.getsize(os.path.join(d, fn)) for fn in os.listdir(d)) / (1024 * 1024)
    print(f"  -> {d}  p_frames={p_frames.shape} ({sz:.1f}MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SCENARIOS = ['D', 'E', 'F', 'G', 'H']


def main():
    parser = argparse.ArgumentParser(description="Generate Route B full-field GT data")
    parser.add_argument('--n_per_type', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='data/wavefield_gt')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['all'] + SCENARIOS)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--frame_stride', type=int, default=FRAME_STRIDE)
    parser.add_argument('--frame_start', type=int, default=FRAME_START)
    parser.add_argument('--frame_end', type=int, default=FRAME_END)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    idx = args.start_idx
    t_total = time.time()
    n_ok, n_fail = 0, 0

    for sc in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario {sc}: {args.n_per_type} samples (full-field wavefield)")
        print(f"{'=' * 60}")

        for i in range(args.n_per_type):
            print(f"\n[{sc}] {i + 1}/{args.n_per_type} (global idx={idx})")
            t0 = time.time()
            try:
                # Generate phantom maps
                c_map, rho_map, alpha_map, meta = generate_phantom(sc, rng)

                # Run full-field simulation
                p_frames, sim_meta = run_fullfield_sim(
                    c_map, rho_map, alpha_map,
                    frame_stride=args.frame_stride,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                )
                meta.update(sim_meta)

                # Save
                save_routeB_sample(args.output_dir, idx,
                                   p_frames, c_map, rho_map, alpha_map, meta)
                n_ok += 1
                print(f"  OK ({time.time() - t0:.1f}s, {p_frames.shape[0]} frames)")

            except Exception as e:
                n_fail += 1
                print(f"  ERROR: {e}")
                traceback.print_exc()

            idx += 1

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Done! {n_ok} OK, {n_fail} failed, {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
