#!/usr/bin/env python3
"""
generate_kwave_gt.py — Generate k-Wave ground truth dataset for DPC-GNN-Acoustic.

Uses k-wave-python (OMP binary) to run 2D acoustic simulations and saves results
in the directory-based format expected by KWaveDataset:
  sample_NNN/
    ct_slice.npy      # (Nx, Ny) pseudo-CT in HU
    bmode_gt.npy      # (n_elements, Nt) B-mode image (log-compressed envelope)
    metadata.json     # scenario config, probe parameters
    sensor_data.npy   # (n_elements, Nt) raw RF data

Three scenario types (A/B/C), configurable count per type.

Tested on: k-wave-python 0.4.0 + kspaceFirstOrder-OMP v1.3
           Intel Xeon Platinum 8470Q, ~0.9s per 128x128 simulation.

Usage:
    python generate_kwave_gt.py --n_per_type 5 --output_dir data/kwave_gt_test   # quick test
    python generate_kwave_gt.py --n_per_type 100 --output_dir data/kwave_gt      # full run
"""

import os
import sys
import json
import argparse
import time
import traceback
import numpy as np

# Limit OMP threads before importing k-wave (avoid OOM with 208 threads)
os.environ.setdefault("OMP_NUM_THREADS", "32")

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from scipy.signal import hilbert

# ─── Physical constants & grid ───
C_TISSUE = 1540.0     # m/s, soft tissue reference
RHO_TISSUE = 1000.0   # kg/m³
FREQ = 2e6            # 2 MHz ultrasound
DOMAIN_SIZE = 0.06    # 6 cm square domain
NX, NY = 128, 128     # Grid resolution (dataset loader resizes to model grid_resolution)
DX = DOMAIN_SIZE / NX  # ~4.69e-4 m
PML_SIZE = 20          # PML absorbing layers
N_ELEMENTS = 128       # Transducer elements
N_CYCLES = 3           # Tone burst cycles


def c_to_hu(c):
    """Convert speed of sound [m/s] to approximate HU values."""
    hu = np.zeros_like(c)
    hu[c < 500] = -1000                                                    # air
    m = (c >= 500) & (c < 1470); hu[m] = -100 + (c[m] - 1450) / 20 * 80   # fat
    m = (c >= 1470) & (c < 1510); hu[m] = (c[m] - 1480) / 60 * 40         # water
    m = (c >= 1510) & (c < 1600); hu[m] = 20 + (c[m] - 1540) / 60 * 60    # tissue
    m = (c >= 1600) & (c < 1700); hu[m] = 80 + (c[m] - 1600) / 100 * 220  # muscle
    m = c >= 1700; hu[m] = 300 + (c[m] - 1700) / 1800 * 700               # bone
    return hu


def rho_from_c(c):
    """Estimate density from speed of sound (empirical)."""
    return np.clip(1000.0 + 0.5 * (c - 1540.0), 800, 2200)


def make_tone_burst(freq, n_cycles, dt, Nt):
    """Windowed tone burst signal."""
    t = np.arange(Nt) * dt
    burst_len = n_cycles / freq
    signal = np.zeros(Nt)
    bm = t < burst_len
    if bm.any():
        tw = t[bm]
        signal[bm] = np.sin(2 * np.pi * freq * tw) * \
                      np.exp(-0.5 * ((tw - burst_len/2) / (burst_len/6)) ** 2)
    return signal


def sensor_data_to_bmode(sensor_data):
    """Convert (n_sensors, Nt) RF data → (n_sensors, Nt) B-mode [0,1]."""
    env = np.abs(hilbert(sensor_data, axis=1))
    env /= (env.max() + 1e-10)
    bmode = 20 * np.log10(env + 1e-6)
    bmode = np.clip(bmode, -60, 0)
    return (bmode + 60) / 60.0


def run_sim(c_map, rho_map, source, n_threads=32):
    """Run k-Wave 2D simulation.

    Returns: sensor_data (n_sensors, Nt), metadata dict with Nt/dt.
    """
    kgrid = kWaveGrid([NX, NY], [DX, DX])
    c_max = float(c_map.max())
    kgrid.makeTime(c_max)
    Nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    # Ensure round-trip time coverage
    min_time = 2.0 * DOMAIN_SIZE / C_TISSUE
    if Nt * dt < min_time:
        Nt = int(np.ceil(min_time / dt))
        kgrid.setTime(Nt, dt)
        Nt = int(kgrid.Nt)

    medium = kWaveMedium(
        sound_speed=c_map.astype(np.float64),
        density=rho_map.astype(np.float64),
    )

    # Sensor: linear array at top of domain (just inside PML)
    sensor = kSensor()
    smask = np.zeros((NX, NY), dtype=bool)
    n_elem = min(N_ELEMENTS, NY - 2 * PML_SIZE)
    start_y = (NY - n_elem) // 2
    smask[PML_SIZE + 1, start_y:start_y + n_elem] = True
    sensor.mask = smask
    sensor.record = ["p"]

    sim_opts = SimulationOptions(
        pml_size=[PML_SIZE, PML_SIZE],
        smooth_p0=(source.p0 is not None),
        smooth_c0=False, smooth_rho0=False,
        save_to_disk=True, data_cast="single",
    )
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=False, delete_data=True,
        show_sim_log=False, num_threads=n_threads,
    )

    result = kspaceFirstOrder2DC(kgrid, source, sensor, medium, sim_opts, exec_opts)

    # result["p"] shape = (Nt, n_sensors) → transpose to (n_sensors, Nt)
    p = np.array(result["p"]).T
    return p, {"Nt": Nt, "dt": dt, "n_sensors": int(p.shape[0])}


def make_array_source(Nt, dt):
    """Create driven source for linear array (pulse-echo)."""
    source = kSource()
    smask = np.zeros((NX, NY), dtype=bool)
    n_elem = min(N_ELEMENTS, NY - 2 * PML_SIZE)
    start_y = (NY - n_elem) // 2
    smask[PML_SIZE + 1, start_y:start_y + n_elem] = True
    source.p_mask = smask
    n_active = int(smask.sum())
    signal = make_tone_burst(FREQ, N_CYCLES, dt, Nt)
    source.p = np.tile(signal, (n_active, 1))
    source.p_mode = "dirichlet"
    return source


# ─── Scenario generators ───

def gen_A(rng):
    """A: Homogeneous + Gaussian point source."""
    c_val = C_TISSUE * (1 + rng.uniform(-0.05, 0.05))
    c_map = np.full((NX, NY), c_val)
    rho_map = np.full((NX, NY), RHO_TISSUE)

    # Random point source (Gaussian blob)
    margin = PML_SIZE + 10
    px = rng.integers(margin, NX - margin)
    py = rng.integers(margin, NY - margin)
    xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    p0 = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * 3**2))

    source = kSource()
    source.p0 = p0

    sensor_data, sim_meta = run_sim(c_map, rho_map, source)

    meta = {
        "scenario": "A_homogeneous_point",
        "c_base": float(c_val), "rho_base": float(RHO_TISSUE),
        "point_source": [int(px), int(py)],
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


def gen_B(rng):
    """B: Two-layer medium + linear array."""
    c1 = C_TISSUE
    c2 = rng.uniform(1600, 1800)
    interface = rng.integers(NX // 3, 2 * NX // 3)

    c_map = np.full((NX, NY), c1)
    c_map[interface:, :] = c2
    rho_map = np.full((NX, NY), RHO_TISSUE)
    rho_map[interface:, :] = rho_from_c(c2)

    # Need Nt/dt for source signal — create temporary kgrid
    kgrid_tmp = kWaveGrid([NX, NY], [DX, DX])
    kgrid_tmp.makeTime(float(c_map.max()))
    Nt = int(kgrid_tmp.Nt)
    dt = float(kgrid_tmp.dt)
    min_time = 2.0 * DOMAIN_SIZE / C_TISSUE
    if Nt * dt < min_time:
        Nt = int(np.ceil(min_time / dt))

    source = make_array_source(Nt, dt)
    sensor_data, sim_meta = run_sim(c_map, rho_map, source)

    meta = {
        "scenario": "B_two_layer",
        "c1": float(c1), "c2": float(c2),
        "rho1": float(RHO_TISSUE), "rho2": float(rho_from_c(c2)),
        "interface_row": int(interface),
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


def gen_C(rng):
    """C: Circular inclusion in homogeneous background."""
    c_bg = C_TISSUE
    c_inc = rng.uniform(1600, 2000)
    margin = PML_SIZE + 30
    cx = rng.integers(margin, NX - margin)
    cy = rng.integers(margin, NY - margin)
    radius = rng.integers(5, 18)  # pixels (~2–8 mm)

    xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    inc_mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= radius

    c_map = np.full((NX, NY), c_bg)
    c_map[inc_mask] = c_inc
    rho_map = np.full((NX, NY), RHO_TISSUE)
    rho_map[inc_mask] = rho_from_c(c_inc)

    kgrid_tmp = kWaveGrid([NX, NY], [DX, DX])
    kgrid_tmp.makeTime(float(c_map.max()))
    Nt = int(kgrid_tmp.Nt)
    dt = float(kgrid_tmp.dt)
    min_time = 2.0 * DOMAIN_SIZE / C_TISSUE
    if Nt * dt < min_time:
        Nt = int(np.ceil(min_time / dt))

    source = make_array_source(Nt, dt)
    sensor_data, sim_meta = run_sim(c_map, rho_map, source)

    meta = {
        "scenario": "C_inclusion",
        "c_bg": float(c_bg), "c_inclusion": float(c_inc),
        "rho_bg": float(RHO_TISSUE), "rho_inclusion": float(rho_from_c(c_inc)),
        "center": [int(cx), int(cy)], "radius_px": int(radius),
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


# ─── Save ───

def save_sample(out_dir, idx, c_map, rho_map, sensor_data, meta):
    d = os.path.join(out_dir, f"sample_{idx:04d}")
    os.makedirs(d, exist_ok=True)

    ct_slice = c_to_hu(c_map).astype(np.float32)
    bmode = sensor_data_to_bmode(sensor_data).astype(np.float32)

    np.save(os.path.join(d, "ct_slice.npy"), ct_slice)
    np.save(os.path.join(d, "bmode_gt.npy"), bmode)
    np.save(os.path.join(d, "sensor_data.npy"), sensor_data.astype(np.float32))

    # Add common metadata
    meta.update({
        "frequency": FREQ,
        "n_elements": N_ELEMENTS,
        "domain_size": [DOMAIN_SIZE, DOMAIN_SIZE],
        "grid_size": [NX, NY],
        "dx": DX,
        "pml_size": PML_SIZE,
    })
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sz = sum(os.path.getsize(os.path.join(d, fn))
             for fn in os.listdir(d)) / 1024
    print(f"  → {d}  ct={ct_slice.shape} bmode={bmode.shape} "
          f"rf={sensor_data.shape} ({sz:.0f}KB)")


# ─── Main ───

GENERATORS = {"A": gen_A, "B": gen_B, "C": gen_C}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_type", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="data/kwave_gt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "A", "B", "C"])
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scenarios = list(GENERATORS.keys()) if args.scenario == "all" else [args.scenario]
    idx = args.start_idx
    t_total = time.time()
    n_ok, n_fail = 0, 0

    for sc in scenarios:
        gen_fn = GENERATORS[sc]
        print(f"\n{'='*60}")
        print(f"Scenario {sc}: {args.n_per_type} samples")
        print(f"{'='*60}")

        for i in range(args.n_per_type):
            print(f"\n[{sc}] {i+1}/{args.n_per_type} (global idx={idx})")
            t0 = time.time()
            try:
                c_map, rho_map, sensor_data, meta = gen_fn(rng)
                save_sample(args.output_dir, idx, c_map, rho_map, sensor_data, meta)
                n_ok += 1
                print(f"  ✓ {time.time()-t0:.1f}s")
            except Exception as e:
                n_fail += 1
                print(f"  ✗ ERROR: {e}")
                traceback.print_exc()
            idx += 1

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done! {n_ok} OK, {n_fail} failed, {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
