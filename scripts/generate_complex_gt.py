"""
Generate complex k-Wave GT scenarios that require GNN learning.

Simple two-layer data → c_table lookup is perfect → GNN has nothing to learn.
Complex data → c_table fails → GNN residual must learn spatial corrections.

New scenarios:
  D: Multi-layer (3-5 layers with varying c/ρ)
  E: Random scatterers (10-30 small inclusions in background)
  F: Smooth gradient (continuous c variation, not step function)
  G: Multi-layer + scatterers (combination)
  H: Realistic abdominal (fat→muscle→organ with sub-resolution scatterers)

Usage:
  python scripts/generate_complex_gt.py --scenario D --n_per_type 50 --start_idx 200
"""
import sys, os, argparse, json, time, traceback, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC

# Constants (match generate_kwave_gt.py)
NX, NY = 256, 256
DX = 2.34e-4  # m
DOMAIN_SIZE = NX * DX  # ~60mm
PML_SIZE = 20
FREQ = 2.0e6
N_CYCLES = 3
N_ELEMENTS = 128
C_TISSUE = 1540.0
RHO_TISSUE = 1000.0


def rho_from_c(c):
    """Empirical c→ρ mapping for soft tissue."""
    return 1000.0 + (c - 1540.0) * 0.5


def c_to_hu(c):
    """Speed of sound → pseudo-HU (CT value)."""
    return (c - 1400.0) / (1700.0 - 1400.0) * 400.0


def make_tone_burst(freq, n_cycles, dt, Nt):
    """Generate Gaussian-windowed tone burst."""
    t = np.arange(Nt) * dt
    burst_len = n_cycles / freq
    n_burst = int(burst_len / dt)
    signal = np.zeros(Nt)
    tw = t[:n_burst]
    window = np.exp(-0.5 * ((tw - burst_len/2) / (burst_len/6))**2)
    signal[:n_burst] = np.sin(2 * np.pi * freq * tw) * window
    signal /= np.abs(signal).max() + 1e-12
    return signal


def make_array_source(Nt, dt):
    """Create linear array source (matching generate_kwave_gt.py)."""
    source = kSource()
    smask = np.zeros((NX, NY), dtype=bool)
    n_elem = min(N_ELEMENTS, NY - 2 * PML_SIZE)
    start_y = (NY - n_elem) // 2
    smask[PML_SIZE + 1, start_y:start_y + n_elem] = True
    source.p_mask = smask
    n_active = int(smask.sum())
    signal = make_tone_burst(FREQ, N_CYCLES, dt, Nt)
    # CRITICAL: Only provide burst-length signal, NOT zero-padded to Nt!
    # Dirichlet mode: source.p=0 means SET pressure to 0 (not "no injection").
    # If padded to Nt, echoes arriving at source row get zeroed out.
    # k-Wave stops Dirichlet injection after source.p ends.
    burst_steps = int(np.ceil(N_CYCLES / FREQ / dt)) + 1
    signal_trimmed = signal[:burst_steps]
    source.p = np.tile(signal_trimmed, (n_active, 1))
    source.p_mode = "dirichlet"
    return source


def run_sim(c_map, rho_map, source):
    """Run k-Wave simulation."""
    kgrid = kWaveGrid([NX, NY], [DX, DX])
    kgrid.makeTime(float(c_map.max()))
    Nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    min_time = 2.0 * DOMAIN_SIZE / C_TISSUE
    if Nt * dt < min_time:
        Nt = int(np.ceil(min_time / dt))
    kgrid.setTime(Nt, dt)

    medium = kWaveMedium(
        sound_speed=c_map.astype(np.float64),
        density=rho_map.astype(np.float64),
    )

    sensor = kSensor()
    smask = np.zeros((NX, NY), dtype=bool)
    n_elem = min(N_ELEMENTS, NY - 2 * PML_SIZE)
    start_y = (NY - n_elem) // 2
    smask[PML_SIZE + 1, start_y:start_y + n_elem] = True
    sensor.mask = smask

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

    # If source doesn't have p_mask yet (for p0 sources), set Nt
    if not hasattr(source, 'p_mask') or source.p_mask is None:
        kgrid.setTime(Nt, dt)

    result = kspaceFirstOrder2DC(kgrid, source, sensor, medium, sim_options, exec_options)
    p = np.array(result["p"]).T
    return p, {"Nt": Nt, "dt": dt, "n_sensors": int(p.shape[0])}


def sensor_data_to_bmode(sensor_data):
    """DAS beamform sensor_data → B-mode image (matching regenerate_gt_bmode.py)."""
    from scipy.signal import hilbert
    
    n_elem, n_samples = sensor_data.shape
    image_h, image_w = 128, 128
    
    dx = DX
    G = NX
    pml = PML_SIZE
    
    start_col = (G - n_elem) // 2
    elem_lateral = np.linspace(start_col * dx, (start_col + n_elem - 1) * dx, n_elem)
    sensor_row = pml + 1
    elem_axial = sensor_row * dx
    
    px = np.linspace(start_col * dx, (start_col + n_elem - 1) * dx, image_w)
    py = np.linspace(sensor_row * dx, (G - pml) * dx, image_h)
    
    # Get dt from sensor_data timing
    kgrid_tmp = kWaveGrid([NX, NY], [DX, DX])
    kgrid_tmp.makeTime(1700.0)
    dt = float(kgrid_tmp.dt)
    
    c_ref = 1540.0
    bmode = np.zeros((image_h, image_w), dtype=np.float64)
    
    for iy in range(image_h):
        for ix in range(image_w):
            for ie in range(n_elem):
                dist_tx = py[iy] - elem_axial
                dist_rx = np.sqrt((py[iy] - elem_axial)**2 + (px[ix] - elem_lateral[ie])**2)
                total_dist = dist_tx + dist_rx
                delay_samples = total_dist / (c_ref * dt)
                idx = int(round(delay_samples))
                if 0 <= idx < n_samples:
                    bmode[iy, ix] += sensor_data[ie, idx]
    
    # Envelope detection + log compression (matching GT convention)
    analytic = hilbert(bmode, axis=0)
    envelope = np.abs(analytic)
    log_env = np.log(envelope + 1e-6)
    # Per-sample min-max normalize to [0, 1]
    log_min, log_max = log_env.min(), log_env.max()
    if log_max > log_min:
        bmode_norm = (log_env - log_min) / (log_max - log_min)
    else:
        bmode_norm = np.zeros_like(log_env)
    
    return bmode_norm.astype(np.float32)


# ─── Complex Scenario Generators ───

def gen_D(rng):
    """D: Multi-layer (3-5 random layers with different c/ρ)."""
    n_layers = rng.integers(3, 6)  # 3 to 5 layers
    
    # Generate random interface positions
    interfaces = sorted(rng.integers(NX // 6, 5 * NX // 6, size=n_layers - 1))
    
    # Generate random c for each layer
    c_values = [C_TISSUE]  # First layer always tissue
    for _ in range(n_layers - 1):
        c_values.append(rng.uniform(1400, 1700))
    
    c_map = np.full((NX, NY), c_values[0])
    rho_map = np.full((NX, NY), rho_from_c(c_values[0]))
    
    for i, iface in enumerate(interfaces):
        c_map[iface:, :] = c_values[i + 1]
        rho_map[iface:, :] = rho_from_c(c_values[i + 1])
    
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
        "scenario": "D_multi_layer",
        "n_layers": n_layers,
        "c_values": [float(c) for c in c_values],
        "rho_values": [float(rho_from_c(c)) for c in c_values],
        "interfaces": [int(i) for i in interfaces],
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


def gen_E(rng):
    """E: Random scatterers (10-30 small circular inclusions)."""
    c_bg = C_TISSUE
    n_scatterers = rng.integers(10, 31)
    
    c_map = np.full((NX, NY), c_bg)
    rho_map = np.full((NX, NY), RHO_TISSUE)
    
    xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    margin = PML_SIZE + 15
    
    scatterers = []
    for _ in range(n_scatterers):
        cx = rng.integers(margin, NX - margin)
        cy = rng.integers(margin, NY - margin)
        radius = rng.uniform(2, 8)  # 0.5-2mm
        c_inc = rng.uniform(1450, 1680)
        
        mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= radius
        c_map[mask] = c_inc
        rho_map[mask] = rho_from_c(c_inc)
        scatterers.append({
            "cx": int(cx), "cy": int(cy),
            "radius": float(radius), "c": float(c_inc)
        })
    
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
        "scenario": "E_scatterers",
        "c_bg": float(c_bg),
        "n_scatterers": n_scatterers,
        "scatterers": scatterers,
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


def gen_F(rng):
    """F: Smooth gradient (continuous c variation — NOT a step function).
    GNN must learn nonlinear CT→c mapping that c_table can't capture."""
    c_top = rng.uniform(1450, 1550)
    c_bottom = rng.uniform(1600, 1700)
    
    # Smooth gradient with random curvature
    gradient_type = rng.choice(["linear", "sigmoid", "quadratic"])
    row_coords = np.linspace(0, 1, NX)
    
    if gradient_type == "linear":
        c_profile = c_top + (c_bottom - c_top) * row_coords
    elif gradient_type == "sigmoid":
        steepness = rng.uniform(3, 10)
        midpoint = rng.uniform(0.3, 0.7)
        c_profile = c_top + (c_bottom - c_top) / (1 + np.exp(-steepness * (row_coords - midpoint)))
    else:  # quadratic
        c_profile = c_top + (c_bottom - c_top) * row_coords ** 2
    
    # Add lateral variation (sinusoidal)
    col_coords = np.linspace(0, 1, NY)
    n_waves = rng.integers(1, 4)
    amplitude = rng.uniform(10, 50)  # m/s variation
    
    c_map = np.zeros((NX, NY), dtype=np.float64)
    for i in range(NX):
        c_map[i, :] = c_profile[i] + amplitude * np.sin(2 * np.pi * n_waves * col_coords)
    
    c_map = np.clip(c_map, 1400, 1700)
    rho_map = np.vectorize(rho_from_c)(c_map)
    
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
        "scenario": "F_gradient",
        "c_top": float(c_top), "c_bottom": float(c_bottom),
        "gradient_type": gradient_type,
        "lateral_waves": int(n_waves),
        "lateral_amplitude": float(amplitude),
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


def gen_G(rng):
    """G: Multi-layer + scatterers (realistic combination)."""
    # 2-3 layers as background
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
    
    # Add scatterers on top
    xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
    margin = PML_SIZE + 15
    n_scat = rng.integers(5, 20)
    scatterers = []
    for _ in range(n_scat):
        cx = rng.integers(margin, NX - margin)
        cy = rng.integers(margin, NY - margin)
        radius = rng.uniform(2, 6)
        # Scatterer c differs from local background
        local_c = c_map[cx, cy]
        c_inc = local_c + rng.uniform(-100, 100)
        c_inc = np.clip(c_inc, 1400, 1700)
        mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= radius
        c_map[mask] = c_inc
        rho_map[mask] = rho_from_c(c_inc)
        scatterers.append({"cx": int(cx), "cy": int(cy), "r": float(radius), "c": float(c_inc)})
    
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
        "scenario": "G_layered_scatterers",
        "n_layers": n_layers, "c_values": [float(c) for c in c_values],
        "interfaces": [int(i) for i in interfaces],
        "n_scatterers": n_scat, "scatterers": scatterers,
    }
    meta.update(sim_meta)
    return c_map, rho_map, sensor_data, meta


def gen_H(rng):
    """H: Realistic abdominal cross-section.
    fat(1450) → muscle(1580) → organ(1560-1600) with sub-resolution speckle."""
    # Fat layer (skin + subcutaneous)
    fat_depth = rng.integers(30, 50)  # rows
    muscle_depth = rng.integers(20, 40)
    
    c_fat = rng.uniform(1430, 1470)
    c_muscle = rng.uniform(1560, 1600)
    c_organ = rng.uniform(1540, 1580)
    
    c_map = np.full((NX, NY), c_fat)
    rho_map = np.full((NX, NY), rho_from_c(c_fat))
    
    # Muscle layer
    c_map[fat_depth:fat_depth + muscle_depth, :] = c_muscle
    rho_map[fat_depth:fat_depth + muscle_depth, :] = rho_from_c(c_muscle)
    
    # Organ
    c_map[fat_depth + muscle_depth:, :] = c_organ
    rho_map[fat_depth + muscle_depth:, :] = rho_from_c(c_organ)
    
    # Add sub-resolution speckle noise to c map (±10-30 m/s)
    speckle_amplitude = rng.uniform(10, 30)
    speckle = rng.normal(0, speckle_amplitude, size=(NX, NY))
    # Smooth speckle slightly (local average over 3x3)
    from scipy.ndimage import uniform_filter
    speckle = uniform_filter(speckle, size=3)
    c_map += speckle
    c_map = np.clip(c_map, 1400, 1700)
    rho_map = np.vectorize(rho_from_c)(c_map)
    
    # Optional: add a "lesion" (circular region with different c)
    if rng.random() > 0.3:
        xx, yy = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        margin = fat_depth + muscle_depth + 10
        lcx = rng.integers(margin, NX - PML_SIZE - 10)
        lcy = rng.integers(PML_SIZE + 20, NY - PML_SIZE - 20)
        lr = rng.uniform(5, 15)
        lc = rng.uniform(1480, 1620)
        mask = np.sqrt((xx - lcx)**2 + (yy - lcy)**2) <= lr
        c_map[mask] = lc
        rho_map[mask] = rho_from_c(lc)
        lesion = {"cx": int(lcx), "cy": int(lcy), "r": float(lr), "c": float(lc)}
    else:
        lesion = None
    
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
        "scenario": "H_abdominal",
        "c_fat": float(c_fat), "c_muscle": float(c_muscle), "c_organ": float(c_organ),
        "fat_depth": int(fat_depth), "muscle_depth": int(muscle_depth),
        "speckle_amplitude": float(speckle_amplitude),
        "lesion": lesion,
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
    
    meta.update({
        "frequency": FREQ,
        "n_elements": N_ELEMENTS,
        "domain_size": [DOMAIN_SIZE, DOMAIN_SIZE],
        "grid_size": [NX, NY],
        "dx": DX,
        "pml_size": PML_SIZE,
    })
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, cls=NpEncoder)
    
    sz = sum(os.path.getsize(os.path.join(d, fn)) for fn in os.listdir(d)) / 1024
    print(f"  → {d}  ct={ct_slice.shape} bmode={bmode.shape} rf={sensor_data.shape} ({sz:.0f}KB)")


# ─── Main ───

GENERATORS = {
    "D": gen_D,
    "E": gen_E,
    "F": gen_F,
    "G": gen_G,
    "H": gen_H,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_type", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="data/kwave_gt")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--start_idx", type=int, default=200)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "D", "E", "F", "G", "H"])
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
