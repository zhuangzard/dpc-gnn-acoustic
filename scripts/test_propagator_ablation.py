"""
Propagator ablation: systematically test which physics gap matters most.

With beamformer now fixed (SSIM=1.0 on k-Wave data), the remaining gap
is purely in the propagator. Oracle test with perfect c gives SSIM~0.38.

Key differences between V4 propagator and k-Wave:
1. Density: k-Wave has rho(x,y), V4 assumes uniform rho=1
2. dt: V4 uses 2e-8, k-Wave uses ~4e-8 (different time sampling)
3. n_steps: V4 uses 1754, k-Wave uses ~1946
4. Absorption model: different
5. Source waveform: may differ in details

This script tests:
A. V4 propagator (perfect c, no rho) → V4 beamformer (baseline, ~0.38)
B. Compare V4 sensor_data vs k-Wave sensor_data directly (before beamforming)
C. Check if dt/n_steps mismatch causes the gap
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from skimage.metrics import structural_similarity
from scipy.signal import hilbert

# Load k-Wave GT for sample_0100
sdir = "data/kwave_gt/sample_0100"
meta = json.load(open(os.path.join(sdir, "metadata.json")))
kwave_sensor = np.load(os.path.join(sdir, "sensor_data.npy"))
gt_bmode = np.load(os.path.join(sdir, "bmode_gt.npy"))

print("=" * 80)
print("PROPAGATOR ABLATION DIAGNOSTIC")
print("=" * 80)
print(f"Sample: sample_0100")
print(f"Scenario: {meta['scenario']}")
print(f"c1={meta['c1']:.0f}, c2={meta['c2']:.0f}, interface={meta['interface_row']}")
print(f"rho1={meta['rho1']:.0f}, rho2={meta['rho2']:.0f}")
print(f"k-Wave dt={meta['dt']:.6e}, Nt={meta['Nt']}")
print(f"k-Wave sensor shape: {kwave_sensor.shape}")

# === Test A: V4 propagator with perfect c, compare sensor data ===
print("\n" + "=" * 80)
print("TEST A: V4 Propagator (perfect c) — Sensor Data Comparison")
print("=" * 80)

from src.models.wave_propagator_v4 import AcousticLeapfrogV4
import yaml
with open("configs/v4_default.yaml") as f:
    cfg = yaml.safe_load(f)

prop_cfg = cfg.get("propagator", {})
v4_dt = prop_cfg.get("dt", 2.0e-8)
v4_nsteps = prop_cfg.get("n_time_steps", 1754)

prop = AcousticLeapfrogV4(
    nx=256, ny=256,
    dx=prop_cfg.get("dx", 2.34e-4),
    dt=v4_dt,
    n_steps=v4_nsteps,
    pml_width=20,
    n_elements=128,
    checkpoint_every=50,
    c_ref=1700.0,
).cuda()

# Perfect c_map
c_map = np.ones((256, 256), dtype=np.float32) * meta["c1"]
c_map[meta["interface_row"]:, :] = meta["c2"]
alpha_map = np.zeros((256, 256), dtype=np.float32)

c_t = torch.from_numpy(c_map).unsqueeze(0).unsqueeze(0).cuda()
alpha_t = torch.from_numpy(alpha_map).unsqueeze(0).unsqueeze(0).cuda()

# Source (2MHz tone burst)
t = torch.arange(v4_nsteps, device="cuda", dtype=torch.float32) * v4_dt
f0 = 2.0e6
n_cycles = 3
burst_len = n_cycles / f0
signal = torch.zeros(v4_nsteps, device="cuda")
n_burst = int(burst_len / v4_dt)
tw = t[:n_burst]
window = torch.exp(-0.5 * ((tw - burst_len/2) / (burst_len/6)) ** 2)
sine = torch.sin(2 * math.pi * f0 * tw)
signal[:n_burst] = sine * window
signal = signal / (signal.abs().max() + 1e-12) * 1e10
source = signal.unsqueeze(0)

with torch.no_grad():
    v4_sensor = prop(c_t, alpha_t, source)

v4_sensor_np = v4_sensor.squeeze().cpu().numpy()  # [128, 1754]

print(f"V4 sensor shape: {v4_sensor_np.shape}")
print(f"V4 sensor range: [{v4_sensor_np.min():.6e}, {v4_sensor_np.max():.6e}]")
print(f"kWave sensor range: [{kwave_sensor.min():.6e}, {kwave_sensor.max():.6e}]")

# Compare physical time coverage
v4_total_time = v4_nsteps * v4_dt
kw_total_time = meta["Nt"] * meta["dt"]
print(f"\nV4 total time: {v4_total_time*1e6:.2f} us ({v4_nsteps} steps x {v4_dt:.2e})")
print(f"kWave total time: {kw_total_time*1e6:.2f} us ({meta['Nt']} steps x {meta['dt']:.2e})")
print(f"Time ratio: {kw_total_time/v4_total_time:.2f}x")

# Compare element 64 (center) waveform
elem = 64
print(f"\nElement {elem} (center):")
print(f"  V4 max: {abs(v4_sensor_np[elem]).max():.6e}")
print(f"  kWave max: {abs(kwave_sensor[elem]).max():.6e}")

# DAS beamform both with GT beamformer
from scripts.test_physics_gap import gt_beamform

# Beamform V4 sensor data with V4's dt
meta_v4 = dict(meta)
meta_v4["dt"] = v4_dt
meta_v4["Nt"] = v4_nsteps
bmode_v4 = gt_beamform(v4_sensor_np, meta_v4)
ssim_v4 = structural_similarity(gt_bmode, bmode_v4,
    data_range=max(gt_bmode.max()-gt_bmode.min(), bmode_v4.max()-bmode_v4.min()))

print(f"\n=== BEAMFORMED COMPARISON ===")
print(f"V4 sensor → GT beamformer:    SSIM = {ssim_v4:.4f}")

# === Test B: What if we match dt? ===
print("\n" + "=" * 80)
print("TEST B: dt Mismatch Analysis")
print("=" * 80)

# Resample V4 sensor to k-Wave time axis
from scipy.interpolate import interp1d

v4_time = np.arange(v4_nsteps) * v4_dt
kw_time = np.arange(meta["Nt"]) * meta["dt"]

# Only interpolate up to V4's max time
valid_kw = kw_time <= v4_total_time
kw_time_valid = kw_time[valid_kw]

print(f"V4 time: 0 to {v4_total_time*1e6:.2f} us")
print(f"kWave time: 0 to {kw_total_time*1e6:.2f} us")
print(f"kWave samples within V4 time: {valid_kw.sum()} of {meta['Nt']}")

# === Test C: Energy comparison ===
print("\n" + "=" * 80)
print("TEST C: Energy & Waveform Analysis")
print("=" * 80)

v4_energy = (v4_sensor_np ** 2).sum()
kw_energy = (kwave_sensor ** 2).sum()
print(f"V4 total energy: {v4_energy:.6e}")
print(f"kWave total energy: {kw_energy:.6e}")
print(f"Energy ratio (V4/kWave): {v4_energy/kw_energy:.4f}")

# First arrival time comparison
threshold = 0.01 * abs(kwave_sensor[elem]).max()
kw_first = np.argmax(abs(kwave_sensor[elem]) > threshold)
v4_first = np.argmax(abs(v4_sensor_np[elem]) > threshold)

print(f"\nFirst arrival (element {elem}):")
print(f"  kWave: sample {kw_first} = {kw_first*meta['dt']*1e6:.2f} us")
print(f"  V4: sample {v4_first} = {v4_first*v4_dt*1e6:.2f} us")
print(f"  Physical time diff: {abs(kw_first*meta['dt'] - v4_first*v4_dt)*1e6:.2f} us")

# === Test D: Density effect estimate ===
print("\n" + "=" * 80)
print("TEST D: Density Effect Analysis")
print("=" * 80)
print(f"k-Wave uses density: rho1={meta['rho1']}, rho2={meta['rho2']}")
print(f"V4 assumes uniform density (not modeled)")
print(f"Acoustic impedance mismatch at interface:")
Z1 = meta["c1"] * meta["rho1"]
Z2 = meta["c2"] * meta["rho2"]
R = (Z2 - Z1) / (Z2 + Z1)
print(f"  Z1 = c1*rho1 = {Z1:.0f}")
print(f"  Z2 = c2*rho2 = {Z2:.0f}")
print(f"  Reflection coefficient R = {R:.4f}")
print(f"  Without density: R_no_rho = {(meta['c2']-meta['c1'])/(meta['c2']+meta['c1']):.4f}")
print(f"  Density contribution to R: {abs(R - (meta['c2']-meta['c1'])/(meta['c2']+meta['c1'])):.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Oracle SSIM (perfect c, V4 prop+BF): {ssim_v4:.4f}")
print(f"Beamformer gap: ZERO (fixed)")
print(f"Remaining gap sources:")
print(f"  1. dt mismatch: V4={v4_dt:.2e} vs kWave={meta['dt']:.2e}")
print(f"  2. Time coverage: V4={v4_total_time*1e6:.1f}us vs kWave={kw_total_time*1e6:.1f}us")
print(f"  3. Density: V4 has none, kWave has rho1={meta['rho1']}/rho2={meta['rho2']}")
print(f"  4. Source waveform: may differ in amplitude/phase")
print(f"  5. Numerical method: k-space FD vs k-Wave PSTD")
