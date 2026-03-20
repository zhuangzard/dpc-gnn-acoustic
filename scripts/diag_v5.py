#!/usr/bin/env python3
"""Detailed V5 vs k-Wave diagnostic on a single sample."""
import json, os, sys, math
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.wave_propagator_v5 import AcousticPropagatorV5

DX = 2.34e-4
DT_FIXED = 4.0e-8
FREQ = 2e6
N_CYCLES = 3

sid = "sample_0200"
gt_dir = "data/kwave_gt"

# Load metadata
meta = json.load(open(f"{gt_dir}/{sid}/metadata.json"))
print(f"=== {sid} ({meta.get('scenario','?')}) ===")
for k in ['dt', 'Nt', 'c_max', 'c_min', 'dx', 'NX', 'NY', 'PML_SIZE', 'frequency']:
    print(f"  {k} = {meta.get(k, 'NOT STORED')}")

# Load GT data
sd_gt = np.load(f"{gt_dir}/{sid}/sensor_data.npy")
ct = np.load(f"{gt_dir}/{sid}/ct_slice.npy").astype(np.float32)
print(f"\n  GT sensor shape = {sd_gt.shape}")
print(f"  CT range = [{ct.min():.2f}, {ct.max():.2f}]")

# Check density
rho_path = f"{gt_dir}/{sid}/density.npy"
if os.path.exists(rho_path):
    rho_gt = np.load(rho_path).astype(np.float32)
    print(f"  Density range = [{rho_gt.min():.1f}, {rho_gt.max():.1f}]")
else:
    print("  NO density.npy")
    rho_gt = None

# k-Wave dt
kw_dt = meta.get('dt', DT_FIXED)
if isinstance(kw_dt, str):
    kw_dt = float(kw_dt)
print(f"\n  k-Wave dt = {kw_dt:.4e}")
print(f"  V5 dt     = {DT_FIXED:.4e}")
print(f"  dt ratio  = {DT_FIXED / kw_dt:.6f}")

n_steps = sd_gt.shape[1]

# Make V5 source  
def make_source(n_steps, dt):
    t = torch.arange(n_steps, dtype=torch.float32) * dt
    burst_len = N_CYCLES / FREQ
    signal = torch.zeros(n_steps)
    n_burst = int(burst_len / dt) + 1
    tw = t[:n_burst]
    center = burst_len / 2.0
    sigma = burst_len / 6.0
    window = torch.exp(-0.5 * ((tw - center) / sigma) ** 2)
    sine = torch.sin(2.0 * math.pi * FREQ * tw)
    signal[:n_burst] = sine * window
    signal = signal / (signal.abs().max() + 1e-12)
    return signal.unsqueeze(0)

# Compare source waveforms
source_v5 = make_source(n_steps, DT_FIXED)
source_kw_dt = make_source(n_steps, kw_dt) if kw_dt != DT_FIXED else source_v5

# k-Wave burst from sensor data
kw_burst = sd_gt[64, :50]
v5_burst_sig = source_v5[0, :50].numpy()
print(f"\n  k-Wave burst first 5: {kw_burst[:5]}")
print(f"  V5 source first 5:    {v5_burst_sig[:5]}")
print(f"  k-Wave burst max:     {np.abs(kw_burst).max():.6f}")
print(f"  V5 source max:        {np.abs(v5_burst_sig).max():.6f}")

# Run V5 propagation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
c_perfect = torch.from_numpy((ct * 0.75 + 1400.0).clip(1400, 2000)).unsqueeze(0).to(device)
alpha_t = torch.ones_like(c_perfect) * 5.0
rho_t = torch.from_numpy(rho_gt).unsqueeze(0).to(device) if rho_gt is not None else None

prop = AcousticPropagatorV5(nx=256, ny=256, dx=DX, dt=DT_FIXED,
                            pml_width=20, n_elements=128, c_ref=2000.0).to(device).eval()

source = make_source(n_steps, DT_FIXED).to(device)
with torch.no_grad():
    sd_v5 = prop(c_perfect, alpha_t, source, rho=rho_t)

v5 = sd_v5[0, 64].cpu().numpy()
kw = sd_gt[64]
L = min(len(v5), len(kw))

print(f"\n=== Sensor comparison (elem 64) ===")
print(f"  V5 range:  [{v5.min():.6f}, {v5.max():.6f}]")
print(f"  kW range:  [{kw.min():.6f}, {kw.max():.6f}]")

# Windowed correlations
windows = [(0, 38, "burst"), (38, 100, "early"), (100, 500, "mid-echo"), 
           (500, 1000, "late-echo"), (1000, L, "far")]
for s, e, name in windows:
    e = min(e, L)
    if e <= s:
        continue
    v5w = v5[s:e]
    kww = kw[s:e]
    v5_rms = np.sqrt(np.mean(v5w**2))
    kw_rms = np.sqrt(np.mean(kww**2))
    if v5_rms > 1e-10 and kw_rms > 1e-10:
        corr = np.corrcoef(v5w, kww)[0, 1]
    else:
        corr = float('nan')
    ratio = v5_rms / max(kw_rms, 1e-12)
    print(f"  [{s:4d}:{e:4d}] {name:10s} | corr={corr:+.4f} | V5_rms={v5_rms:.6f} kW_rms={kw_rms:.6f} ratio={ratio:.2f}×")

# Also check: does V5 with alpha=0 differ much?
with torch.no_grad():
    sd_v5_noalpha = prop(c_perfect, torch.zeros_like(c_perfect), source, rho=rho_t)
v5_noalpha = sd_v5_noalpha[0, 64].cpu().numpy()
print(f"\n  Alpha=0 echo [100:500] rms={np.sqrt(np.mean(v5_noalpha[100:500]**2)):.6f} (vs α=5: {np.sqrt(np.mean(v5[100:500]**2)):.6f})")
