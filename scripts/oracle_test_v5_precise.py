#!/usr/bin/env python3
"""
Precise oracle test: reconstruct EXACT c and rho from GT metadata,
use k-Wave's exact dt, match source waveform exactly.
"""
import sys, os, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.wave_propagator_v5 import AcousticPropagatorV5
from src.models.beamform_decoder_v4 import DifferentiableBeamformerV4
from src.losses.combined_loss_v4 import SSIM

NX, NY = 256, 256
DX = 2.34e-4
PML = 20
N_ELEMENTS = 128
C_REF = 2000.0
FREQ = 2.0e6
N_CYCLES = 3


def rho_from_c(c):
    """Same as GT generation."""
    return 1000.0 + (c - 1540.0) * 0.5


def make_source_exact(n_steps, dt):
    """Exactly match GT's make_tone_burst + trimming."""
    t = np.arange(n_steps) * dt
    burst_len = N_CYCLES / FREQ
    n_burst = int(burst_len / dt)
    signal = np.zeros(n_steps)
    tw = t[:n_burst]
    window = np.exp(-0.5 * ((tw - burst_len/2) / (burst_len/6))**2)
    signal[:n_burst] = np.sin(2 * np.pi * FREQ * tw) * window
    signal /= (np.abs(signal).max() + 1e-12)
    return signal


def run_oracle(sample_dir, device='cuda'):
    meta = json.load(open(os.path.join(sample_dir, 'metadata.json')))
    sensor_gt = np.load(os.path.join(sample_dir, 'sensor_data.npy')).astype(np.float32)
    ct_raw = np.load(os.path.join(sample_dir, 'ct_slice.npy')).astype(np.float32)
    
    # Reconstruct c from CT (inverse of c_to_hu)
    # c_to_hu(c) = (c - 1400) / 300 * 400, so c = hu * 0.75 + 1400
    c_map = (ct_raw * 0.75 + 1400.0).clip(1400, 2000).astype(np.float32)
    
    # Reconstruct rho from c (SAME formula as GT generation)
    rho_map = rho_from_c(c_map).astype(np.float32)
    
    kw_dt = float(meta['dt'])
    n_steps = sensor_gt.shape[1]
    scenario = meta.get('scenario', '?')
    
    # Create propagator with k-Wave's exact dt
    prop = AcousticPropagatorV5(
        nx=NX, ny=NY, dx=DX, dt=kw_dt,
        pml_width=PML, n_elements=N_ELEMENTS, c_ref=C_REF,
    ).to(device).eval()
    
    # Tensors
    c_t = torch.from_numpy(c_map).unsqueeze(0).to(device)
    alpha_t = torch.zeros_like(c_t)  # LOSSLESS (matching k-Wave GT)
    rho_t = torch.from_numpy(rho_map).unsqueeze(0).to(device)
    
    # Source: exactly matching GT
    source_np = make_source_exact(n_steps, kw_dt)
    source_t = torch.from_numpy(source_np.astype(np.float32)).unsqueeze(0).to(device)
    
    gt_sensor = torch.from_numpy(sensor_gt).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sensor_v5 = prop(c_t, alpha_t, source_t, rho=rho_t)
    
    # Analysis
    L = min(sensor_v5.shape[2], gt_sensor.shape[2])
    v5_center = sensor_v5[0, 64, :L].cpu().numpy()
    gt_center = gt_sensor[0, 64, :L].cpu().numpy()
    
    # Burst region
    burst_steps = int(N_CYCLES / FREQ / kw_dt) + 5
    corr_burst = np.corrcoef(v5_center[:burst_steps], gt_center[:burst_steps])[0,1]
    
    # Echo region
    echo_start = max(burst_steps + 20, 100)
    echo_end = min(1200, L)
    v5_echo = v5_center[echo_start:echo_end]
    gt_echo = gt_center[echo_start:echo_end]
    v5_echo_rms = np.sqrt(np.mean(v5_echo**2))
    gt_echo_rms = np.sqrt(np.mean(gt_echo**2))
    
    if gt_echo_rms > 1e-12 and v5_echo_rms > 1e-12:
        corr_echo = np.corrcoef(v5_echo, gt_echo)[0,1]
    else:
        corr_echo = float('nan')
    
    # Amplitude comparison at different depths
    print(f"\n{os.path.basename(sample_dir)} ({scenario}):")
    print(f"  dt={kw_dt:.4e}, steps={n_steps}")
    print(f"  c range: [{c_map.min():.0f}, {c_map.max():.0f}] m/s")
    print(f"  rho range: [{rho_map.min():.0f}, {rho_map.max():.0f}] kg/m³")
    print(f"  Burst corr:  {corr_burst:.4f}")
    print(f"  Echo corr:   {corr_echo:.4f}")
    print(f"  Echo ratio:  {v5_echo_rms / max(gt_echo_rms, 1e-12):.2f}× (V5/GT)")
    print(f"  V5 echo rms: {v5_echo_rms:.6f}")
    print(f"  GT echo rms: {gt_echo_rms:.6f}")
    
    # Amplitude profile comparison
    print(f"  Amplitude profile (V5 vs GT):")
    for s in range(0, min(1500, L), 200):
        e = min(s + 200, L)
        v5_rms = np.sqrt(np.mean(v5_center[s:e]**2))
        gt_rms = np.sqrt(np.mean(gt_center[s:e]**2))
        ratio = v5_rms / max(gt_rms, 1e-12)
        print(f"    [{s:5d}:{e:5d}] V5={v5_rms:.4e} GT={gt_rms:.4e} ratio={ratio:.2f}×")
    
    # Save waveforms for visualization
    np.savez(
        os.path.join(sample_dir, 'oracle_v5_comparison.npz'),
        v5_center=v5_center,
        gt_center=gt_center,
        v5_sensor=sensor_v5[0].cpu().numpy(),
    )
    print(f"  Saved oracle_v5_comparison.npz")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_dir = 'data/kwave_gt'
    
    samples = ['sample_0200', 'sample_0250', 'sample_0300', 'sample_0350', 'sample_0400']
    
    print("=" * 70)
    print("V5.1 Precise Oracle Test (exact c + rho + lossless)")
    print("=" * 70)
    
    for sid in samples:
        path = os.path.join(gt_dir, sid)
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'sensor_data.npy')):
            run_oracle(path, device)
        else:
            print(f"\n{sid}: SKIP (no sensor_data)")
    
    print("\n" + "=" * 70)
