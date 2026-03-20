#!/usr/bin/env python3
"""Oracle test using saved c_map.npy and density.npy (no reconstruction needed)."""
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


def make_source(n_steps, dt):
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
    bmode_gt = np.load(os.path.join(sample_dir, 'bmode_gt.npy')).astype(np.float32)
    c_map = np.load(os.path.join(sample_dir, 'c_map.npy')).astype(np.float32)
    rho_map = np.load(os.path.join(sample_dir, 'density.npy')).astype(np.float32)
    
    kw_dt = float(meta['dt'])
    n_steps = sensor_gt.shape[1]
    scenario = meta.get('scenario', '?')
    
    prop = AcousticPropagatorV5(
        nx=NX, ny=NY, dx=DX, dt=kw_dt,
        pml_width=PML, n_elements=N_ELEMENTS, c_ref=C_REF,
    ).to(device).eval()
    
    bf = DifferentiableBeamformerV4(
        n_elements=N_ELEMENTS, c_ref=1540.0,
        image_size=(128, 128), dx=DX, dt=kw_dt,
        grid_size=NX, pml_size=PML,
        gt_dx=DX, gt_grid_size=NX, gt_pml=PML,
    ).to(device).eval()
    
    ssim_fn = SSIM().to(device)
    
    c_t = torch.from_numpy(c_map).unsqueeze(0).to(device)
    alpha_t = torch.zeros_like(c_t)
    rho_t = torch.from_numpy(rho_map).unsqueeze(0).to(device)
    source = torch.from_numpy(make_source(n_steps, kw_dt).astype(np.float32)).unsqueeze(0).to(device)
    gt_bmode_t = torch.from_numpy(bmode_gt).unsqueeze(0).unsqueeze(0).to(device)
    gt_sensor_t = torch.from_numpy(sensor_gt).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sensor_v5 = prop(c_t, alpha_t, source, rho=rho_t)
        bmode_v5 = bf(sensor_v5)
        ssim_val = ssim_fn(bmode_v5, gt_bmode_t).item()
    
    L = min(sensor_v5.shape[2], gt_sensor_t.shape[2])
    v5_center = sensor_v5[0, 64, :L].cpu().numpy()
    gt_center = gt_sensor_t[0, 64, :L].cpu().numpy()
    
    burst_steps = int(N_CYCLES / FREQ / kw_dt) + 5
    corr_burst = np.corrcoef(v5_center[:burst_steps], gt_center[:burst_steps])[0,1]
    
    echo_s, echo_e = max(burst_steps + 20, 100), min(800, L)
    v5_echo = v5_center[echo_s:echo_e]
    gt_echo = gt_center[echo_s:echo_e]
    v5_rms = np.sqrt(np.mean(v5_echo**2))
    gt_rms = np.sqrt(np.mean(gt_echo**2))
    corr_echo = np.corrcoef(v5_echo, gt_echo)[0,1] if gt_rms > 1e-12 and v5_rms > 1e-12 else float('nan')
    
    # Cross-correlation
    from scipy.signal import correlate
    v5_n = (v5_echo - v5_echo.mean()) / (v5_echo.std() + 1e-12)
    gt_n = (gt_echo - gt_echo.mean()) / (gt_echo.std() + 1e-12)
    xcorr = correlate(v5_n, gt_n, mode='full') / len(v5_n)
    lags = np.arange(-len(gt_n)+1, len(v5_n))
    best_lag = lags[np.argmax(xcorr)]
    best_corr = xcorr.max()
    
    # All elements echo corr
    elem_corrs = []
    for e in range(0, 128, 4):
        v = sensor_v5[0, e, echo_s:echo_e].cpu().numpy()
        g = gt_sensor_t[0, e, echo_s:echo_e].cpu().numpy()
        if np.std(v) > 1e-12 and np.std(g) > 1e-12:
            elem_corrs.append(np.corrcoef(v, g)[0,1])
    
    print(f"\n{os.path.basename(sample_dir)} ({scenario}):")
    print(f"  c: [{c_map.min():.0f}, {c_map.max():.0f}], rho: [{rho_map.min():.0f}, {rho_map.max():.0f}]")
    print(f"  ★ B-mode SSIM:   {ssim_val:.4f}")
    print(f"  Burst corr:      {corr_burst:.4f}")
    print(f"  Echo corr:       {corr_echo:.4f}")
    print(f"  Best corr:       {best_corr:.4f} at lag={best_lag}")
    print(f"  Echo ratio:      {v5_rms / max(gt_rms, 1e-12):.2f}×")
    print(f"  Elem mean corr:  {np.mean(elem_corrs):.4f} (n={len(elem_corrs)})")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    samples = [f'sample_050{i}' for i in range(5)]
    
    print("=" * 70)
    print("V5.3 Oracle Test — EXACT c_map + density from saved .npy")
    print("=" * 70)
    
    for sid in samples:
        path = os.path.join('data/kwave_gt', sid)
        if os.path.exists(os.path.join(path, 'c_map.npy')):
            run_oracle(path, device)
        else:
            print(f"\n{sid}: SKIP (no c_map.npy)")
    
    print("\n" + "=" * 70)
