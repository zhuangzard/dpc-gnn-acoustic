#!/usr/bin/env python3
"""
Oracle test with EXACT c_map reconstructed from GT metadata.
No CT→c conversion (which is lossy for old samples).
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
    return 1000.0 + (c - 1540.0) * 0.5


def reconstruct_c_rho(meta):
    """Reconstruct exact c_map and rho_map from metadata."""
    scenario = meta.get("scenario", "unknown")
    c_map = np.full((NX, NY), 1540.0, dtype=np.float64)
    
    if scenario == "C_inclusion":
        c_bg = meta["c_bg"]
        c_inc = meta["c_inclusion"]
        rho_bg = meta.get("rho_bg", rho_from_c(c_bg))
        rho_inc = meta.get("rho_inclusion", rho_from_c(c_inc))
        cx, cy = meta["center"]
        r = meta["radius_px"]
        
        c_map[:] = c_bg
        Y, X = np.ogrid[:NX, :NY]
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        c_map[mask] = c_inc
        
        rho_map = np.full((NX, NY), rho_bg, dtype=np.float64)
        rho_map[mask] = rho_inc
        
    elif scenario == "D_multi_layer":
        c_values = meta["c_values"]
        interfaces = meta["interfaces"]
        c_map[:] = c_values[0]
        for i, iface in enumerate(interfaces):
            c_map[iface:, :] = c_values[i + 1]
        rho_map = rho_from_c(c_map)
        
    elif scenario == "E_scatterers":
        c_bg = meta.get("c_background", meta.get("c_bg", 1540.0))
        c_map[:] = c_bg
        for sc in meta.get("scatterers", []):
            cx, cy, r, c_sc = sc["x"], sc["y"], sc["radius"], sc["c"]
            Y, X = np.ogrid[:NX, :NY]
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            c_map[mask] = c_sc
        rho_map = rho_from_c(c_map)
        
    elif scenario == "F_gradient":
        c_top = meta.get("c_top", 1450)
        c_bottom = meta.get("c_bottom", 1650)
        for row in range(NX):
            frac = row / (NX - 1)
            c_map[row, :] = c_top + (c_bottom - c_top) * frac
        rho_map = rho_from_c(c_map)
        
    elif scenario == "G_layered_scatterers":
        # Layers + scatterers
        c_values = meta.get("c_values", [1540.0])
        interfaces = meta.get("interfaces", [])
        c_map[:] = c_values[0]
        for i, iface in enumerate(interfaces):
            if i + 1 < len(c_values):
                c_map[iface:, :] = c_values[i + 1]
        # Add scatterers if present
        for sc in meta.get("scatterers", []):
            cx, cy, r, c_sc = sc["x"], sc["y"], sc["radius"], sc["c"]
            Y, X = np.ogrid[:NX, :NY]
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            c_map[mask] = c_sc
        rho_map = rho_from_c(c_map)
        
    elif scenario == "H_abdominal":
        # Complex multi-organ
        c_values = meta.get("c_values", [1540.0])
        interfaces = meta.get("interfaces", [])
        c_map[:] = c_values[0]
        for i, iface in enumerate(interfaces):
            if i + 1 < len(c_values):
                c_map[iface:, :] = c_values[i + 1]
        for organ in meta.get("organs", []):
            cx, cy, r, c_o = organ["x"], organ["y"], organ["radius"], organ["c"]
            Y, X = np.ogrid[:NX, :NY]
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            c_map[mask] = c_o
        rho_map = rho_from_c(c_map)
        
    else:
        # Fallback: use CT conversion
        print(f"  ⚠️ Unknown scenario '{scenario}', falling back to CT→c")
        ct = np.load(os.path.join(os.path.dirname(meta.get("_path", "")), "ct_slice.npy"))
        c_map = (ct * 0.75 + 1400.0).clip(1400, 2000)
        rho_map = rho_from_c(c_map)
    
    return c_map.astype(np.float32), rho_map.astype(np.float32)


def make_source_exact(n_steps, dt):
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
    meta["_path"] = sample_dir
    sensor_gt = np.load(os.path.join(sample_dir, 'sensor_data.npy')).astype(np.float32)
    bmode_gt = np.load(os.path.join(sample_dir, 'bmode_gt.npy')).astype(np.float32)
    
    # Reconstruct EXACT c and rho from metadata
    c_exact, rho_exact = reconstruct_c_rho(meta)
    
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
    
    c_t = torch.from_numpy(c_exact).unsqueeze(0).to(device)
    alpha_t = torch.zeros_like(c_t)  # lossless
    rho_t = torch.from_numpy(rho_exact).unsqueeze(0).to(device)
    source = torch.from_numpy(make_source_exact(n_steps, kw_dt).astype(np.float32)).unsqueeze(0).to(device)
    gt_bmode = torch.from_numpy(bmode_gt).unsqueeze(0).unsqueeze(0).to(device)
    gt_sensor = torch.from_numpy(sensor_gt).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sensor_v5 = prop(c_t, alpha_t, source, rho=rho_t)
        bmode_v5 = bf(sensor_v5)
        ssim_val = ssim_fn(bmode_v5, gt_bmode).item()
    
    # Sensor correlation
    L = min(sensor_v5.shape[2], gt_sensor.shape[2])
    v5_center = sensor_v5[0, 64, :L].cpu().numpy()
    gt_center = gt_sensor[0, 64, :L].cpu().numpy()
    
    burst_steps = int(N_CYCLES / FREQ / kw_dt) + 5
    corr_burst = np.corrcoef(v5_center[:burst_steps], gt_center[:burst_steps])[0,1]
    
    echo_start = max(burst_steps + 20, 100)
    echo_end = min(800, L)
    v5_echo = v5_center[echo_start:echo_end]
    gt_echo = gt_center[echo_start:echo_end]
    v5_echo_rms = np.sqrt(np.mean(v5_echo**2))
    gt_echo_rms = np.sqrt(np.mean(gt_echo**2))
    
    if gt_echo_rms > 1e-12 and v5_echo_rms > 1e-12:
        corr_echo = np.corrcoef(v5_echo, gt_echo)[0,1]
    else:
        corr_echo = float('nan')
    
    # Cross-correlation for best lag
    from scipy.signal import correlate
    v5_n = (v5_echo - v5_echo.mean()) / (v5_echo.std() + 1e-12)
    gt_n = (gt_echo - gt_echo.mean()) / (gt_echo.std() + 1e-12)
    xcorr = correlate(v5_n, gt_n, mode='full') / len(v5_n)
    lags = np.arange(-len(gt_n)+1, len(v5_n))
    best_lag = lags[np.argmax(xcorr)]
    best_corr = xcorr.max()
    
    print(f"\n{os.path.basename(sample_dir)} ({scenario}):")
    print(f"  c range: [{c_exact.min():.0f}, {c_exact.max():.0f}] m/s")
    print(f"  rho range: [{rho_exact.min():.0f}, {rho_exact.max():.0f}] kg/m³")
    print(f"  B-mode SSIM:     {ssim_val:.4f}")
    print(f"  Burst corr:      {corr_burst:.4f}")
    print(f"  Echo corr:       {corr_echo:.4f}")
    print(f"  Best corr:       {best_corr:.4f} at lag={best_lag}")
    print(f"  Echo ratio:      {v5_echo_rms / max(gt_echo_rms, 1e-12):.2f}×")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_dir = 'data/kwave_gt'
    
    samples = ['sample_0200', 'sample_0250', 'sample_0300', 'sample_0350', 'sample_0400']
    
    print("=" * 70)
    print("V5.3 EXACT Oracle Test (c/rho from metadata, lossless)")
    print("=" * 70)
    
    for sid in samples:
        path = os.path.join(gt_dir, sid)
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'sensor_data.npy')):
            try:
                run_oracle(path, device)
            except Exception as e:
                print(f"\n{sid}: ERROR — {e}")
        else:
            print(f"\n{sid}: SKIP")
    
    print("\n" + "=" * 70)
