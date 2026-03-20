#!/usr/bin/env python3
"""
Oracle test for V5 propagator: perfect c → V5 propagation → DAS → compare with k-Wave GT.
Expected: SSIM close to 1.0 if V5 matches k-Wave physics.
"""
import sys, os, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.wave_propagator_v5 import AcousticPropagatorV5
from src.models.beamform_decoder_v4 import DifferentiableBeamformerV4
from src.losses.combined_loss_v4 import SSIM

# Physics params (must match GT generation)
NX, NY = 256, 256
DX = 2.34e-4
DT = 4.0e-8
PML = 20
N_ELEMENTS = 128
C_REF = 2000.0
FREQ = 2.0e6
N_CYCLES = 3

def make_source(n_steps, dt, device):
    """Match k-Wave GT source exactly."""
    t = torch.arange(n_steps, device=device, dtype=torch.float32) * dt
    burst_len = N_CYCLES / FREQ
    signal = torch.zeros(n_steps, device=device)
    n_burst = int(burst_len / dt) + 1
    if n_burst > n_steps:
        n_burst = n_steps
    tw = t[:n_burst]
    center = burst_len / 2.0
    sigma = burst_len / 6.0
    window = torch.exp(-0.5 * ((tw - center) / sigma) ** 2)
    sine = torch.sin(2.0 * math.pi * FREQ * tw)
    signal[:n_burst] = sine * window
    signal = signal / (signal.abs().max() + 1e-12)
    return signal.unsqueeze(0)  # [1, n_steps]

def run_oracle(sample_dir, device='cuda'):
    """Run oracle test on one sample."""
    # Load GT
    ct_raw = np.load(os.path.join(sample_dir, 'ct_slice.npy')).astype(np.float32)
    bmode_gt = np.load(os.path.join(sample_dir, 'bmode_gt.npy')).astype(np.float32)
    sensor_gt = np.load(os.path.join(sample_dir, 'sensor_data.npy')).astype(np.float32)
    meta = json.load(open(os.path.join(sample_dir, 'metadata.json')))
    
    # Perfect c from CT (same mapping as training)
    c_perfect = (ct_raw * 0.75 + 1400.0).clip(1400, 2000)
    
    # Load density if available
    rho_path = os.path.join(sample_dir, 'density.npy')
    if os.path.exists(rho_path):
        rho = np.load(rho_path).astype(np.float32)
    else:
        rho = np.ones_like(c_perfect) * 1000.0
    
    n_steps = sensor_gt.shape[1]
    
    # Use k-Wave's actual dt (CRITICAL: must match GT timing)
    kw_dt = meta.get('dt', DT)
    if isinstance(kw_dt, str):
        kw_dt = float(kw_dt)
    
    # Create propagator with k-Wave's dt
    prop = AcousticPropagatorV5(
        nx=NX, ny=NY, dx=DX, dt=kw_dt,
        pml_width=PML, n_elements=N_ELEMENTS, c_ref=C_REF,
    ).to(device).eval()
    
    # Create beamformer
    bf = DifferentiableBeamformerV4(
        n_elements=N_ELEMENTS, c_ref=1540.0,
        image_size=(128, 128), dx=DX, dt=DT,
        grid_size=NX, pml_size=PML,
        gt_dx=4.69e-4, gt_grid_size=128, gt_pml=20,
    ).to(device).eval()
    
    ssim_fn = SSIM().to(device)
    
    # Tensors
    c_t = torch.from_numpy(c_perfect).unsqueeze(0).to(device)  # [1, nx, ny]
    alpha_t = torch.ones_like(c_t) * 5.0  # moderate attenuation
    rho_t = torch.from_numpy(rho).unsqueeze(0).to(device)
    source = make_source(n_steps, kw_dt, device)  # Use k-Wave's dt!
    gt_bmode = torch.from_numpy(bmode_gt).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,128,128]
    gt_sensor = torch.from_numpy(sensor_gt).unsqueeze(0).to(device)  # [1, 128, n_steps]
    
    with torch.no_grad():
        # V5 propagation
        sensor_v5 = prop(c_t, alpha_t, source, rho=rho_t)
        
        # Beamform
        bmode_v5 = bf(sensor_v5)
        
        # SSIM
        ssim_val = ssim_fn(bmode_v5, gt_bmode).item()
    
    # Sensor correlation
    L = min(sensor_v5.shape[2], gt_sensor.shape[2])
    v5_center = sensor_v5[0, 64, :L].cpu().numpy()
    gt_center = gt_sensor[0, 64, :L].cpu().numpy()
    
    burst_end = 38
    corr_burst = np.corrcoef(v5_center[:burst_end], gt_center[:burst_end])[0,1]
    
    echo_start, echo_end = 100, min(1000, L)
    v5_echo_rms = np.sqrt(np.mean(v5_center[echo_start:echo_end]**2))
    gt_echo_rms = np.sqrt(np.mean(gt_center[echo_start:echo_end]**2))
    if gt_echo_rms > 1e-10 and v5_echo_rms > 1e-10:
        corr_echo = np.corrcoef(v5_center[echo_start:echo_end], gt_center[echo_start:echo_end])[0,1]
    else:
        corr_echo = float('nan')
    
    return {
        'ssim': ssim_val,
        'corr_burst': corr_burst,
        'corr_echo': corr_echo,
        'v5_echo_rms': v5_echo_rms,
        'gt_echo_rms': gt_echo_rms,
        'echo_ratio': v5_echo_rms / max(gt_echo_rms, 1e-12),
        'scenario': meta.get('scenario', '?'),
        'n_steps_v5': sensor_v5.shape[2],
        'n_steps_gt': gt_sensor.shape[2],
        'kw_dt': f"{kw_dt:.4e}",
    }

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_dir = 'data/kwave_gt'
    
    # Test representative samples from each scenario
    test_samples = [
        'sample_0200',  # D_multi_layer
        'sample_0250',  # E_scatterers
        'sample_0300',  # F_gradient
        'sample_0350',  # G_layered_scatterers
        'sample_0400',  # H_abdominal
    ]
    
    print("=" * 70)
    print("V5 Propagator Oracle Test")
    print("=" * 70)
    
    for sid in test_samples:
        sample_path = os.path.join(gt_dir, sid)
        if not os.path.exists(sample_path):
            print(f"{sid}: NOT FOUND")
            continue
        
        result = run_oracle(sample_path, device)
        print(f"\n{sid} ({result['scenario']}):")
        print(f"  Oracle SSIM:     {result['ssim']:.4f}")
        print(f"  Burst corr:      {result['corr_burst']:.4f}")
        print(f"  Echo corr:       {result['corr_echo']:.4f}")
        print(f"  Echo ratio:      {result['echo_ratio']:.2f}× (V5/GT, ideal=1.0)")
        print(f"  V5 echo rms:     {result['v5_echo_rms']:.6f}")
        print(f"  GT echo rms:     {result['gt_echo_rms']:.6f}")
        print(f"  Steps: V5={result['n_steps_v5']}, GT={result['n_steps_gt']}")
        print(f"  kW dt: {result.get('kw_dt', '?')}")
    
    print("\n" + "=" * 70)
