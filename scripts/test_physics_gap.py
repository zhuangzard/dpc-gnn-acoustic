"""
Physics gap diagnostic: systematically test what causes SSIM=0.35 with perfect c.

Tests:
1. V4 propagator → V4 beamformer → compare with GT  (baseline, already = 0.35)
2. k-Wave sensor_data → V4 beamformer → compare with GT  (isolate beamformer)
3. k-Wave sensor_data → GT beamformer → compare with GT  (should be ~1.0, sanity check)

This tells us: is the problem in the propagator, the beamformer, or both?
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import hilbert
from skimage.metrics import structural_similarity
import torch


def gt_beamform(sensor_data, metadata, output_size=128, c_ref=1540.0):
    """GT beamformer (copied from regenerate_gt_bmode.py)"""
    n_elements, n_samples = sensor_data.shape
    grid_nx = metadata.get("grid_size", [128, 128])[0]
    grid_ny = metadata.get("grid_size", [128, 128])[1]
    dx = metadata.get("dx", 4.69e-4)
    pml = metadata.get("pml_size", 20)
    dt_sim = metadata.get("dt", 4e-8)

    active_width = grid_ny - 2 * pml
    n_elem = min(n_elements, active_width)
    start_col = (grid_ny - n_elem) // 2
    elem_lateral = np.linspace(start_col * dx, (start_col + n_elem - 1) * dx, n_elements)
    sensor_row = pml + 1
    elem_axial = sensor_row * dx

    px = np.linspace(start_col * dx, (start_col + n_elem - 1) * dx, output_size)
    py = np.linspace(sensor_row * dx, (grid_nx - pml) * dx, output_size)
    grid_axial, grid_lateral = np.meshgrid(py, px, indexing="ij")

    rf_image = np.zeros((output_size, output_size), dtype=np.float64)
    for e in range(n_elements):
        d_lat = grid_lateral - elem_lateral[e]
        d_ax = grid_axial - elem_axial
        dist = np.sqrt(d_lat ** 2 + d_ax ** 2 + 1e-12)
        d_ax_abs = np.abs(d_ax)
        delay = (d_ax_abs + dist) / (c_ref * dt_sim)
        delay = np.clip(delay, 0, n_samples - 2)
        idx_lo = delay.astype(np.int64)
        idx_hi = np.minimum(idx_lo + 1, n_samples - 1)
        frac = delay - idx_lo
        val = sensor_data[e, idx_lo] * (1 - frac) + sensor_data[e, idx_hi] * frac
        rf_image += val

    analytic = hilbert(rf_image, axis=0)
    envelope = np.abs(analytic)
    log_env = np.log(envelope + 1e-6)
    log_min, log_max = log_env.min(), log_env.max()
    if log_max - log_min > 1e-8:
        bmode = (log_env - log_min) / (log_max - log_min)
    else:
        bmode = np.zeros_like(log_env)
    return bmode.astype(np.float32)


def main():
    from src.models.beamform_decoder_v4 import DifferentiableBeamformerV4

    data_dir = "data/kwave_gt"
    samples = sorted([d for d in os.listdir(data_dir) if d.startswith("sample_")])[:5]

    bf_v4 = DifferentiableBeamformerV4(
        n_elements=128, c_ref=1540.0, image_size=(128, 128),
        dx=2.34e-4, dt=2.0e-8, grid_size=256, pml_size=20
    ).cuda()

    print("=" * 80)
    print("PHYSICS GAP DIAGNOSTIC")
    print("=" * 80)

    for s in samples:
        sdir = os.path.join(data_dir, s)
        meta = json.load(open(os.path.join(sdir, "metadata.json")))
        gt_bmode = np.load(os.path.join(sdir, "bmode_gt.npy"))
        kwave_sensor = np.load(os.path.join(sdir, "sensor_data.npy"))

        # Test 3: k-Wave sensor → GT beamformer (sanity, should be ~1.0)
        bmode_3 = gt_beamform(kwave_sensor, meta)
        ssim_3 = structural_similarity(
            gt_bmode, bmode_3,
            data_range=max(gt_bmode.max() - gt_bmode.min(), bmode_3.max() - bmode_3.min())
        )

        # Test 2: k-Wave sensor → V4 beamformer (isolate beamformer difference)
        # IMPORTANT: V4 beamformer uses dt=2e-8 for delay computation,
        # but k-Wave sensor_data was sampled at dt_kw ~ 4e-8.
        # We need a beamformer with the CORRECT dt for k-Wave data.
        bf_kw = DifferentiableBeamformerV4(
            n_elements=128, c_ref=1540.0, image_size=(128, 128),
            dx=2.34e-4, dt=meta["dt"],  # Use k-Wave's actual dt!
            grid_size=256, pml_size=20
        ).cpu()
        kw_t = torch.from_numpy(kwave_sensor).float().unsqueeze(0)
        with torch.no_grad():
            bmode_2_t = bf_kw(kw_t)
        bmode_2 = bmode_2_t.squeeze().cpu().numpy()
        ssim_2 = structural_similarity(
            gt_bmode, bmode_2,
            data_range=max(gt_bmode.max() - gt_bmode.min(), bmode_2.max() - bmode_2.min())
        )

        c1 = meta["c1"]
        c2 = meta["c2"]
        iface = meta["interface_row"]
        dt_kw = meta["dt"]

        print(f"\n{s} (c1={c1:.0f}, c2={c2:.0f}, interface={iface}, dt_kw={dt_kw:.2e}):")
        print(f"  Test 3 (kWave sensor -> GT beamformer):  SSIM = {ssim_3:.4f}  [sanity]")
        print(f"  Test 2 (kWave sensor -> V4 beamformer):  SSIM = {ssim_2:.4f}  [BF gap]")
        print(f"  Test 1 (V4 prop+BF, perfect c):         SSIM ~ 0.35        [total gap]")
        print(f"  Beamformer alone: {ssim_3:.3f} -> {ssim_2:.3f} (drop = {ssim_3 - ssim_2:.3f})")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  Test 3 ~ 1.0: GT beamformer self-consistent")
    print("  Test 2 ~ Test 3: V4 beamformer OK, propagator is bottleneck")
    print("  Test 2 << Test 3: V4 beamformer is the problem (dt mismatch?)")
    print("=" * 80)


if __name__ == "__main__":
    main()
