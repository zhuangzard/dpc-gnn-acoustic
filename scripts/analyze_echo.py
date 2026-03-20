"""Analyze k-Wave GT sensor data to understand echo structure."""
import numpy as np
import sys

samples = [
    "data/kwave_gt/sample_0200",
    "data/kwave_gt/sample_0250",
    "data/kwave_gt/sample_0300",
    "data/kwave_gt/sample_0350",
    "data/kwave_gt/sample_0400",
]

for sample_path in samples:
    try:
        sensor = np.load(f"{sample_path}/sensor_data.npy")
        meta = np.load(f"{sample_path}/metadata.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print(f"SKIP {sample_path} (not found)")
        continue

    name = sample_path.split("/")[-1]
    scenario = meta.get("scenario", "unknown")
    dt_val = meta.get("dt", "N/A")
    nt_val = meta.get("Nt", sensor.shape[-1])
    
    print("=" * 70)
    print(f"{name} ({scenario})")
    print(f"  Sensor shape: {sensor.shape}, dt={dt_val}, Nt={nt_val}")
    
    center = sensor[64]  # center element
    n_steps = len(center)
    
    print(f"  Max abs: {np.abs(center).max():.6e}")
    
    # RMS in windows
    windows = [
        (0, 50, "burst"),
        (50, 100, "post-burst"),
        (100, 300, "early-echo"),
        (300, 600, "mid-echo"),
        (600, 1000, "late-echo"),
        (1000, min(1500, n_steps), "far-echo"),
        (1500, n_steps, "tail"),
    ]
    
    for start, end, label in windows:
        if start < n_steps and end <= n_steps and start < end:
            seg = center[start:end]
            rms = np.sqrt(np.mean(seg**2))
            peak = np.abs(seg).max()
            print(f"  [{start:5d}:{end:5d}] {label:12s}: RMS={rms:.4e}, peak={peak:.4e}")
    
    # Find echo events (local maxima above threshold)
    abs_sig = np.abs(center)
    burst_max = abs_sig[:50].max()
    echo_threshold = burst_max * 0.001  # 0.1% of burst
    
    # Smooth and find peaks
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    smoothed = uniform_filter1d(abs_sig, size=10)
    local_max = maximum_filter1d(smoothed, size=50) == smoothed
    echo_peaks = np.where((smoothed > echo_threshold) & local_max & (np.arange(n_steps) > 60))[0]
    
    if len(echo_peaks) > 0:
        print(f"  Echo peaks ({len(echo_peaks)} found):")
        for pk in echo_peaks[:10]:
            # Estimate reflector depth: depth = (step * dt * c_avg) / 2
            c_avg = (meta.get("c_min", 1500) + meta.get("c_max", 1500)) / 2
            if isinstance(dt_val, (int, float)):
                depth_mm = pk * dt_val * c_avg / 2 * 1000
                print(f"    step {pk:5d}: amp={smoothed[pk]:.4e}, depth~{depth_mm:.1f}mm")
            else:
                print(f"    step {pk:5d}: amp={smoothed[pk]:.4e}")
    else:
        print(f"  NO echo peaks found above threshold {echo_threshold:.4e}")
    
    # Check if GT has actual echoes
    post_burst_rms = np.sqrt(np.mean(center[100:]**2))
    burst_rms = np.sqrt(np.mean(center[:50]**2))
    ratio = post_burst_rms / (burst_rms + 1e-30)
    print(f"  Echo/Burst RMS ratio: {ratio:.6f}")
    if ratio < 0.001:
        print(f"  ⚠️ WARNING: GT appears to have NO echoes!")
    elif ratio < 0.01:
        print(f"  ⚠️ Very weak echoes")
    else:
        print(f"  ✅ Echoes present")

print("\n" + "=" * 70)
print("DONE")
