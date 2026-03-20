"""Deep echo analysis on regenerated k-Wave GT data."""
import numpy as np
import json
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

samples = [200, 260, 350, 360, 400, 410]

for idx in samples:
    path = f"data/kwave_gt/sample_{idx:04d}"
    try:
        sd = np.load(f"{path}/sensor_data.npy")
        m = json.load(open(f"{path}/metadata.json"))
    except FileNotFoundError:
        continue

    dt = m["dt"]
    scenario = m["scenario"]
    print("=" * 70)
    print(f"sample_{idx:04d} ({scenario})")
    print(f"  dt={dt:.4e}, grid={m['grid_size']}, Nt={m['Nt']}")
    print(f"  Sensor shape: {sd.shape}")
    
    center = sd[sd.shape[0]//2]
    n = len(center)
    
    # Windowed RMS
    print("\n  Windowed RMS:")
    for s in range(0, min(1800, n), 100):
        e = min(s + 100, n)
        rms = np.sqrt(np.mean(center[s:e]**2))
        depth_mm = s * dt * 1540 / 2 * 1000
        bars = "#" * max(0, int((np.log10(rms + 1e-30) + 5) * 8))
        print(f"    [{s:5d}:{e:5d}] depth~{depth_mm:5.1f}mm RMS={rms:.3e} {bars}")
    
    # Find echo peaks
    abs_center = np.abs(center)
    smoothed = uniform_filter1d(abs_center, size=20)
    burst_peak = smoothed[:60].max()
    
    peaks, _ = find_peaks(smoothed[60:], height=burst_peak * 0.005, distance=30)
    peaks += 60
    
    print(f"\n  Echo peaks (>{burst_peak*0.005:.3e}, {len(peaks)} found):")
    for p in peaks[:10]:
        depth = p * dt * 1540 / 2 * 1000
        relative_db = 20 * np.log10(smoothed[p] / (burst_peak + 1e-30))
        print(f"    step {p:5d}: amp={smoothed[p]:.3e} ({relative_db:+.1f} dB) depth~{depth:.1f}mm")
    
    # Dynamic range
    echo_max = smoothed[100:].max()
    dynamic_range_db = 20 * np.log10(burst_peak / (echo_max + 1e-30))
    print(f"\n  Dynamic range: {dynamic_range_db:.1f} dB (burst/echo_peak)")
    
    # Check if echoes are physically plausible
    # Load speed of sound to find interfaces
    try:
        ct = np.load(f"{path}/ct_slice.npy")
        # Find where c changes significantly (interfaces)
        c_row = ct[ct.shape[0]//2, :]  # center row along depth
        c_diff = np.abs(np.diff(c_row))
        interfaces = np.where(c_diff > c_diff.max() * 0.1)[0]
        pml = m.get("pml_size", 20)
        transducer_row = pml + 1
        
        print(f"\n  CT interfaces (center column):")
        for iface in interfaces[:5]:
            dist_from_sensor = abs(iface - transducer_row) * m["dx"]
            expected_step = int(2 * dist_from_sensor / (1540 * dt))
            print(f"    row {iface}: dist={dist_from_sensor*1000:.1f}mm, expected echo step~{expected_step}")
    except Exception as e:
        print(f"\n  Could not analyze CT: {e}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
