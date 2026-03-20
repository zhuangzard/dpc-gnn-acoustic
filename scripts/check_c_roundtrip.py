"""Check if CT→c roundtrip loses precision."""
import numpy as np
import json

for idx in [200, 300, 400]:
    path = f"data/kwave_gt/sample_{idx:04d}"
    m = json.load(open(f"{path}/metadata.json"))
    ct = np.load(f"{path}/ct_slice.npy")
    
    # Our assumed c reconstruction
    c_recon = ct * 0.75 + 1400.0
    
    scenario = m.get("scenario", "?")
    print(f"sample_{idx:04d} ({scenario}):")
    print(f"  ct range: [{ct.min():.2f}, {ct.max():.2f}]")
    print(f"  c_recon range: [{c_recon.min():.1f}, {c_recon.max():.1f}]")
    
    # Check metadata for actual c values
    for key in ["c_values", "c_base", "c_inclusion", "c_background", 
                "interfaces", "n_layers", "inclusions"]:
        if key in m:
            print(f"  {key}: {m[key]}")
    
    # Reconstruct c_map from metadata if possible
    if "c_values" in m and "interfaces" in m:
        # D_multi_layer: reconstruct from c_values + interfaces
        c_true = np.full((256, 256), m["c_values"][0], dtype=np.float32)
        for i, iface in enumerate(m["interfaces"]):
            c_true[iface:, :] = m["c_values"][i + 1]
        
        # Compare
        diff = np.abs(c_recon - c_true)
        print(f"  c_recon vs c_true: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
        if diff.max() > 0.1:
            print(f"  ⚠️ C MAP MISMATCH!")
    
    if "c_background" in m and "inclusions" in m:
        # C_inclusion: reconstruct
        c_true = np.full((256, 256), m["c_background"], dtype=np.float32)
        for inc in m["inclusions"]:
            cx, cy, r = inc["center_x"], inc["center_y"], inc["radius"]
            c_inc = inc["c"]
            Y, X = np.ogrid[:256, :256]
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            c_true[mask] = c_inc
        
        diff = np.abs(c_recon - c_true)
        print(f"  c_recon vs c_true: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
        if diff.max() > 0.1:
            print(f"  ⚠️ C MAP MISMATCH!")
    
    # Check c_to_hu inversion accuracy
    # c_to_hu(c) = (c - 1400) / 300 * 400 = (c-1400)*4/3
    # hu_to_c(hu) = hu * 3/4 + 1400 = hu * 0.75 + 1400
    # roundtrip: c → (c-1400)*4/3 → float32 → *0.75+1400
    test_c = np.array([1400, 1450, 1500, 1540, 1600, 1700], dtype=np.float64)
    hu = ((test_c - 1400) / 300 * 400).astype(np.float32)
    c_back = hu.astype(np.float32) * 0.75 + 1400.0
    print(f"  Roundtrip test: max error = {np.abs(test_c - c_back).max():.6f} m/s")
    print()
