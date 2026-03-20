"""Check c_map reconstruction accuracy for oracle test."""
import json, numpy as np

m = json.load(open("data/kwave_gt/sample_0200/metadata.json"))
print("c_bg:", m["c_bg"])
print("c_inclusion:", m["c_inclusion"])
print("rho_bg:", m["rho_bg"])
print("rho_inclusion:", m["rho_inclusion"])
print("center:", m["center"])
print("radius_px:", m["radius_px"])

# Reconstruct exact c_map from metadata
NX, NY = 256, 256
c_exact = np.full((NX, NY), m["c_bg"], dtype=np.float64)
rho_exact = np.full((NX, NY), m["rho_bg"], dtype=np.float64)
cx, cy = m["center"]
r = m["radius_px"]
Y, X = np.ogrid[:NX, :NY]
mask = (X - cx)**2 + (Y - cy)**2 <= r**2
c_exact[mask] = m["c_inclusion"]
rho_exact[mask] = m["rho_inclusion"]

print(f"\nExact c_map: [{c_exact.min():.1f}, {c_exact.max():.1f}]")
print(f"Exact rho: [{rho_exact.min():.1f}, {rho_exact.max():.1f}]")

# Compare with CT-based reconstruction
ct = np.load("data/kwave_gt/sample_0200/ct_slice.npy")
c_from_ct = ct * 0.75 + 1400.0
diff = np.abs(c_exact - c_from_ct)

c_inc = m["c_inclusion"]
c_ct_at_inc = c_from_ct[mask].max()
print(f"\nc_map vs ct_recon: max_diff={diff.max():.1f}, mean={diff.mean():.4f}")
print(f"Inclusion: true={c_inc:.1f}, from_ct={c_ct_at_inc:.1f}")
print(f"ERROR at inclusion: {c_inc - c_ct_at_inc:.1f} m/s")
if diff.max() > 1.0:
    print("⚠️ CRITICAL: CT→c reconstruction is LOSSY!")
