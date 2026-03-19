# Figure 6: Predicted Material Property Maps

**Type:** Half-page (7.16 × 3.5 inches)
**Format:** PNG (300 DPI) + PDF

## Content Description
Visualization of GNN-predicted material properties vs ground truth.

### Layout (3 rows × 4 columns)
```
           c_table(CT)    c_residual(GNN)   c_total        GT c
Row 1 (c): [viridis]      [coolwarm ±150]   [viridis]      [viridis]
           1400-1700 m/s   -150 to +150      1400-1700      1400-1700

           α_predicted     GT α (if available)
Row 2 (α): [magma]         [magma]
           0-50 Np/m        0-50 Np/m

           σ_predicted     GT σ (if available)
Row 3 (σ): [plasma]         [plasma]
           0-1              0-1
```

### Visual Requirements
1. Scientific colormaps: viridis (c), coolwarm (residual), magma (α), plasma (σ)
2. Colorbars with physical units on right side
3. All maps 256×256 resolution
4. c_residual uses diverging colormap centered at 0
5. Phantom structure (inclusion boundary) should be visible in c_total

### Caption
"Fig. 6. Predicted material property maps for a representative test case (single inclusion phantom). Top row: speed-of-sound decomposition — (a) c_table from CT lookup, (b) learned residual c_residual from GNN (diverging colormap, ±150 m/s), (c) final c = c_table + c_residual, (d) ground truth. The GNN correctly identifies the inclusion region and adjusts the speed-of-sound beyond the CT-derived prior. Middle row: attenuation coefficient α. Bottom row: reflectivity σ."

## Generation Code Requirements
```python
# Save to: fig06_material_maps/code/visualize_maps.py
# Input: trained model + test CT slice
# Output: fig06_material_maps.png (300dpi)
# Extract: c, alpha, sigma from model.encoder(ct)
# Also save c_table, c_residual separately
```
