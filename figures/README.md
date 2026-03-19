# DPC-GNN-Acoustic V4 — Paper Figures

**Target:** IEEE TMI / Nature Communications
**Standard:** 300 DPI minimum, vector (SVG) preferred, PNG for raster

## Directory Structure
```
figures/
├── fig01_architecture/     # Architecture overview (full page)
├── fig02_taxonomy/         # L0-L3 classification (half page)
├── fig03_cross_domain_mp/  # Antisymmetric MP transfer (half page)
├── fig04_convergence/      # V3 vs V4 training curves (half page)
├── fig05_bmode_comparison/ # GT vs predicted B-mode grid (full page)
├── fig06_material_maps/    # c/α/σ field visualization (half page)
├── fig07_ablation/         # Ablation study results (half page)
├── fig08_performance/      # Speed/memory comparison (half page)
└── README.md               # This file
```

Each figure directory contains:
- `SPEC.md` — Figure specification (content, style, dimensions)
- `code/` — Generation scripts (Python/matplotlib/tikz)
- `data/` — Raw data files (CSV/NPY) used to generate the figure
- `*.svg` — Vector output (primary)
- `*.png` — Raster output (300 DPI, for review)
- `*.pdf` — PDF output (for LaTeX)

## Color Scheme
- **Primary:** #1B3A5C (dark blue) — physics/framework elements
- **Accent:** #E67E22 (orange) — learnable/GNN elements
- **Success:** #27AE60 (green) — our method results
- **Baseline:** #95A5A6 (grey) — comparison methods
- **Alert:** #E74C3C (red) — failure/limitation

## Typography
- Font: Helvetica/Arial (sans-serif)
- Axis labels: 10pt
- Tick labels: 8pt
- Annotations: 9pt
- Title: 12pt bold

## IEEE TMI Figure Requirements
- Column width: 3.5 inches (single) / 7.16 inches (double)
- Max height: 9.5 inches
- Resolution: 300 DPI minimum (600 DPI for line art)
- File format: PDF/EPS preferred, TIFF/PNG accepted
- Color: RGB for online, CMYK conversion handled by publisher
