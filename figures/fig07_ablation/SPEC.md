# Figure 7: Ablation Study Results

**Type:** Half-page (7.16 × 3.0 inches)
**Format:** SVG + PNG (300 DPI) + PDF

## Content Description
Horizontal bar chart showing SSIM for each ablation variant.

### Layout
Horizontal bar chart, sorted by SSIM (highest at top).

### Ablation Variants (8 experiments)
| ID | Variant | Expected SSIM | ΔSSIM |
|----|---------|---------------|-------|
| Full | Full V4 model | [PLACEHOLDER] | — |
| A1 | w/o Residual (direct c prediction) | [PLACEHOLDER] | ~-5-10% |
| A2 | w/o Antisymmetric MP (symmetric) | [PLACEHOLDER] | ~-3-8% |
| A3 | w/o PML (reflective boundaries) | [PLACEHOLDER] | ~-5-15% |
| A4 | 500 time steps (vs 1754) | [PLACEHOLDER] | ~-10-20% |
| A5 | 877 time steps (half) | [PLACEHOLDER] | ~-5-10% |
| A6 | L1 loss only (no SSIM) | [PLACEHOLDER] | ~-2-5% |
| A7 | w/o Attenuation (α=0) | [PLACEHOLDER] | ~-3-8% |

### Visual Style
- Bars: gradient from green (full model) to orange/red (worst ablation)
- Full model bar highlighted with bold border
- ΔSSIM annotated on each bar
- Vertical dashed line at full model SSIM
- Error bars: ±std from 5-seed runs (when available)

### Caption
"Fig. 7. Ablation study results. Each variant removes one component from the full DPC-GNN-Acoustic V4 model. The dashed line indicates full model performance. Removing PML boundaries (-XX%) and reducing time steps (-XX%) have the largest impact, confirming that physical completeness of the forward model directly determines output quality. The residual parameterization and antisymmetric message passing each contribute XX% SSIM improvement."

## Generation Code Requirements
```python
# Save to: fig07_ablation/code/plot_ablation.py
# Input: data/ablation_results.csv (columns: variant, ssim_mean, ssim_std)
# Output: fig07_ablation.svg, fig07_ablation.png
```

## Data Files Needed
- `data/ablation_results.csv` — generated after running all ablation experiments
