# Figure 1: Architecture Overview

**Type:** Full-page diagram (7.16 × 4.5 inches)
**Format:** SVG (primary) + PNG (300 DPI) + PDF
**Style:** Clean vector, minimal decoration, IEEE TMI standard

## Content Description
The complete DPC-GNN-Acoustic V4 pipeline showing the Physics-as-Forward paradigm.

### Layout (top-to-bottom flow)
```
┌─────────────────────────────────────────────────────────┐
│  CT Image (256×256)                                      │
│  [grey box, sample CT slice preview]                     │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────────────────┐                            │
│  │  GNN Encoder             │  ← ORANGE box (learnable) │
│  │  253K parameters         │                            │
│  │  Antisymmetric MP        │                            │
│  │  m_ij = -m_ji            │                            │
│  └──────┬───────┬───────┬───┘                            │
│         │       │       │                                │
│     c(x,y)  α(x,y)  σ(x,y)  ← Material property maps   │
│   [1400-1700] [0-50]  [0-1]    with colorbars            │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────┐                            │
│  │  Leapfrog Wave Equation  │  ← BLUE box (0 params)    │
│  │  1754 time steps         │                            │
│  │  CFL = 0.206, PML       │                            │
│  │  Pulse-echo geometry     │                            │
│  └──────┬───────────────────┘                            │
│         │                                                │
│     p(x,y,t)  ← pressure snapshots (2-3 time points)    │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────┐                            │
│  │  DAS Beamformer          │  ← BLUE box (0 params)    │
│  │  128 elements            │                            │
│  │  Hilbert envelope        │                            │
│  │  Log compression         │                            │
│  └──────┬───────────────────┘                            │
│         │                                                │
│     B-mode (128×128)                                     │
│         │                                                │
│         ▼                                                │
│  Loss: L1 + (1-SSIM) vs k-Wave GT                       │
│  [comparison: predicted vs GT side by side]              │
│                                                          │
│  ◄──── Gradient flows backward through entire pipeline   │
│        (dashed orange arrow from loss back to GNN)       │
└─────────────────────────────────────────────────────────┘
```

### Key Visual Elements
1. **Orange box** for GNN Encoder — ONLY learnable component
2. **Blue boxes** for Leapfrog and DAS — "0 learnable parameters" label
3. **Dashed orange arrow** showing gradient backpropagation through physics
4. **Small preview images** at each stage (CT, c-field, pressure snapshots, B-mode)
5. **Parameter counts** annotated at each module

### Caption
"Fig. 1. Architecture of DPC-GNN-Acoustic. The GNN encoder (orange, 253K parameters) predicts spatially-varying material properties from CT input. The Leapfrog wave equation solver and DAS beamformer (blue) contain zero learnable parameters — they execute deterministic physics. Gradients flow backward through the entire 1754-step physics pipeline (dashed arrow), enabling end-to-end optimization of material property prediction."

## Generation Notes
- Tool: matplotlib + custom drawing OR tikz OR Excalidraw
- Pressure snapshots: extract from actual model forward pass
- CT preview: use sample_0100 (has inclusion structure)
