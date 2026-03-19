# Figure 2: L0-L3 Taxonomy of Physics-Informed Learning

**Type:** Half-page, 4-panel (7.16 × 3.5 inches)
**Format:** SVG + PNG (300 DPI) + PDF

## Content Description
Four-panel conceptual diagram showing each level of physics integration.

### Layout (2×2 grid)

```
┌────────────────────┬────────────────────┐
│ L0: Data-only      │ L1: Physics-as-Loss│
│                    │                    │
│ CT ──► [NN] ──► US │ CT ──► [NN] ──► US │
│         ▲          │    │    ▲    │      │
│     data only      │    └─ PDE ──┘      │
│                    │    residual loss    │
│ ❌ No physics      │ ⚠️ Soft physics    │
│ Pix2Pix, CycleGAN │ PINN (Raissi 2019)│
├────────────────────┼────────────────────┤
│ L2: Physics-as-Arch│ L3: Physics-as-Fwd │
│                    │                    │
│ CT ──► [NN+Φ] ──►US│ CT ──► [GNN] ──► θ │
│    conserv. built  │         │          │
│    into weights    │    [PHYSICS] ──► US│
│                    │    0 learnable     │
│ 🟡 Partial physics │ ✅ Full physics    │
│ HNN, LNN          │ DPC-GNN (OURS)    │
└────────────────────┴────────────────────┘
```

### Visual Style
- Each panel: simple flow diagram
- Color coding:
  - L0: grey (#95A5A6) — no physics
  - L1: yellow (#F1C40F) — soft/partial
  - L2: light green (#82E0AA) — structural
  - L3: bold green (#27AE60) — full guarantee, highlighted with border
- Physics guarantee icons: ❌ / ⚠️ / 🟡 / ✅
- Below grid: summary table with formal definitions

### Accompanying Table (below figure or in text)
| Level | What NN learns | What is fixed | Physics guarantee |
|-------|---------------|---------------|-------------------|
| L0 | Entire mapping f: CT→US | Nothing | None |
| L1 | f_θ(CT)→US, regularized by R(PDE) | PDE form | Soft (violated at test time) |
| L2 | f_θ with Φ-structured weights | Conservation structure | Structural (by design) |
| L3 | g_θ(CT)→(c,α,σ), US = P(g_θ(CT)) | Physics solver P | **Complete** (P has 0 params) |

### Caption
"Fig. 2. L0-L3 taxonomy of physics-informed learning for image synthesis. (a) L0: purely data-driven mapping with no physics constraints. (b) L1: PDE residual used as soft regularization in loss function (PINN paradigm). (c) L2: conservation laws embedded in network architecture. (d) L3: Physics-as-Forward — the neural network predicts only material properties; the entire forward model is a parameter-free physics solver. Our DPC-GNN-Acoustic belongs to L3, providing the strongest physics guarantees."

## Generation Notes
- Tool: matplotlib with custom patches OR tikz
- Keep diagrams minimal — clarity over detail
- L3 panel should be visually highlighted (thicker border, slight background tint)
