# Figure 3: Cross-Domain Transfer of Antisymmetric Message Passing

**Type:** Half-page (7.16 × 3.0 inches)
**Format:** SVG + PNG (300 DPI) + PDF

## Content Description
Side-by-side comparison showing how the SAME antisymmetric MP architecture works in two different physics domains.

### Layout (left-center-right)

```
┌──────────────────┐   ┌─────────────┐   ┌──────────────────┐
│  Solid Mechanics  │   │   Shared    │   │  Wave Acoustics  │
│  (DPC-GNN orig.)  │   │   Math      │   │  (This work)     │
│                   │   │             │   │                   │
│  ●──F_ij──●       │   │ m_ij = -m_ji│   │  ●──p_ij──●      │
│  ●──F_ji──●       │   │             │   │  ●──p_ji──●      │
│  F_ij = -F_ji     │   │ Same arch.  │   │  p_ij = -p_ji    │
│                   │   │ Different   │   │                   │
│  Newton's 3rd Law │   │ physics     │   │  Acoustic         │
│  Force balance    │   │ interpreta- │   │  reciprocity      │
│  at material pts  │   │ tion        │   │  at spatial pts   │
│                   │   │             │   │                   │
│  Nodes: material  │   │             │   │  Nodes: grid pts  │
│  Edges: bonds     │   │             │   │  Edges: neighbors │
│  Output: deform.  │   │             │   │  Output: c,α,σ    │
└──────────────────┘   └─────────────┘   └──────────────────┘
```

### Visual Elements
1. **Left panel (blue)**: Graph visualization of solid mechanics
   - Nodes = material points (circles)
   - Edges with force arrows F_ij, F_ji (equal and opposite)
   - Label: "Newton's Third Law"
   
2. **Center**: Mathematical bridge
   - Equation: m_ij = -m_ji (highlighted)
   - "Same architecture, different physics"
   - Double-headed arrow connecting both panels

3. **Right panel (orange)**: Graph visualization of acoustic grid
   - Nodes = spatial grid points (squares/circles)
   - Edges with pressure arrows p_ij, p_ji (equal and opposite)
   - Label: "Acoustic Reciprocity"

### Caption
"Fig. 3. Cross-domain transfer of antisymmetric message passing. Left: In the original DPC-GNN for solid mechanics, antisymmetric messages (m_ij = -m_ji) enforce Newton's third law between material points. Right: In DPC-GNN-Acoustic, the same architectural constraint is reinterpreted as acoustic spatial reciprocity between grid points. The identical GNN architecture with antisymmetric message passing transfers across PDE families by changing only the physical interpretation of the inductive bias."

## Generation Notes
- Tool: matplotlib with networkx graph drawing OR tikz
- Use actual small graph examples (5-6 nodes each)
- Force/pressure arrows must be visually equal and opposite
- Color: left=blue (#1B3A5C), right=orange (#E67E22), center=neutral
