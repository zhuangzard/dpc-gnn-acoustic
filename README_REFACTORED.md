# DPC-GNN-Acoustic (Refactored)

**Refactored to match DPC-GNN style conventions**

This is a refactored version of DPC-GNN-Acoustic, fully aligned with the coding style and architecture of the main [DPC-GNN](https://github.com/taisenzhuang/DPC-GNN) project.

## 🎯 What Changed

### Core Changes
1. **Naming Convention**: Unified with DPC-GNN
   - `hidden_dim` → `hdim`
   - `edge_index` → `ei` (in functions)
   - `edge_attr` → `ea` (in functions)
   - `node_feats` → `nf`

2. **Architecture**: Mirrors SolidGNN structure
   - `enc` → Encoder (MLP)
   - `mps` → Message Passing layers
   - `dec` → Decoder (MLP)

3. **Training Script**: New `train_acoustic.py` analogous to `solid_tissue_train_hires.py`
   - `train_medium()` for single medium
   - `main()` loops over multiple media
   - Same optimizer, scheduler, and checkpoint structure

## 📁 File Structure

```
DPC-GNN-Acoustic/
├── src/
│   └── models/
│       ├── acoustic_wave_gnn.py      # ✨ NEW: Core GNN (DPC-GNN style)
│       ├── wave_equation_mp.py       # Original (kept for reference)
│       └── acoustic_gnn.py           # Original (kept for reference)
├── train_acoustic.py                 # ✨ NEW: Training script
├── docs/
│   └── STYLE_COMPARISON.md           # ✨ NEW: Style comparison table
└── README_REFACTORED.md              # This file
```

## 🚀 Quick Start

### Test the Model
```bash
python3 src/models/acoustic_wave_gnn.py
```

### Train on Single Medium
```bash
python3 train_acoustic.py --medium liver --epochs 500
```

### Train All Media
```bash
python3 train_acoustic.py --epochs 500
```

### Custom Medium
```bash
python3 train_acoustic.py --medium custom --c 1600 --rho 1100 --alpha 1.0
```

## 📊 Supported Media

| Medium   | c [m/s] | ρ [kg/m³] | α [Np/m] |
|----------|---------|-----------|----------|
| liver    | 1540    | 1050      | 0.5      |
| fat      | 1450    | 950       | 0.3      |
| muscle   | 1580    | 1050      | 0.8      |
| bone     | 3000    | 1900      | 10.0     |
| water    | 1480    | 1000      | 0.02     |
| blood    | 1570    | 1060      | 0.2      |

## 🔧 Architecture

### AcousticWaveGNN (类比 SolidGNN)

```python
class AcousticWaveGNN(nn.Module):
    def __init__(self, hdim=64, n_layers=6, node_dim=4, edge_dim=6):
        super().__init__()
        self.hdim = hdim
        self.n_layers = n_layers
        
        # Encoder (类比 SolidGNN.enc)
        self.enc = nn.Sequential(
            nn.Linear(node_dim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU()
        )
        
        # Message Passing (类比 SolidGNN.mps)
        self.mps = nn.ModuleList([
            WaveEquationMP(hdim=hdim, edge_dim=edge_dim)
            for _ in range(n_layers)
        ])
        
        # Decoder (类比 SolidGNN.dec)
        self.dec = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
            nn.SiLU(),
            nn.Linear(hdim, 1)
        )
    
    def forward(self, nf, ei, ea, dt, c):
        h = self.enc(nf)
        for mp in self.mps:
            h = h + mp(h, ei, ea)
        p = self.dec(h)
        return p
```

## 📖 Key Features

### ✅ DPC-GNN Style Compliance
- Naming: `hdim`, `ei`, `ea`, `nf`
- Architecture: `enc/mps/dec`
- Training: `train_medium()` + `MEDIA` dict
- Initialization: `uniform(-0.01, 0.01)` + `zeros` bias
- LayerNorm + SiLU activation

### ✅ Physics-Informed
- Wave equation: ∂²p/∂t² = c²∇²p
- Leapfrog time integration (2nd-order accurate)
- CFL stability condition check
- Graph Laplacian for spatial discretization

### ✅ Complete Training Pipeline
- CosineAnnealingLR with warmup
- Gradient clipping (max_norm=1.0)
- Automatic checkpointing
- JSON logging (results + history)
- Multi-medium training loop

## 📚 Documentation

- **[STYLE_COMPARISON.md](docs/STYLE_COMPARISON.md)**: Detailed comparison with original code
- **Expert Council Comments**: 5-expert review format in code

## 🔬 Example Output

```
======================================================================
  MEDIUM: LIVER | c=1540 m/s | ρ=1050 kg/m³ | α=0.5 Np/m
  Grid: 20x20x20 | Device: cuda | Epochs: 500
  Time step: 2.50e-08 s (CFL)
======================================================================
  Nodes: 9261 | Edges: 146456
  Params: 33,489

     Ep |         Loss |      Physics |      P_max |        lr
  ------------------------------------------------------------
      1 |   1.234567e+00 |   1.234567e+00 |   1.2345e+00 |   1.00e-06
     25 |   5.678901e-01 |   5.678901e-01 |   2.3456e+00 |   9.95e-04
    ...
    500 |   1.234567e-04 |   1.234567e-04 |   9.8765e-01 |   1.00e-05

======================================================================
  ✅ LIVER COMPLETE
  Best loss: 1.234567e-04 (epoch 487)
  Training time: 123.4s (2.1min)
======================================================================
```

## 🤝 Integration with DPC-GNN

This refactored version is designed to seamlessly integrate with the main DPC-GNN project:

1. **Consistent naming** allows code sharing
2. **Similar architecture** enables transfer learning
3. **Same training loop** simplifies experimentation
4. **Unified style** improves maintainability

## 📝 Citation

If you use this code, please cite both DPC-GNN and the acoustic simulation references:

```bibtex
@article{dpcgnn2024,
  title={DPC-GNN: Differentiable Physics-Constrained Graph Neural Networks},
  author={...},
  journal={...},
  year={2024}
}

@article{meshgraphnets2021,
  title={MeshGraphNets: Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and others},
  booktitle={ICLR},
  year={2021}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Last Updated**: 2026-03-18  
**Refactored by**: GLM-5 (following DPC-GNN style guidelines)
