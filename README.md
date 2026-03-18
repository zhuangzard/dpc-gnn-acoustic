# DPC-GNN-Acoustic

**Data-free Physics-Constrained Graph Neural Network for Acoustic Simulation**

Real-time differentiable ultrasound simulation with hard physics constraints.

## Overview

DPC-GNN-Acoustic extends the DPC-GNN paradigm to acoustic wave propagation, enabling real-time CT-to-Ultrasound synthesis with guaranteed physical correctness.

### Key Innovation

**Hard Physics + Differentiability**: Wave equation as a differentiable Message Passing layer

```
Traditional: Soft constraint (loss function) → PINNs
DPC-GNN-Mech: Hard constraint (network structure) → Not differentiable  
DPC-GNN-Acoustic: Hard + Differentiable constraint → OUR METHOD ⭐
```

## Physics

### Wave Equation as MP Layer

```python
∂²p/∂t² = c² ∇²p  (Acoustic wave equation)

Discrete form in GNN:
p_new = 2*p - p_old + (c*dt)² * Laplacian_MP(p)
```

### Key Features

- **Hard constraint**: Each MP step satisfies wave equation
- **Differentiable**: Gradients flow through physics
- **Real-time**: 100ms inference (vs 40min physical simulation)

## Project Structure

```
dpc-gnn-acoustic/
├── src/
│   ├── models/          # WaveEquationMP, AcousticGNN
│   ├── physics/         # Acoustic properties, wave propagation
│   ├── data/            # CT loader, US generator
│   └── training/        # Training scripts, losses
├── configs/             # Experiment configs
├── experiments/         # Experiment logs
└── docs/                # Paper outline, notes
```

## Installation

```bash
git clone https://github.com/zhuangzard/dpc-gnn-acoustic.git
cd dpc-gnn-acoustic
pip install -r requirements.txt
```

## Usage

```python
from dpc_gnn_acoustic import AcousticGNN

model = AcousticGNN(
    acoustic_props={'c': 1540, 'rho': 1000, 'alpha': 0.5},
    mesh_resolution=0.5  # mm
)

# CT → US (real-time, physically correct)
us_image = model(ct_volume, probe_pos=(x, y, z))
```

## Paper

**Title**: "DPC-GNN-Acoustic: Differentiable Hard-Physics Ultrasound Simulation"

**Target**: MICCAI 2026 / TMI

## Related Work

- DPC-GNN (Mechanics): github.com/zhuangzard/medical-robotics-sim
- DiffUS (Differentiable Rendering): arXiv:2508.06768
- PINNs: Physics-Informed Neural Networks
- MeshGraphNets: DeepMind, ICML 2021

## License

MIT

## Contact

Taisen Zhuang <tszhuang@wharton.upenn.edu>
