#!/usr/bin/env python3
"""
example_usage.py — Simple example of using refactored AcousticWaveGNN

This script demonstrates how to:
1. Create an AcousticWaveGNN model
2. Prepare input data
3. Run forward pass
4. Compute loss and backpropagate
"""

import sys
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, 'src')
from models.acoustic_wave_gnn import AcousticWaveGNN

print("=" * 70)
print("DPC-GNN-Acoustic: Example Usage")
print("=" * 70)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# ─────────────────────────────────────────────────────────────
# 1. Create Model
# ─────────────────────────────────────────────────────────────
print("\n1. Creating AcousticWaveGNN model...")

model = AcousticWaveGNN(
    hdim=64,       # Hidden dimension (类比 SolidGNN.hdim)
    n_layers=6,    # Number of MP layers
    node_dim=4,    # Input: [ρ, c, α, HU]
    edge_dim=6     # Edge: [r_vec(3), distance(1), Z_ratio(1), atten_factor(1)]
).to(device)

print(f"   Model created with {model.count_params():,} parameters")
print(f"   Architecture: Encoder → {model.n_layers}×WaveEquationMP → Decoder")

# ─────────────────────────────────────────────────────────────
# 2. Prepare Input Data
# ─────────────────────────────────────────────────────────────
print("\n2. Preparing input data...")

N = 500  # Number of nodes
E = 3000  # Number of edges

# Node features: [ρ, c, α, HU]
# For liver tissue: ρ=1050 kg/m³, c=1540 m/s, α=0.5 Np/m, HU=50
nf = torch.tensor([
    [1050.0/1000.0, 1540.0/1540.0, 0.5/10.0, 50.0/1000.0]
] * N, device=device)

# Random graph (in practice, use build_acoustic_graph)
ei = torch.randint(0, N, (2, E), device=device)

# Edge attributes: [dx, dy, dz, distance, Z_ratio, atten_factor]
ea = torch.randn(E, 6, device=device)
ea[:, 3] = ea[:, 3].abs() + 0.001  # positive distance
ea[:, 4] = 1.0  # Z_ratio (uniform for same medium)
ea[:, 5] = 0.1  # attenuation factor

# Time step and sound speed
dt = torch.tensor(1e-7, device=device)
c = torch.full((N, 1), 1540.0, device=device)

print(f"   Nodes: {N} | Edges: {E}")
print(f"   Node features: {nf.shape}")
print(f"   Edge index: {ei.shape}")
print(f"   Edge attributes: {ea.shape}")
print(f"   Time step: {dt.item():.2e} s")
print(f"   Sound speed: {c[0].item():.0f} m/s")

# ─────────────────────────────────────────────────────────────
# 3. Forward Pass
# ─────────────────────────────────────────────────────────────
print("\n3. Running forward pass...")

p = model(nf, ei, ea, dt, c)

print(f"   Output pressure shape: {p.shape}")
print(f"   Pressure range: [{p.min().item():.4e}, {p.max().item():.4e}]")
print(f"   Mean pressure: {p.mean().item():.4e}")

# ─────────────────────────────────────────────────────────────
# 4. Compute Loss
# ─────────────────────────────────────────────────────────────
print("\n4. Computing loss...")

# Target: zero pressure (relaxation)
target = torch.zeros_like(p)
loss = nn.MSELoss()(p, target)

print(f"   MSE Loss: {loss.item():.6e}")

# ─────────────────────────────────────────────────────────────
# 5. Backpropagation
# ─────────────────────────────────────────────────────────────
print("\n5. Backpropagation...")

loss.backward()

# Check gradients
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"   Total gradient norm: {grad_norm:.4e}")

# ─────────────────────────────────────────────────────────────
# 6. Optimization Step
# ─────────────────────────────────────────────────────────────
print("\n6. Optimization step...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.step()
optimizer.zero_grad()

print(f"   ✅ Optimizer step completed")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ Example completed successfully!")
print("=" * 70)

print("\nKey points:")
print("  - Model uses DPC-GNN naming: hdim, ei, ea, nf")
print("  - Architecture: enc → mps → dec (类比 SolidGNN)")
print("  - Input: node features [ρ, c, α, HU]")
print("  - Output: pressure field [N, 1]")
print("\nNext steps:")
print("  - Run training: python3 train_acoustic.py --medium liver")
print("  - Check docs: docs/STYLE_COMPARISON.md")
print("  - View README: README_REFACTORED.md")

print("\n" + "=" * 70)
