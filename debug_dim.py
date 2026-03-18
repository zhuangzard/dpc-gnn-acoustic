#!/usr/bin/env python3
"""Debug the dimension mismatch error."""

import torch
import sys
sys.path.insert(0, '/Users/taisenzhuang/workspace/DPC-GNN-Acoustic/src')

from models.acoustic_wave_gnn import WaveEquationMP

device = "cpu"
N = 10  # Use smaller N for debugging
hdim = 16

# Create MP layer
mp = WaveEquationMP(hdim=hdim, edge_dim=6, frequency=5e6).to(device)

# Create test data
x = torch.randn(N, hdim, device=device)
print(f"x shape: {x.shape}")

# Create edge index
ei = torch.randint(0, N, (2, 20), device=device)  # 20 edges
print(f"edge_index shape: {ei.shape}")

# Create edge attributes
ea = torch.randn(20, 6, device=device)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
print(f"edge_attr shape: {ea.shape}")

# Create node volumes
node_volumes = torch.randn(N, 1, device=device).abs() + 0.1
print(f"node_volumes shape: {node_volumes.shape}")

# Test forward pass
print("\nTesting forward pass...")
try:
    output = mp(x, ei, ea, node_volumes=node_volumes)
    print(f"✅ Output shape: {output.shape}")
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    print("✅ Shape matches!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
