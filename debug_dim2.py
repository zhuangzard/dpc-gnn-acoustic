#!/usr/bin/env python3
"""Debug the dimension mismatch error - test both files."""

import torch
import sys
sys.path.insert(0, '/Users/taisenzhuang/workspace/DPC-GNN-Acoustic/src')

# Test acoustic_wave_gnn.py version
print("=" * 70)
print("Testing acoustic_wave_gnn.py WaveEquationMP")
print("=" * 70)

from models.acoustic_wave_gnn import WaveEquationMP as WaveEquationMP1

device = "cpu"
N = 10
hdim = 16

mp1 = WaveEquationMP1(hdim=hdim, edge_dim=6, frequency=5e6).to(device)
x = torch.randn(N, hdim, device=device)
ei = torch.randint(0, N, (2, 20), device=device)
ea = torch.randn(20, 6, device=device)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
node_volumes = torch.randn(N, 1, device=device).abs() + 0.1

print(f"x shape: {x.shape}")
print(f"edge_index shape: {ei.shape}")
print(f"edge_attr shape: {ea.shape}")
print(f"node_volumes shape: {node_volumes.shape}")

try:
    output1 = mp1(x, ei, ea, node_volumes=node_volumes)
    print(f"Output shape: {output1.shape}")
    if output1.shape == x.shape:
        print("✅ Shape matches!")
    else:
        print(f"❌ Shape mismatch: {output1.shape} != {x.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test wave_equation_mp.py version
print("\n" + "=" * 70)
print("Testing wave_equation_mp.py WaveEquationMP")
print("=" * 70)

from models.wave_equation_mp import WaveEquationMP as WaveEquationMP2

mp2 = WaveEquationMP2(aggr='add', frequency=5e6).to(device)

try:
    output2 = mp2(x, ei, ea, node_volumes=node_volumes)
    print(f"Output shape: {output2.shape}")
    if output2.shape == x.shape:
        print("✅ Shape matches!")
    else:
        print(f"❌ Shape mismatch: {output2.shape} != {x.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
