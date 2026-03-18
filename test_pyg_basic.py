#!/usr/bin/env python3
"""Test basic PyG MessagePassing behavior."""

import torch
from torch_geometric.nn import MessagePassing

print("=" * 70)
print("Testing basic PyG MessagePassing")
print("=" * 70)

class SimpleMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j, x_i):
        # Simple message: just difference
        return x_j - x_i
    
    def update(self, aggr_out):
        return aggr_out

device = "cpu"
N = 10
hdim = 16

mp = SimpleMP()
x = torch.randn(N, hdim, device=device)
edge_index = torch.randint(0, N, (2, 20), device=device)

print(f"x shape: {x.shape}")
print(f"edge_index shape: {edge_index.shape}")

output = mp(x, edge_index)
print(f"Output shape: {output.shape}")
print(f"Expected shape: {x.shape}")
print(f"Match: {output.shape == x.shape}")

if output.shape != x.shape:
    print("\n❌ Basic PyG MessagePassing is broken or we're using it wrong!")
else:
    print("\n✅ Basic PyG MessagePassing works correctly")
