#!/usr/bin/env python3
"""Test using a non-reserved parameter name."""

import torch
from torch_geometric.nn import MessagePassing

# Try with non-reserved parameter name
class TestMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
        self.eps = 1e-8
    
    def forward(self, x, ei, ea, node_volumes):
        # Pre-compute corrected weights
        edge_weight_tensor = self._compute_edge_weight(ea, node_volumes, ei)
        my_weights = edge_weight_tensor.squeeze(-1) if edge_weight_tensor.dim() == 2 else edge_weight_tensor
        # Use non-reserved name 'my_weights' instead of 'edge_weight'
        return self.propagate(ei, x=x, my_weights=my_weights)
    
    def message(self, x_j, x_i, my_weights):
        return my_weights.unsqueeze(-1) * (x_j - x_i)
    
    def _compute_edge_weight(self, edge_attr, node_volumes, edge_index):
        distance = edge_attr[:, 3:4] + self.eps
        if node_volumes is not None and edge_index is not None:
            src = edge_index[0]
            V_i = node_volumes[src].unsqueeze(-1)
            weight = 1.0 / (V_i * distance)
        else:
            weight = 1.0 / distance
        return weight

x = torch.randn(10, 16)
ei = torch.randint(0, 10, (2, 20))
ea = torch.randn(20, 6)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

print("=" * 70)
print("Testing with non-reserved parameter name 'my_weights':")
print("=" * 70)

mp = TestMP()
try:
    out = mp(x, ei, ea, node_volumes)
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: {x.shape}")
    if out.shape == x.shape:
        print("✅ SUCCESS!")
    else:
        print(f"❌ FAIL: {out.shape} != {x.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
