#!/usr/bin/env python3
"""Compare original vs fixed code to find the critical difference."""

import torch
from torch_geometric.nn import MessagePassing

# Original (works)
class OriginalMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
        self.eps = 1e-8
    
    def forward(self, x, ei, ea, node_volumes):
        return self.propagate(ei, x=x, ea=ea, node_volumes=node_volumes)
    
    def message(self, x_j, x_i, ea, node_volumes):
        weight = self._compute_edge_weight(ea, node_volumes)
        return weight * (x_j - x_i)
    
    def _compute_edge_weight(self, edge_attr, node_volumes):
        distance = edge_attr[:, 3:4] + self.eps
        weight = 1.0 / distance  # Doesn't use node_volumes
        return weight

# Fixed (broken)
class FixedMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
        self.eps = 1e-8
    
    def forward(self, x, ei, ea, node_volumes):
        self._current_edge_index = ei  # <-- This line causes the issue!
        result = self.propagate(ei, x=x, ea=ea, node_volumes=node_volumes)
        del self._current_edge_index
        return result
    
    def message(self, x_j, x_i, ea, node_volumes):
        edge_index = getattr(self, '_current_edge_index', None)
        weight = self._compute_edge_weight(ea, node_volumes, edge_index)
        return weight * (x_j - x_i)
    
    def _compute_edge_weight(self, edge_attr, node_volumes, edge_index):
        distance = edge_attr[:, 3:4] + self.eps
        if node_volumes is not None and edge_index is not None:
            src = edge_index[0]
            V_i = node_volumes[src].unsqueeze(-1)
            weight = 1.0 / (V_i * distance)
        else:
            weight = 1.0 / distance
        return weight

# Test
x = torch.randn(10, 16)
ei = torch.randint(0, 10, (2, 20))
ea = torch.randn(20, 6)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

print("Testing Original MP:")
orig = OriginalMP()
out_orig = orig(x, ei, ea, node_volumes)
print(f"  Output shape: {out_orig.shape} - {'✅' if out_orig.shape == x.shape else '❌'}")

print("\nTesting Fixed MP (with instance variable):")
fixed = FixedMP()
try:
    out_fixed = fixed(x, ei, ea, node_volumes)
    print(f"  Output shape: {out_fixed.shape} - {'✅' if out_fixed.shape == x.shape else '❌'}")
except Exception as e:
    print(f"  Error: {e}")

# Now test WITHOUT storing edge_index
print("\n" + "=" * 70)
print("Testing Fixed MP WITHOUT storing edge_index (pass as kwarg):")
print("=" * 70)

class FixedMP2(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
        self.eps = 1e-8
    
    def forward(self, x, ei, ea, node_volumes):
        # DON'T store edge_index, just compute weights here
        edge_weight = self._compute_edge_weight(ea, node_volumes, ei)
        if edge_weight.dim() == 2:
            edge_weight = edge_weight.squeeze(-1)
        return self.propagate(ei, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, x_i, edge_weight):
        return edge_weight.unsqueeze(-1) * (x_j - x_i)
    
    def _compute_edge_weight(self, edge_attr, node_volumes, edge_index):
        distance = edge_attr[:, 3:4] + self.eps
        if node_volumes is not None and edge_index is not None:
            src = edge_index[0]
            V_i = node_volumes[src].unsqueeze(-1)
            weight = 1.0 / (V_i * distance)
        else:
            weight = 1.0 / distance
        return weight

fixed2 = FixedMP2()
try:
    out_fixed2 = fixed2(x, ei, ea, node_volumes)
    print(f"  Output shape: {out_fixed2.shape} - {'✅' if out_fixed2.shape == x.shape else '❌'}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
