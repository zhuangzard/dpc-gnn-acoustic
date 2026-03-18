#!/usr/bin/env python3
"""Test the NEW fixed version."""

import torch
from torch_geometric.nn import MessagePassing

# NEW Fixed version: pre-compute weights in forward, pass as edge_weight
class NewFixedMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
        self.eps = 1e-8
    
    def forward(self, x, ei, ea, node_volumes):
        # Pre-compute corrected weights in forward
        edge_weight_tensor = self._compute_edge_weight(ea, node_volumes, ei)
        edge_weight = edge_weight_tensor.squeeze(-1) if edge_weight_tensor.dim() == 2 else edge_weight_tensor
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

x = torch.randn(10, 16)
ei = torch.randint(0, 10, (2, 20))
ea = torch.randn(20, 6)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

print("=" * 70)
print("Testing NEW Fixed MP (pre-compute weights in forward):")
print("=" * 70)

mp = NewFixedMP()
try:
    out = mp(x, ei, ea, node_volumes)
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: {x.shape}")
    if out.shape == x.shape:
        print("✅ SUCCESS! Shape matches!")
        
        # Verify that weights are actually using V_i
        print("\nVerifying that V_i is used in weight calculation:")
        src = ei[0, 0].item()
        V_i_0 = node_volumes[src].item()
        d_0 = ea[0, 3].item()
        expected_weight = 1.0 / (V_i_0 * d_0)
        
        # Get actual weight
        actual_weight_tensor = mp._compute_edge_weight(ea[0:1], node_volumes, ei[:, 0:1])
        actual_weight = actual_weight_tensor.item()
        
        print(f"  Edge 0: src={src}, V_i={V_i_0:.4f}, d={d_0:.4f}")
        print(f"  Expected weight 1/(V_i*d): {expected_weight:.4f}")
        print(f"  Actual weight: {actual_weight:.4f}")
        print(f"  Match: {abs(expected_weight - actual_weight) < 1e-5} {'✅' if abs(expected_weight - actual_weight) < 1e-5 else '❌'}")
    else:
        print(f"❌ FAIL: Shape mismatch {out.shape} != {x.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
