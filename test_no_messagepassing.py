#!/usr/bin/env python3
"""Test completely manual implementation (no MessagePassing)."""

import torch
import torch.nn as nn

class ManualLaplacian(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, x, edge_index, edge_attr, node_volumes):
        # 1. Compute corrected edge weights
        edge_weight = self._compute_edge_weight(edge_attr, node_volumes, edge_index)
        
        # 2. Get node features for edges
        x_j = x[edge_index[0]]  # Source nodes (E, F)
        x_i = x[edge_index[1]]  # Target nodes (E, F)
        
        # 3. Compute messages: w_ij * (x_j - x_i)
        msg = edge_weight.unsqueeze(-1) * (x_j - x_i)  # (E, F)
        
        # 4. Aggregate messages to target nodes
        # Create output tensor
        N = x.shape[0]
        aggr_msg = torch.zeros_like(x)  # (N, F)
        
        # Scatter add using index_add_
        aggr_msg.index_add_(0, edge_index[1], msg)
        
        return aggr_msg
    
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
edge_index = torch.randint(0, 10, (2, 20))
edge_attr = torch.randn(20, 6)
edge_attr[:, 3] = torch.abs(edge_attr[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

print("=" * 70)
print("Testing completely manual implementation (no MessagePassing):")
print("=" * 70)

mp = ManualLaplacian()
try:
    out = mp(x, edge_index, edge_attr, node_volumes)
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: {x.shape}")
    if out.shape == x.shape:
        print("✅ SUCCESS! Shape matches!")
        
        # Verify V_i is used
        src = edge_index[0, 0].item()
        V_i_0 = node_volumes[src].item()
        d_0 = edge_attr[0, 3].item()
        expected_weight = 1.0 / (V_i_0 * d_0)
        actual_weight_tensor = mp._compute_edge_weight(edge_attr[0:1], node_volumes, edge_index[:, 0:1])
        actual_weight = actual_weight_tensor.item()
        
        print(f"\nVerifying V_i usage:")
        print(f"  Expected weight: {expected_weight:.4f}")
        print(f"  Actual weight: {actual_weight:.4f}")
        print(f"  Match: {abs(expected_weight - actual_weight) < 1e-5} {'✅' if abs(expected_weight - actual_weight) < 1e-5 else '❌'}")
    else:
        print(f"❌ FAIL: {out.shape} != {x.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
