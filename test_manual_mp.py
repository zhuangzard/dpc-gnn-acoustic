#!/usr/bin/env python3
"""Test manual message passing (bypass propagate)."""

import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class ManualMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
        self.eps = 1e-8
    
    def forward(self, x, ei, ea, node_volumes):
        # Manually implement message passing to avoid PyG's broadcasting issues
        # 1. Compute corrected edge weights
        edge_weight_tensor = self._compute_edge_weight(ea, node_volumes, ei)
        edge_weight = edge_weight_tensor.squeeze(-1) if edge_weight_tensor.dim() == 2 else edge_weight_tensor
        
        # 2. Get node features for edges
        x_j = x[ei[0]]  # Source nodes (E, F)
        x_i = x[ei[1]]  # Target nodes (E, F)
        
        # 3. Compute messages
        msg = edge_weight.unsqueeze(-1) * (x_j - x_i)  # (E, F)
        
        # 4. Aggregate messages to target nodes
        aggr_msg = scatter(msg, ei[1], dim=0, dim_size=x.shape[0], reduce='add')  # (N, F)
        
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
ei = torch.randint(0, 10, (2, 20))
ea = torch.randn(20, 6)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

print("=" * 70)
print("Testing manual message passing (bypass propagate):")
print("=" * 70)

mp = ManualMP()
try:
    out = mp(x, ei, ea, node_volumes)
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: {x.shape}")
    if out.shape == x.shape:
        print("✅ SUCCESS! Shape matches!")
        
        # Verify V_i is used
        src = ei[0, 0].item()
        V_i_0 = node_volumes[src].item()
        d_0 = ea[0, 3].item()
        expected_weight = 1.0 / (V_i_0 * d_0)
        actual_weight_tensor = mp._compute_edge_weight(ea[0:1], node_volumes, ei[:, 0:1])
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
