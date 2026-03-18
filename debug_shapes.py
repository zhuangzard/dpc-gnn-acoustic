#!/usr/bin/env python3
"""Debug the shapes step by step."""

import torch
import torch.nn as nn

class DebugLaplacian(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, x, edge_index, edge_attr, node_volumes):
        print(f"\n[DEBUG] Step 1: Inputs")
        print(f"  x shape: {x.shape}")
        print(f"  edge_index shape: {edge_index.shape}")
        print(f"  edge_attr shape: {edge_attr.shape}")
        print(f"  node_volumes shape: {node_volumes.shape}")
        
        # 1. Compute corrected edge weights
        edge_weight = self._compute_edge_weight(edge_attr, node_volumes, edge_index)
        print(f"\n[DEBUG] Step 2: edge_weight")
        print(f"  edge_weight shape: {edge_weight.shape}")
        
        # 2. Get node features for edges
        x_j = x[edge_index[0]]
        x_i = x[edge_index[1]]
        print(f"\n[DEBUG] Step 3: Node features")
        print(f"  x_j shape: {x_j.shape}")
        print(f"  x_i shape: {x_i.shape}")
        
        # 3. Compute messages
        print(f"\n[DEBUG] Step 4: Before computing message")
        print(f"  edge_weight.unsqueeze(-1) shape: {edge_weight.unsqueeze(-1).shape}")
        print(f"  (x_j - x_i) shape: {(x_j - x_i).shape}")
        
        msg = edge_weight.unsqueeze(-1) * (x_j - x_i)
        print(f"\n[DEBUG] Step 5: message")
        print(f"  msg shape: {msg.shape}")
        
        return msg
    
    def _compute_edge_weight(self, edge_attr, node_volumes, edge_index):
        distance = edge_attr[:, 3:4] + self.eps
        print(f"\n[DEBUG] _compute_edge_weight")
        print(f"  distance shape: {distance.shape}")
        
        if node_volumes is not None and edge_index is not None:
            src = edge_index[0]
            print(f"  src shape: {src.shape}")
            V_i = node_volumes[src].unsqueeze(-1)
            print(f"  V_i shape: {V_i.shape}")
            weight = 1.0 / (V_i * distance)
            print(f"  weight shape: {weight.shape}")
        else:
            weight = 1.0 / distance
        return weight

x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 20))
edge_attr = torch.randn(20, 6)
edge_attr[:, 3] = torch.abs(edge_attr[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

mp = DebugLaplacian()
msg = mp(x, edge_index, edge_attr, node_volumes)
