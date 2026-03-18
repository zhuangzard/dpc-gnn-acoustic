#!/usr/bin/env python3
"""Test using PyG's special edge_index_j parameter."""

import torch
from torch_geometric.nn import MessagePassing

print("=" * 70)
print("Testing PyG's edge_index_j parameter")
print("=" * 70)

class MP5(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index, edge_attr, node_volumes):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, node_volumes=node_volumes)
    
    def message(self, x_j, x_i, edge_attr, node_volumes, edge_index_j):
        # edge_index_j is the source node index (automatically provided by PyG)
        V_i = node_volumes[edge_index_j].unsqueeze(-1)
        distance = edge_attr[:, 3:4] + 1e-8
        weight = 1.0 / (V_i * distance)
        return weight * (x_j - x_i)

mp5 = MP5()
x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 20))
edge_attr = torch.randn(20, 6)
edge_attr[:, 3] = torch.abs(edge_attr[:, 3]) + 0.001
node_volumes = torch.randn(10, 1).abs() + 0.1

print(f"x shape: {x.shape}")
print(f"edge_index shape: {edge_index.shape}")
print(f"edge_attr shape: {edge_attr.shape}")
print(f"node_volumes shape: {node_volumes.shape}")

out5 = mp5(x, edge_index, edge_attr, node_volumes)
print(f"\nOutput shape: {out5.shape} (expected: {x.shape}) - {'✅' if out5.shape == x.shape else '❌'}")

if out5.shape == x.shape:
    print("\n✅ Using edge_index_j works correctly!")
    print("This is the proper PyG way to access source node indices")
