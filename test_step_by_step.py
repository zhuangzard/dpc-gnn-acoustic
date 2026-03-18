#!/usr/bin/env python3
"""Step-by-step debugging to find where the shape breaks."""

import torch
from torch_geometric.nn import MessagePassing

print("=" * 70)
print("Step 1: Basic MP (works)")
print("=" * 70)

class MP1(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j, x_i):
        return x_j - x_i

mp1 = MP1()
x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 20))
out1 = mp1(x, edge_index)
print(f"Output shape: {out1.shape} (expected: {x.shape}) - {'✅' if out1.shape == x.shape else '❌'}")

print("\n" + "=" * 70)
print("Step 2: MP with edge_attr")
print("=" * 70)

class MP2(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, x_i, edge_attr):
        return edge_attr[:, :1] * (x_j - x_i)

mp2 = MP2()
edge_attr = torch.randn(20, 6)
out2 = mp2(x, edge_index, edge_attr)
print(f"Output shape: {out2.shape} (expected: {x.shape}) - {'✅' if out2.shape == x.shape else '❌'}")

print("\n" + "=" * 70)
print("Step 3: MP with edge_attr and node_volumes")
print("=" * 70)

class MP3(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index, edge_attr, node_volumes):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, node_volumes=node_volumes)
    
    def message(self, x_j, x_i, edge_attr, node_volumes):
        weight = edge_attr[:, 3:4] + 1e-8
        return (1.0 / weight) * (x_j - x_i)

mp3 = MP3()
node_volumes = torch.randn(10, 1).abs() + 0.1
out3 = mp3(x, edge_index, edge_attr, node_volumes)
print(f"Output shape: {out3.shape} (expected: {x.shape}) - {'✅' if out3.shape == x.shape else '❌'}")

print("\n" + "=" * 70)
print("Step 4: MP with edge_index stored")
print("=" * 70)

class MP4(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index, edge_attr, node_volumes):
        self._ei = edge_index
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, node_volumes=node_volumes)
        self._ei = None
        return out
    
    def message(self, x_j, x_i, edge_attr, node_volumes):
        # Use stored edge_index
        ei = self._ei
        src = ei[0]
        V_i = node_volumes[src].unsqueeze(-1)
        distance = edge_attr[:, 3:4] + 1e-8
        weight = 1.0 / (V_i * distance)
        return weight * (x_j - x_i)

mp4 = MP4()
out4 = mp4(x, edge_index, edge_attr, node_volumes)
print(f"Output shape: {out4.shape} (expected: {x.shape}) - {'✅' if out4.shape == x.shape else '❌'}")

if out4.shape != x.shape:
    print("\n❌ Shape breaks when we store and use edge_index!")
    print("This is likely a PyG version or configuration issue")
