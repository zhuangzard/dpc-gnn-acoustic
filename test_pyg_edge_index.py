#!/usr/bin/env python3
"""Check if PyG MessagePassing provides access to edge_index."""

import torch
from torch_geometric.nn import MessagePassing

class TestMP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', node_dim=0)
    
    def forward(self, x, edge_index):
        print(f"Before propagate: edge_index attr exists? {hasattr(self, 'edge_index')}")
        if hasattr(self, 'edge_index'):
            print(f"  self.edge_index = {self.edge_index}")
        
        result = self.propagate(edge_index, x=x)
        
        print(f"After propagate: edge_index attr exists? {hasattr(self, 'edge_index')}")
        if hasattr(self, 'edge_index'):
            print(f"  self.edge_index = {self.edge_index}")
        
        return result
    
    def message(self, x_j, x_i):
        print(f"In message: edge_index attr exists? {hasattr(self, 'edge_index')}")
        if hasattr(self, 'edge_index'):
            print(f"  self.edge_index = {self.edge_index}")
        return x_j - x_i

mp = TestMP()
x = torch.randn(10, 16)
ei = torch.randint(0, 10, (2, 20))

print("Testing...")
out = mp(x, ei)
print(f"\nOutput shape: {out.shape}")
