#!/usr/bin/env python3
"""Test if the issue existed before our fix."""

import torch
import sys

# Temporarily use the original (unfixed) version
print("=" * 70)
print("Testing ORIGINAL (buggy) version from GitHub")
print("=" * 70)

# Simulate the original buggy _compute_edge_weight
class OriginalWaveEquationMP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import MessagePassing
        self.mp = type('MP', (MessagePassing,), {
            'forward': lambda self, x, ei, ea, node_volumes: self.propagate(ei, x=x, ea=ea, node_volumes=node_volumes),
            'message': lambda self, x_j, x_i, ea, node_volumes: self._compute_edge_weight(ea, node_volumes) * (x_j - x_i),
            '_compute_edge_weight': lambda self, edge_attr, node_volumes: 1.0 / (edge_attr[:, 3:4] + 1e-8),
            'update': lambda self, aggr_out: aggr_out,
            'aggr': 'add',
            'node_dim': 0,
        })()
    
    def forward(self, x, ei, ea, node_volumes=None):
        return self.mp(x, ei, ea, node_volumes)

try:
    from torch_geometric.nn import MessagePassing
    
    device = "cpu"
    N = 10
    hdim = 16
    
    mp = OriginalWaveEquationMP().mp
    x = torch.randn(N, hdim, device=device)
    ei = torch.randint(0, N, (2, 20), device=device)
    ea = torch.randn(20, 6, device=device)
    ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
    node_volumes = torch.randn(N, 1, device=device).abs() + 0.1
    
    print(f"x shape: {x.shape}")
    print(f"edge_index shape: {ei.shape}")
    
    output = mp(x, ei, ea, node_volumes)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {x.shape}")
    print(f"Match: {output.shape == x.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
