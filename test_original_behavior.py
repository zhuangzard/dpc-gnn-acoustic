#!/usr/bin/env python3
"""Test the ORIGINAL (before our fix) code to see if it has the same issue."""

import torch
import sys
sys.path.insert(0, '/Users/taisenzhuang/workspace/DPC-GNN-Acoustic/src')

# Create a simple version that mimics the original buggy code
from torch_geometric.nn import MessagePassing

class OriginalWaveEquationMP(MessagePassing):
    """Original (buggy) version that doesn't use V_i"""
    def __init__(self, hdim, edge_dim=6, aggr='add', eps=1e-8, frequency=5e6):
        super().__init__(aggr=aggr, node_dim=0)
        self.hdim = hdim
        self.edge_dim = edge_dim
        self.eps = eps
        self.frequency = frequency
    
    def forward(self, x, ei, ea, node_volumes):
        return self.propagate(ei, x=x, ea=ea, node_volumes=node_volumes)
    
    def message(self, x_j, x_i, ea, node_volumes):
        pressure_diff = x_j - x_i
        weight = self._compute_edge_weight(ea, node_volumes)
        msg = weight * pressure_diff
        return msg
    
    def _compute_edge_weight(self, edge_attr, node_volumes):
        """Original BUGGY version: claims to use V_i but doesn't"""
        D = edge_attr.shape[1]
        distance = edge_attr[:, 3:4] + self.eps
        
        # BUGGY: doesn't actually use node_volumes!
        weight = 1.0 / distance
        
        if D > 4:
            Z_ratio = edge_attr[:, 4:5]
            weight = weight * Z_ratio
        
        if D > 5:
            alpha_0 = edge_attr[:, 5:6]
            f_ref = 1e6
            alpha_f = alpha_0 * (self.frequency / f_ref)
            atten_factor = torch.exp(-alpha_f * distance)
            weight = weight * atten_factor
        
        return weight
    
    def update(self, aggr_out):
        return aggr_out

print("=" * 70)
print("Testing ORIGINAL (buggy) WaveEquationMP")
print("=" * 70)

device = "cpu"
N = 10
hdim = 16

mp = OriginalWaveEquationMP(hdim=hdim, edge_dim=6, frequency=5e6).to(device)
x = torch.randn(N, hdim, device=device)
ei = torch.randint(0, N, (2, 20), device=device)
ea = torch.randn(20, 6, device=device)
ea[:, 3] = torch.abs(ea[:, 3]) + 0.001
node_volumes = torch.randn(N, 1, device=device).abs() + 0.1

print(f"x shape: {x.shape}")
print(f"edge_index shape: {ei.shape}")
print(f"edge_attr shape: {ea.shape}")
print(f"node_volumes shape: {node_volumes.shape}")

try:
    output = mp(x, ei, ea, node_volumes)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: {x.shape}")
    print(f"Match: {output.shape == x.shape} {'✅' if output.shape == x.shape else '❌'}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
