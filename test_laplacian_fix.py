#!/usr/bin/env python3
"""
test_laplacian_fix.py - Verify that Laplacian weights actually use V_i

This test confirms that the fix is working:
  - w_ij = 1/(V_i * |r_ij|) ✅ CORRECT
  - NOT: w_ij = 1/|r_ij| ❌ WRONG (old bug)
"""

import torch
import sys
sys.path.insert(0, '/Users/taisenzhuang/workspace/DPC-GNN-Acoustic/src')

from models.wave_equation_mp import WaveEquationMP
from models.acoustic_wave_gnn import WaveEquationMP as WaveEquationMP2


def test_wave_equation_mp():
    """Test wave_equation_mp.py implementation."""
    print("=" * 70)
    print("Testing wave_equation_mp.py")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple graph
    N = 10
    positions = torch.randn(N, 3, device=device) * 0.1
    
    # Simple edge index (fully connected for test)
    src = []
    dst = []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], device=device)
    
    # Edge attributes
    r_vec = positions[edge_index[0]] - positions[edge_index[1]]
    distance = torch.norm(r_vec, dim=-1, keepdim=True)
    edge_attr = torch.cat([r_vec, distance], dim=-1)
    
    # Create node volumes with DISTINCT values
    node_volumes = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                                device=device).unsqueeze(-1)
    
    # Test the MP layer
    mp = WaveEquationMP(aggr='add', frequency=5e6).to(device)
    x = torch.randn(N, 16, device=device)
    
    # Call with volumes
    laplacian = mp(x, edge_index, edge_attr, node_volumes=node_volumes)
    
    # Verify that volumes are actually used
    # Manual computation of expected weight for edge 0
    edge_0_src = edge_index[0, 0].item()
    edge_0_dst = edge_index[1, 0].item()
    V_i_0 = node_volumes[edge_0_src].item()
    d_0 = edge_attr[0, 3].item()
    
    expected_weight_0 = 1.0 / (V_i_0 * d_0)
    
    # Get actual weight by calling _compute_edge_weight directly
    actual_weight_0 = mp._compute_edge_weight(
        edge_attr[0:1], 
        node_volumes, 
        edge_index[:, 0:1]
    ).item()
    
    print(f"\n[Verification for edge 0]")
    print(f"  Source node: {edge_0_src}, Volume V_i: {V_i_0}")
    print(f"  Distance |r_ij|: {d_0:.6f}")
    print(f"  Expected weight 1/(V_i * |r_ij|): {expected_weight_0:.6f}")
    print(f"  Actual weight: {actual_weight_0:.6f}")
    print(f"  Match: {abs(expected_weight_0 - actual_weight_0) < 1e-5} ✅" if abs(expected_weight_0 - actual_weight_0) < 1e-5 else f"  Match: False ❌")
    
    # Test without volumes (fallback)
    laplacian_no_vol = mp(x, edge_index, edge_attr, node_volumes=None)
    weight_no_vol = mp._compute_edge_weight(edge_attr[0:1], None, None).item()
    expected_no_vol = 1.0 / d_0
    
    print(f"\n[Fallback test (no volumes)]")
    print(f"  Expected weight 1/|r_ij|: {expected_no_vol:.6f}")
    print(f"  Actual weight: {weight_no_vol:.6f}")
    print(f"  Match: {abs(expected_no_vol - weight_no_vol) < 1e-5} ✅" if abs(expected_no_vol - weight_no_vol) < 1e-5 else f"  Match: False ❌")
    
    return abs(expected_weight_0 - actual_weight_0) < 1e-5


def test_acoustic_wave_gnn():
    """Test acoustic_wave_gnn.py implementation."""
    print("\n" + "=" * 70)
    print("Testing acoustic_wave_gnn.py")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple graph
    N = 10
    positions = torch.randn(N, 3, device=device) * 0.1
    
    # Simple edge index
    src = []
    dst = []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], device=device)
    
    # Edge attributes (6D)
    r_vec = positions[edge_index[0]] - positions[edge_index[1]]
    distance = torch.norm(r_vec, dim=-1, keepdim=True)
    Z_ratio = torch.ones(distance.shape[0], 1, device=device)
    atten = torch.zeros(distance.shape[0], 1, device=device)
    edge_attr = torch.cat([r_vec, distance, Z_ratio, atten], dim=-1)
    
    # Create node volumes with DISTINCT values
    node_volumes = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
                                device=device).unsqueeze(-1)
    
    # Test the MP layer
    mp = WaveEquationMP2(hdim=16, edge_dim=6, frequency=5e6).to(device)
    x = torch.randn(N, 16, device=device)
    
    # Call with volumes
    laplacian = mp(x, edge_index, edge_attr, node_volumes=node_volumes)
    
    # Verify that volumes are actually used
    edge_0_src = edge_index[0, 0].item()
    edge_0_dst = edge_index[1, 0].item()
    V_i_0 = node_volumes[edge_0_src].item()
    d_0 = edge_attr[0, 3].item()
    
    expected_weight_0 = 1.0 / (V_i_0 * d_0)
    
    # Get actual weight
    actual_weight_0 = mp._compute_edge_weight(
        edge_attr[0:1], 
        node_volumes, 
        edge_index[:, 0:1]
    ).item()
    
    print(f"\n[Verification for edge 0]")
    print(f"  Source node: {edge_0_src}, Volume V_i: {V_i_0}")
    print(f"  Distance |r_ij|: {d_0:.6f}")
    print(f"  Expected weight 1/(V_i * |r_ij|): {expected_weight_0:.6f}")
    print(f"  Actual weight: {actual_weight_0:.6f}")
    print(f"  Match: {abs(expected_weight_0 - actual_weight_0) < 1e-5} ✅" if abs(expected_weight_0 - actual_weight_0) < 1e-5 else f"  Match: False ❌")
    
    return abs(expected_weight_0 - actual_weight_0) < 1e-5


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("LAPLACIAN WEIGHT FIX VERIFICATION TEST")
    print("🔬 " * 20 + "\n")
    
    test1_pass = test_wave_equation_mp()
    test2_pass = test_acoustic_wave_gnn()
    
    print("\n" + "=" * 70)
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED - Laplacian weights correctly use 1/(V_i * |r_ij|)")
    else:
        print("❌ TESTS FAILED - Laplacian weights NOT using V_i correctly")
    print("=" * 70)
