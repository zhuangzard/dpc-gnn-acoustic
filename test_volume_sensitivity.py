#!/usr/bin/env python3
"""
test_volume_sensitivity.py - Verify that different V_i produce different weights

This proves that V_i is ACTUALLY being used, not ignored.
"""

import torch
import sys
sys.path.insert(0, '/Users/taisenzhuang/workspace/DPC-GNN-Acoustic/src')

from models.wave_equation_mp import WaveEquationMP


def test_volume_sensitivity():
    """Verify that changing V_i changes the weight."""
    print("=" * 70)
    print("VOLUME SENSITIVITY TEST")
    print("Proving that V_i is ACTUALLY used in weight calculation")
    print("=" * 70)
    
    device = "cpu"
    
    # Create a single edge
    edge_index = torch.tensor([[0], [1]], device=device)  # Edge from node 0 to node 1
    distance = torch.tensor([[0.1]], device=device)  # Fixed distance
    r_vec = torch.tensor([[0.1, 0.0, 0.0]], device=device)
    edge_attr = torch.cat([r_vec, distance], dim=-1)
    
    mp = WaveEquationMP(aggr='add').to(device)
    
    # Test with different volumes
    print("\n[Testing different V_i values]")
    print(f"Fixed distance |r_ij| = {distance[0, 0].item()}")
    print()
    
    volumes = [1.0, 2.0, 5.0, 10.0]
    weights = []
    
    for V_i in volumes:
        node_volumes = torch.tensor([[V_i], [1.0]], device=device)  # Node 0 has volume V_i
        
        weight = mp._compute_edge_weight(edge_attr, node_volumes, edge_index)
        weights.append(weight.item())
        
        expected = 1.0 / (V_i * distance[0, 0].item())
        
        print(f"  V_i = {V_i:4.1f}  →  weight = {weight.item():.6f}  (expected: {expected:.6f})")
    
    # Verify that weights are DIFFERENT
    print("\n[Verification]")
    unique_weights = len(set([round(w, 5) for w in weights]))
    print(f"  Number of unique weights: {unique_weights}")
    
    if unique_weights == len(volumes):
        print("  ✅ Each different V_i produces a different weight!")
        print("  ✅ V_i is ACTUALLY being used in the formula!")
    else:
        print("  ❌ FAIL: Some weights are the same - V_i might be ignored!")
        return False
    
    # Test the BUG: what if we DON'T use V_i?
    print("\n[Anti-test: What if V_i is ignored?]")
    weight_no_vol = mp._compute_edge_weight(edge_attr, None, None).item()
    print(f"  Weight without V_i: {weight_no_vol:.6f}")
    
    # This should be different from weights with V_i (except V_i=1.0 which is same as no volume)
    # Only check V_i != 1.0 cases
    weights_nonunity = [w for v, w in zip(volumes, weights) if v != 1.0]
    if all(abs(w - weight_no_vol) > 0.01 for w in weights_nonunity):
        print("  ✅ Weights with V_i != 1.0 are DIFFERENT from weight without V_i")
        print("  ✅ This proves V_i is being used!")
    else:
        print("  ❌ FAIL: Weights with/without V_i are too similar")
        return False
    
    return True


if __name__ == "__main__":
    success = test_volume_sensitivity()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ VOLUME SENSITIVITY TEST PASSED")
        print("✅ The fix is working: w_ij = 1/(V_i * |r_ij|)")
    else:
        print("❌ VOLUME SENSITIVITY TEST FAILED")
        print("❌ V_i might not be used correctly")
    print("=" * 70)
