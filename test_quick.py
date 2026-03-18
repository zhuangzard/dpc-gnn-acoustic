#!/usr/bin/env python3
"""
test_quick.py — Quick sanity check for refactored code
"""

import sys
import torch

# Test imports
print("=" * 60)
print("Testing Refactored DPC-GNN-Acoustic")
print("=" * 60)

# Test 1: Import
print("\n✓ Test 1: Import modules")
try:
    sys.path.insert(0, 'src')
    from models.acoustic_wave_gnn import AcousticWaveGNN, WaveEquationMP
    print("  ✅ Imports successful")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Model creation
print("\n✓ Test 2: Create AcousticWaveGNN")
try:
    model = AcousticWaveGNN(hdim=32, n_layers=4, node_dim=4, edge_dim=6)
    params = model.count_params()
    print(f"  ✅ Model created with {params:,} parameters")
except Exception as e:
    print(f"  ❌ Model creation failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n✓ Test 3: Forward pass")
try:
    N = 50
    nf = torch.randn(N, 4)
    ei = torch.randint(0, N, (2, 200))
    ea = torch.randn(200, 6)
    ea[:, 3] = ea[:, 3].abs() + 0.001  # positive distance
    dt = torch.tensor(1e-7)
    c = torch.randn(N, 1).abs() * 1000 + 1000
    
    p = model(nf, ei, ea, dt, c)
    print(f"  ✅ Forward pass successful: input {nf.shape} → output {p.shape}")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Gradient flow
print("\n✓ Test 4: Gradient flow")
try:
    nf_var = nf.clone().requires_grad_(True)
    p = model(nf_var, ei, ea, dt, c)
    loss = p.sum()
    loss.backward()
    assert nf_var.grad is not None
    print(f"  ✅ Gradient computed: norm = {nf_var.grad.norm().item():.4e}")
except Exception as e:
    print(f"  ❌ Gradient flow failed: {e}")
    sys.exit(1)

# Test 5: WaveEquationMP
print("\n✓ Test 5: WaveEquationMP standalone")
try:
    mp = WaveEquationMP(hdim=16, edge_dim=6)
    x = torch.randn(N, 16)
    laplacian = mp(x, ei, ea)
    print(f"  ✅ Laplacian computed: {laplacian.shape}")
except Exception as e:
    print(f"  ❌ WaveEquationMP failed: {e}")
    sys.exit(1)

# Test 6: Training script imports
print("\n✓ Test 6: Training script imports")
try:
    # Don't actually run training, just test imports
    import train_acoustic
    print(f"  ✅ train_acoustic.py imports successful")
    print(f"  ✅ Found {len(train_acoustic.MEDIA)} media: {list(train_acoustic.MEDIA.keys())}")
except Exception as e:
    print(f"  ❌ Training script import failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nRefactored code is working correctly!")
print("\nNext steps:")
print("  1. Run full training: python3 train_acoustic.py --medium liver --epochs 10")
print("  2. Train all media: python3 train_acoustic.py --epochs 10")
print("  3. Check results in /root/results/acoustic_*/")
print("\nFor detailed style comparison, see: docs/STYLE_COMPARISON.md")
print("=" * 60)
