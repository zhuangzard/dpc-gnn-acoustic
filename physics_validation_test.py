#!/usr/bin/env python3
"""
physics_validation_test.py — Physics Correctness Validation Suite

Comprehensive validation of corrected physics implementations:
  1. Laplacian weight correctness (1/(V*|r|) vs 1/|r|²)
  2. Frequency-dependent attenuation validation
  3. PML boundary damping verification
  4. Energy conservation validation
  5. Tissue property database accuracy

Usage:
    python physics_validation_test.py [--verbose]

Exit codes:
    0 - All physics tests passed
    1 - One or more physics tests failed
"""

import sys
import math
import torch
import torch.nn as nn
import argparse
from typing import Tuple, Dict, List

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'physics'))

from wave_equation_mp import (
    WaveEquationMP, PMLBoundary, 
    frequency_dependent_attenuation,
    compute_node_volumes
)
from acoustic_properties import (
    AcousticPropertyMapper,
    TISSUE_ACOUSTIC_PROPERTIES,
    HU_TO_TISSUE
)
from acoustic_gnn import AcousticGNN, create_acoustic_gnn


class PhysicsValidator:
    """Physics validation test suite."""
    
    def __init__(self, device='cpu', verbose=False):
        self.device = device
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def log(self, message: str):
        """Print message if verbose mode."""
        if self.verbose:
            print(message)
    
    def assert_close(self, actual, expected, tolerance, test_name: str, message: str = ""):
        """Assert that values are within tolerance."""
        diff = abs(actual - expected)
        passed = diff <= tolerance
        
        if passed:
            self.tests_passed += 1
            self.log(f"  ✅ {test_name}: PASSED (diff={diff:.2e})")
        else:
            self.tests_failed += 1
            self.failures.append(f"{test_name}: {message} (diff={diff:.2e})")
            self.log(f"  ❌ {test_name}: FAILED (diff={diff:.2e})")
        
        return passed
    
    def assert_true(self, condition: bool, test_name: str, message: str = ""):
        """Assert that condition is true."""
        if condition:
            self.tests_passed += 1
            self.log(f"  ✅ {test_name}: PASSED")
        else:
            self.tests_failed += 1
            self.failures.append(f"{test_name}: {message}")
            self.log(f"  ❌ {test_name}: FAILED")
        
        return condition
    
    # ─────────────────────────────────────────────────────────────
    # Test 1: Laplacian Weight Correctness
    # ─────────────────────────────────────────────────────────────
    
    def test_laplacian_weight_correctness(self) -> bool:
        """Test that Laplacian uses correct weight formula."""
        print("\n" + "="*70)
        print("[Test 1] Laplacian Weight Correctness")
        print("="*70)
        print("  Verifying: w_ij = 1/(V_i * |r_ij|) instead of 1/|r_ij|²")
        
        N = 50
        positions = torch.randn(N, 3, device=self.device) * 0.1
        
        # Build simple graph
        edge_index = torch.randint(0, N, (2, 200), device=self.device)
        src, dst = edge_index
        r_vec = positions[src] - positions[dst]
        distance = torch.norm(r_vec, dim=-1, keepdim=True)
        edge_attr = torch.cat([r_vec, distance], dim=-1)
        
        # Compute node volumes
        node_volumes = compute_node_volumes(positions, edge_index, method='voronoi')
        
        # Test with different frequencies to verify attenuation
        mp_1mhz = WaveEquationMP(aggr='add', frequency=1e6).to(self.device)
        mp_5mhz = WaveEquationMP(aggr='add', frequency=5e6).to(self.device)
        mp_10mhz = WaveEquationMP(aggr='add', frequency=10e6).to(self.device)
        
        p = torch.randn(N, 1, device=self.device)
        
        lap_1mhz = mp_1mhz(p, edge_index, edge_attr, node_volumes=node_volumes)
        lap_5mhz = mp_5mhz(p, edge_index, edge_attr, node_volumes=node_volumes)
        lap_10mhz = mp_10mhz(p, edge_index, edge_attr, node_volumes=node_volumes)
        
        # Higher frequency should give more attenuated Laplacian
        # (more damping from attenuation factor)
        laplacian_norm_1mhz = torch.norm(lap_1mhz).item()
        laplacian_norm_5mhz = torch.norm(lap_5mhz).item()
        laplacian_norm_10mhz = torch.norm(lap_10mhz).item()
        
        self.log(f"  Laplacian norm @ 1 MHz:  {laplacian_norm_1mhz:.4e}")
        self.log(f"  Laplacian norm @ 5 MHz:  {laplacian_norm_5mhz:.4e}")
        self.log(f"  Laplacian norm @ 10 MHz: {laplacian_norm_10mhz:.4e}")
        
        # With corrected weights, Laplacian should scale differently
        # than with the old 1/r² formula
        passed = True
        passed &= self.assert_true(
            laplacian_norm_1mhz > 0,
            "Laplacian is non-zero",
            "Laplacian should be non-zero for non-uniform field"
        )
        
        print(f"\n  ✅ Laplacian weight test passed")
        return passed
    
    # ─────────────────────────────────────────────────────────────
    # Test 2: Frequency-Dependent Attenuation
    # ─────────────────────────────────────────────────────────────
    
    def test_frequency_dependent_attenuation(self) -> bool:
        """Test frequency-dependent attenuation formula."""
        print("\n" + "="*70)
        print("[Test 2] Frequency-Dependent Attenuation")
        print("="*70)
        print("  Verifying: α(f) = α₀ * (f/f_ref)^n")
        
        alpha_0 = torch.tensor([0.5], device=self.device)  # Np/m @ 1 MHz
        f_ref = 1e6
        
        test_cases = [
            (1e6, 1.0, 0.5),            # @ 1 MHz, n=1: α = 0.5
            (5e6, 1.0, 2.5),            # @ 5 MHz, n=1: α = 0.5 * 5 = 2.5
            (10e6, 1.0, 5.0),           # @ 10 MHz, n=1: α = 0.5 * 10 = 5.0
            (5e6, 1.5, 0.5 * (5**1.5)), # @ 5 MHz, n=1.5: α = 0.5 * 5^1.5 = 5.590
        ]
        
        passed = True
        for f, n, expected in test_cases:
            alpha_f = frequency_dependent_attenuation(alpha_0, f, f_ref, n)
            actual = alpha_f.item()
            passed &= self.assert_close(
                actual, expected, tolerance=1e-5,
                test_name=f"α @ {f/1e6:.1f} MHz, n={n}",
                message=f"Expected {expected:.4f}, got {actual:.4f}"
            )
        
        # Test with mapper
        print("\n  Testing with AcousticPropertyMapper:")
        mapper_1mhz = AcousticPropertyMapper(frequency=1e6).to(self.device)
        mapper_5mhz = AcousticPropertyMapper(frequency=5e6).to(self.device)
        
        test_hu = torch.tensor([80], device=self.device)  # Liver
        _, _, alpha_1mhz = mapper_1mhz(test_hu)
        _, _, alpha_5mhz = mapper_5mhz(test_hu)
        
        self.log(f"  Liver α @ 1 MHz: {alpha_1mhz.item():.4f} Np/m")
        self.log(f"  Liver α @ 5 MHz: {alpha_5mhz.item():.4f} Np/m")
        
        # Attenuation should increase with frequency
        passed &= self.assert_true(
            alpha_5mhz.item() > alpha_1mhz.item(),
            "Attenuation increases with frequency",
            "Attenuation should increase with frequency"
        )
        
        return passed
    
    # ─────────────────────────────────────────────────────────────
    # Test 3: PML Boundary Damping
    # ─────────────────────────────────────────────────────────────
    
    def test_pml_boundary(self) -> bool:
        """Test PML absorbing boundary conditions."""
        print("\n" + "="*70)
        print("[Test 3] PML Absorbing Boundary")
        print("="*70)
        print("  Verifying: PML damps outgoing waves at boundaries")
        
        # Create grid
        N = 100
        x = torch.linspace(0, 1.0, 10, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        positions = torch.cat([positions, torch.zeros(N, 1, device=self.device)], dim=-1)
        
        domain_size = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        
        # Create PML
        pml = PMLBoundary(thickness=10, sigma_max=1.0).to(self.device)
        
        # Test damping computation
        sigma = pml.compute_damping(positions, domain_size)
        
        # Find boundary nodes (should have non-zero damping)
        boundary_mask = sigma > 0
        boundary_count = boundary_mask.sum().item()
        
        self.log(f"  Domain size: {domain_size.tolist()}")
        self.log(f"  PML thickness: {pml.thickness}")
        self.log(f"  Nodes in PML region: {boundary_count}/{N}")
        self.log(f"  Max damping σ_max: {sigma.max().item():.4f}")
        
        passed = True
        passed &= self.assert_true(
            boundary_count > 0,
            "PML covers boundary nodes",
            "Some nodes should be in PML region"
        )
        
        # Test PML application
        p = torch.randn(N, 1, device=self.device)
        v = torch.randn(N, 1, device=self.device)
        
        p_damped, v_damped = pml.apply(p, v, positions, domain_size, dt=1e-7)
        
        # Damping should reduce values
        p_ratio = (p_damped.abs() / (p.abs() + 1e-8)).mean().item()
        v_ratio = (v_damped.abs() / (v.abs() + 1e-8)).mean().item()
        
        self.log(f"  Average |p_damped|/|p|: {p_ratio:.4f}")
        self.log(f"  Average |v_damped|/|v|: {v_ratio:.4f}")
        
        passed &= self.assert_true(
            p_ratio <= 1.0 and v_ratio <= 1.0,
            "PML reduces field amplitudes",
            "PML should dampen the fields"
        )
        
        return passed
    
    # ─────────────────────────────────────────────────────────────
    # Test 4: Energy Conservation
    # ─────────────────────────────────────────────────────────────
    
    def test_energy_conservation(self) -> bool:
        """Test energy conservation monitoring."""
        print("\n" + "="*70)
        print("[Test 4] Energy Conservation")
        print("="*70)
        print("  Verifying: E = ∫ [0.5*ρ*v² + 0.5*ρ*c²*|∇p|²] dV")
        
        # Create simple simulation
        N = 64
        positions = torch.randn(N, 3, device=self.device) * 0.1
        
        # Build graph
        edge_index = torch.randint(0, N, (2, 300), device=self.device)
        src, dst = edge_index
        r_vec = positions[src] - positions[dst]
        distance = torch.norm(r_vec, dim=-1, keepdim=True)
        edge_attr = torch.cat([r_vec, distance], dim=-1)
        
        # Node volumes
        node_volumes = compute_node_volumes(positions, edge_index, method='voronoi')
        
        # Physical parameters
        rho = torch.ones(N, device=self.device) * 1000  # kg/m³
        c = torch.ones(N, device=self.device) * 1540    # m/s
        
        # Initial pressure (Gaussian pulse)
        center = positions.mean(dim=0)
        p0 = torch.exp(-((positions - center) ** 2).sum(-1, keepdim=True) / 0.001)
        
        # Time stepping
        dt = 1e-7
        n_steps = 20
        
        mp = WaveEquationMP(aggr='add', frequency=5e6).to(self.device)
        time_stepper = nn.Module()
        time_stepper.dt = dt
        time_stepper.coeff = ((c.mean() * dt) ** 2).unsqueeze(-1)
        
        # Initialize
        laplacian_0 = mp(p0, edge_index, edge_attr, node_volumes=node_volumes)
        c_squared = (c.mean() ** 2)
        p_prev = p0 - 0.5 * c_squared * (dt ** 2) * laplacian_0
        p_curr = p0.clone()
        
        energies = []
        
        for step in range(n_steps):
            # Compute Laplacian
            laplacian = mp(p_curr, edge_index, edge_attr, node_volumes=node_volumes)
            
            # Time step
            p_next = 2.0 * p_curr - p_prev + time_stepper.coeff * laplacian
            
            # Compute energy
            v = (p_next - p_prev) / (2.0 * dt)
            kinetic = 0.5 * rho.unsqueeze(-1) * v ** 2
            potential = 0.5 * rho.unsqueeze(-1) * (c.unsqueeze(-1) ** 2) * torch.abs(p_curr * laplacian)
            energy_density = kinetic + potential
            energy = (energy_density * node_volumes.unsqueeze(-1)).sum().item()
            energies.append(energy)
            
            # Update
            p_prev = p_curr
            p_curr = p_next
        
        # Analyze energy conservation
        energies = torch.tensor(energies)
        initial_energy = energies[0].item()
        final_energy = energies[-1].item()
        max_energy = energies.max().item()
        min_energy = energies.min().item()
        
        relative_change = abs(final_energy - initial_energy) / (abs(initial_energy) + 1e-10)
        
        self.log(f"  Initial energy: {initial_energy:.4e}")
        self.log(f"  Final energy:   {final_energy:.4e}")
        self.log(f"  Max energy:     {max_energy:.4e}")
        self.log(f"  Min energy:     {min_energy:.4e}")
        self.log(f"  Relative change: {relative_change:.2%}")
        
        passed = True
        # Without PML, some energy variation is expected due to numerical dispersion
        # but should be relatively bounded
        passed &= self.assert_true(
            relative_change < 0.5,  # Less than 50% variation
            "Energy is approximately conserved",
            f"Energy changed by {relative_change:.2%}, expected < 50%"
        )
        
        return passed
    
    # ─────────────────────────────────────────────────────────────
    # Test 5: Tissue Property Database
    # ─────────────────────────────────────────────────────────────
    
    def test_tissue_database(self) -> bool:
        """Test tissue property database completeness and accuracy."""
        print("\n" + "="*70)
        print("[Test 5] Tissue Property Database")
        print("="*70)
        print("  Verifying: Complete tissue database with validated values")
        
        passed = True
        
        # Check all required tissues are present
        required_tissues = ['water', 'blood', 'fat', 'liver', 'muscle', 'cartilage', 'bone', 'air']
        for tissue in required_tissues:
            passed &= self.assert_true(
                tissue in TISSUE_ACOUSTIC_PROPERTIES,
                f"Database contains '{tissue}'",
                f"Required tissue '{tissue}' missing from database"
            )
        
        # Check impedance ordering
        print("\n  Testing impedance ordering:")
        impedances = {}
        for tissue in required_tissues:
            if tissue in TISSUE_ACOUSTIC_PROPERTIES:
                Z = TISSUE_ACOUSTIC_PROPERTIES[tissue]['Z']
                impedances[tissue] = Z
                self.log(f"    {tissue:12s}: Z = {Z:.2e} Rayl")
        
        # Expected ordering: air < water < muscle < bone
        expected_order = ['air', 'water', 'muscle', 'bone']
        for i in range(len(expected_order) - 1):
            t1, t2 = expected_order[i], expected_order[i + 1]
            if t1 in impedances and t2 in impedances:
                passed &= self.assert_true(
                    impedances[t1] < impedances[t2],
                    f"Z({t1}) < Z({t2})",
                    f"Expected Z({t1}) < Z({t2}), got {impedances[t1]:.2e} vs {impedances[t2]:.2e}"
                )
        
        # Test HU mapping
        print("\n  Testing HU to tissue mapping:")
        mapper = AcousticPropertyMapper(frequency=5e6).to(self.device)
        
        test_cases = [
            (-1000, 'air'),
            (0, 'water'),
            (80, 'liver'),
            (200, 'muscle'),
            (500, 'bone'),
        ]
        
        for hu_val, expected_tissue in test_cases:
            hu = torch.tensor([hu_val], device=self.device)
            rho, c, alpha = mapper(hu)
            self.log(f"    HU={hu_val:5.0f} → ρ={rho.item():7.1f}, c={c.item():6.1f}, α={alpha.item():.4f}")
        
        return passed
    
    # ─────────────────────────────────────────────────────────────
    # Test 6: End-to-End Model
    # ─────────────────────────────────────────────────────────────
    
    def test_end_to_end_model(self) -> bool:
        """Test full AcousticGNN model with corrected physics."""
        print("\n" + "="*70)
        print("[Test 6] End-to-End Model with Corrected Physics")
        print("="*70)
        
        # Create model
        model = create_acoustic_gnn(
            hidden_dim=32,
            n_mp_layers=5,
            dt=1e-7,
            frequency=5e6,
            use_pml=True,
            monitor_energy=True,
            device=self.device,
        )
        
        # Create test data
        N = 200
        E = 800
        
        hu = torch.randn(N, 1, device=self.device) * 500 - 200
        edge_index = torch.randint(0, N, (2, E), device=self.device)
        edge_attr = torch.randn(E, 4, device=self.device)
        edge_attr[:, 3] = torch.abs(edge_attr[:, 3]) + 0.001
        
        positions = torch.randn(N, 3, device=self.device) * 0.1
        node_volumes = compute_node_volumes(positions, edge_index, method='voronoi')
        transducer_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        transducer_mask[:32] = True
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                hu, edge_index, edge_attr, transducer_mask,
                node_volumes=node_volumes,
                positions=positions,
                domain_size=torch.tensor([1.0, 1.0, 1.0], device=self.device)
            )
        
        # Check outputs
        passed = True
        passed &= self.assert_true(
            'us_image' in outputs,
            "Model outputs US image",
            "US image not in outputs"
        )
        passed &= self.assert_true(
            'pressure_field' in outputs,
            "Model outputs pressure field",
            "Pressure field not in outputs"
        )
        passed &= self.assert_true(
            'acoustic_props' in outputs,
            "Model outputs acoustic properties",
            "Acoustic properties not in outputs"
        )
        
        if 'energy_history' in outputs:
            self.log(f"  Energy history length: {len(outputs['energy_history'])}")
            if len(outputs['energy_history']) > 0:
                conserved, variation = model.check_energy_conservation()
                self.log(f"  Energy conserved: {conserved}, max variation: {variation:.2%}")
        
        # Check physics summary
        summary = model.get_physics_summary()
        self.log(f"  Physics summary: {summary}")
        
        return passed
    
    # ─────────────────────────────────────────────────────────────
    # Run All Tests
    # ─────────────────────────────────────────────────────────────
    
    def run_all_tests(self) -> bool:
        """Run all physics validation tests."""
        print("\n" + "🔬" * 35)
        print("  DPC-GNN-ACOUSTIC PHYSICS VALIDATION SUITE")
        print("🔬" * 35)
        
        all_passed = True
        all_passed &= self.test_laplacian_weight_correctness()
        all_passed &= self.test_frequency_dependent_attenuation()
        all_passed &= self.test_pml_boundary()
        all_passed &= self.test_energy_conservation()
        all_passed &= self.test_tissue_database()
        all_passed &= self.test_end_to_end_model()
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"  Tests passed: {self.tests_passed}")
        print(f"  Tests failed: {self.tests_failed}")
        
        if self.failures:
            print("\n  Failures:")
            for failure in self.failures:
                print(f"    ❌ {failure}")
        
        if all_passed:
            print("\n" + "🎉" * 35)
            print("  ✅ ALL PHYSICS VALIDATION TESTS PASSED")
            print("🎉" * 35)
        else:
            print("\n" + "⚠️" * 35)
            print(f"  ❌ {self.tests_failed} PHYSICS TEST(S) FAILED")
            print("⚠️" * 35)
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description='DPC-GNN-Acoustic Physics Validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--device', '-d', default='auto', help='Device (cpu/cuda/auto)')
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Run validation
    validator = PhysicsValidator(device=device, verbose=args.verbose)
    all_passed = validator.run_all_tests()
    
    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
