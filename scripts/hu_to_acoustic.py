#!/usr/bin/env python3
"""
HU to Acoustic Property Mapping for Ultrasound Simulation

Based on published literature values with smooth, differentiable transitions.

Key References:
1. Duck, F.A. (1990). "Physical Properties of Tissue: A Comprehensive Reference Book"
2. ICRU Report 61 (1998). "Tissue Substitutes, Phantoms and Computational Modelling"
3. Mast, T.D. (2000). "Empirical relationships between acoustic parameters in human tissue"
4. Szabo, T.L. (2004). "Diagnostic Ultrasound Imaging: Inside Out"
5. Parker, K.J. et al. (2018). "Imaging ultrasound yields maps of tissue mechanical properties"
6. Bilaniuk, N. & Wong, G.S.K. (1993). "Speed of sound in pure water" — c(37°C) = 1524 m/s
7. Pichardo, S. et al. (2011). "Multi-frequency characterization of bone" — α₀ ≈ 3-6 dB/cm/MHz

TRUSTED Dataset: 48 kidney patients, HU range [-1024, 2213]

NOTE on alpha_power: The attenuation model is α = α₀ · f^n where:
  - Soft tissue: n ≈ 1.0-1.3 (near-linear frequency dependence)
  - Bone: n ≈ 1.0-1.5 (can be super-linear)
  α₀ values in this file are in [dB/cm/MHz] (i.e., the coefficient at 1 MHz).
"""

import numpy as np
from typing import Tuple, Dict


# =============================================================================
# Literature-Based Anchor Points: HU → (c, ρ, α)
# =============================================================================
# Format: HU value → (speed of sound [m/s], density [kg/m³], attenuation [dB/cm/MHz])
#
# Sources:
# - Air: c=353 m/s (at 37°C), ρ=1.2 kg/m³, α≈0 (Duck 1990)
# - Lung: c≈650 m/s, ρ≈400 kg/m³, α≈4.0 dB/cm/MHz (Mast 2000)
# - Fat: c=1450 m/s, ρ=950 kg/m³, α=0.6 dB/cm/MHz (Duck 1990)
# - Water: c=1524 m/s (at 37°C), ρ=1000 kg/m³, α=0.002 dB/cm/MHz (Bilaniuk 1993)
# - Kidney: c=1560 m/s, ρ=1050 kg/m³, α=1.0 dB/cm/MHz (Duck 1990)
# - Muscle: c=1580 m/s, ρ=1041 kg/m³, α=1.09 dB/cm/MHz (Duck 1990)
# - Liver: c=1578 m/s, ρ=1060 kg/m³, α=0.5 dB/cm/MHz (Duck 1990)
# - Blood: c=1584 m/s, ρ=1060 kg/m³, α=0.2 dB/cm/MHz (Duck 1990)
# - Bone (cortical): c=2800-4080 m/s, ρ=1900 kg/m³, α=3-6 dB/cm/MHz (Pichardo 2011)
# =============================================================================

HU_ANCHOR_POINTS: Dict[int, Tuple[float, float, float]] = {
    # HU → (c [m/s], ρ [kg/m³], α [dB/cm/MHz])
    -1024: (353.0,    1.2,    0.0),       # Air (pure, CT minimum) — 37°C
    -950:  (500.0,    50.0,   2.0),       # Emphysematous lung
    -800:  (650.0,    400.0,  4.0),       # Normal lung tissue
    -600:  (900.0,    600.0,  3.0),       # Aerate lung transition
    -100:  (1450.0,   950.0,  0.6),       # Subcutaneous fat
    -50:   (1450.0,   950.0,  0.6),       # Intra-abdominal fat
    0:     (1524.0,   1000.0, 0.002),     # Water/cerebrospinal fluid — 37°C (Bilaniuk 1993)
    20:    (1520.0,   1020.0, 0.4),       # Edema/inflammation
    30:    (1560.0,   1050.0, 1.0),       # Kidney cortex (reference point)
    40:    (1580.0,   1041.0, 1.09),      # Skeletal muscle
    # NOTE: Soft tissue range HU 20-80 is within measurement uncertainty (±10-20 m/s)
    50:    (1580.0,   1055.0, 0.8),       # Spleen (monotonic with muscle)
    60:    (1578.0,   1060.0, 0.5),       # Liver parenchyma
    70:    (1580.0,   1058.0, 0.6),       # Pancreas
    80:    (1584.0,   1060.0, 0.2),       # Blood (venous)
    100:   (1585.0,   1060.0, 0.2),       # Blood (arterial)
    150:   (1700.0,   1200.0, 2.0),       # Early calcification/fibrosis
    300:   (2000.0,   1500.0, 5.0),       # Soft bone / osteoid
    500:   (2500.0,   1700.0, 8.0),       # Trabecular bone
    1000:  (3000.0,   1850.0, 4.0),       # Cortical bone (low density) — Pichardo 2011
    1500:  (3500.0,   1950.0, 5.0),       # Cortical bone (medium density)
    2000:  (4000.0,   2000.0, 6.0),       # Cortical bone (high density)
    2213:  (4080.0,   2000.0, 6.5),       # Dense cortical bone (TRUSTED max)
}


def smooth_step(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """
    Smooth Hermite interpolation (differentiable cubic).
    
    Returns values in [0, 1] with smooth transition from 0 to 1 between x0 and x1.
    Outside [x0, x1], clamps to 0 or 1.
    
    This is the key to making the entire mapping differentiable.
    """
    # Normalize to [0, 1] range
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    # Smoothstep: 3t² - 2t³ (Hermite interpolation)
    return t * t * (3.0 - 2.0 * t)


def hu_to_acoustic(
    hu_slice: np.ndarray,
    return_confidence: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert CT Hounsfield Units to acoustic properties for ultrasound simulation.
    
    Parameters
    ----------
    hu_slice : np.ndarray
        2D or 3D array of Hounsfield Unit values from CT scan.
        Typical range: [-1024, 3000] but handles wider ranges.
    
    return_confidence : bool, optional
        If True, also return confidence map based on proximity to anchor points.
    
    Returns
    -------
    c_map : np.ndarray
        Speed of sound [m/s]. Range: ~343 (air) to ~4080 (dense bone)
    
    rho_map : np.ndarray
        Density [kg/m³]. Range: ~1.2 (air) to ~2000 (dense bone)
    
    alpha_map : np.ndarray
        Attenuation coefficient [dB/cm/MHz]. Range: ~0 (air/water) to ~6.5 (bone)
    
    (Optional) confidence_map : np.ndarray
        Confidence values [0, 1] based on distance to anchor points.
        Higher values = closer to known tissue types.
    
    Notes
    -----
    The mapping uses piecewise-linear interpolation with smooth Hermite
    transitions at anchor points. This ensures:
    
    1. Physical accuracy: All values based on published measurements
    2. Differentiability: Smooth transitions (C¹ continuous) for gradient-based methods
    3. No hard thresholds: Avoids discontinuities that break optimization
    
    Literature Sources:
    - Duck (1990): Comprehensive tissue acoustic properties
    - ICRU 61 (1998): Standardized tissue phantom values
    - Mast (2000): Empirical relationships and measurement data
    """
    # Ensure float for smooth interpolation
    hu = np.asarray(hu_slice, dtype=np.float64)

    # NaN/Inf guard: masked CT regions contain NaN → treat as air
    nan_mask = ~np.isfinite(hu)
    hu = np.where(nan_mask, -1024.0, hu)

    # Get sorted anchor points
    hu_values = np.array(sorted(HU_ANCHOR_POINTS.keys()))
    
    # Initialize output arrays
    c_map = np.zeros_like(hu)
    rho_map = np.zeros_like(hu)
    alpha_map = np.zeros_like(hu)
    
    # =========================================================================
    # Piecewise-linear interpolation with smooth blending
    # =========================================================================
    # For each interval between anchor points, compute linear interpolation
    # and blend smoothly at boundaries.
    # =========================================================================
    
    for i in range(len(hu_values) - 1):
        hu_low = hu_values[i]
        hu_high = hu_values[i + 1]
        
        c_low, rho_low, alpha_low = HU_ANCHOR_POINTS[hu_low]
        c_high, rho_high, alpha_high = HU_ANCHOR_POINTS[hu_high]
        
        # Smooth blending weight for this interval
        # t = 0 at hu_low, t = 1 at hu_high, smooth in between
        t = smooth_step(hu, hu_low, hu_high)
        
        # Create mask for values in this interval
        # Include upper bound for last interval to handle max HU value
        is_last_interval = (i == len(hu_values) - 2)
        if is_last_interval:
            in_interval = (hu >= hu_low) & (hu <= hu_high)
        else:
            in_interval = (hu >= hu_low) & (hu < hu_high)
        
        # Linear interpolation within interval
        c_interp = c_low + t * (c_high - c_low)
        rho_interp = rho_low + t * (rho_high - rho_low)
        alpha_interp = alpha_low + t * (alpha_high - alpha_low)
        
        # Apply to output
        c_map = np.where(in_interval, c_interp, c_map)
        rho_map = np.where(in_interval, rho_interp, rho_map)
        alpha_map = np.where(in_interval, alpha_interp, alpha_map)
    
    # Handle edge cases (HU outside anchor range)
    # Below minimum (shouldn't happen with CT, but safe)
    c_map = np.where(hu < -1024, HU_ANCHOR_POINTS[-1024][0], c_map)
    rho_map = np.where(hu < -1024, HU_ANCHOR_POINTS[-1024][1], rho_map)
    alpha_map = np.where(hu < -1024, HU_ANCHOR_POINTS[-1024][2], alpha_map)
    
    # Above maximum (dense bone, metal artifacts)
    # Extrapolate gently - cap at reasonable maximum
    max_hu = hu_values[-1]
    c_map = np.where(hu > max_hu, 4080.0, c_map)
    rho_map = np.where(hu > max_hu, 2000.0, rho_map)
    alpha_map = np.where(hu > max_hu, 6.5, alpha_map)
    
    # =========================================================================
    # Optional: Confidence map based on distance to anchor points
    # =========================================================================
    if return_confidence:
        confidence = np.ones_like(hu)
        
        # Lower confidence for values far from anchor points
        for i in range(len(hu_values) - 1):
            hu_low = hu_values[i]
            hu_high = hu_values[i + 1]
            mid = (hu_low + hu_high) / 2.0
            
            # Distance from nearest anchor (normalized by interval width)
            interval_width = hu_high - hu_low
            dist_from_anchor = np.minimum(
                np.abs(hu - hu_low),
                np.abs(hu - hu_high)
            )
            # Confidence is high near anchors, lower in the middle
            local_conf = 1.0 - 0.3 * (dist_from_anchor / (interval_width / 2.0))
            
            in_interval = (hu >= hu_low) & (hu < hu_high)
            confidence = np.where(in_interval, local_conf, confidence)
        
        return c_map, rho_map, alpha_map, confidence
    
    return c_map, rho_map, alpha_map


def db_per_cm_mhz_to_np_per_m(alpha_db: float, freq_mhz: float = 1.0,
                               alpha_power: float = 1.0) -> float:
    """
    Convert attenuation from dB/cm/MHz to Np/m (k-Wave native unit).

    Parameters
    ----------
    alpha_db : float or np.ndarray
        Attenuation in [dB/cm/MHz].
    freq_mhz : float
        Frequency in MHz (default 1.0, i.e. the α₀ coefficient).
    alpha_power : float
        Power-law exponent n in α = α₀·f^n.
        Soft tissue: ~1.0-1.3, bone: ~1.0-1.5.

    Returns
    -------
    float or np.ndarray
        Attenuation in [Np/m].

    Notes
    -----
    Conversion: 1 Np = 20/ln(10) dB ≈ 8.686 dB
    1 dB/cm = 100 dB/m → divide by 8.686 → Np/m
    α_Np/m = α_dB/cm/MHz × f^n × 100 / 8.686
    """
    return alpha_db * (freq_mhz ** alpha_power) * 100.0 / 8.686


def get_tissue_label(hu_value: float) -> str:
    """
    Get human-readable tissue label for a given HU value.
    
    Useful for debugging and visualization.
    """
    if hu_value < -900:
        return "Air"
    elif hu_value < -500:
        return "Lung"
    elif hu_value < -50:
        return "Fat"
    elif hu_value < 10:
        return "Water/Fluid"
    elif hu_value < 25:
        return "Edema"
    elif hu_value < 35:
        return "Kidney"
    elif hu_value < 45:
        return "Muscle"
    elif hu_value < 65:
        return "Liver/Spleen"
    elif hu_value < 110:
        return "Blood"
    elif hu_value < 200:
        return "Soft Tissue (dense)"
    elif hu_value < 400:
        return "Soft Bone/Calcification"
    elif hu_value < 800:
        return "Trabecular Bone"
    else:
        return "Cortical Bone"


# =============================================================================
# Validation and Testing
# =============================================================================

def validate_mapping():
    """
    Validate the mapping against known tissue values.
    
    Prints a table of HU values and corresponding acoustic properties.
    """
    print("=" * 70)
    print("HU → Acoustic Property Mapping Validation")
    print("=" * 70)
    print(f"{'HU':>8} {'Tissue':<20} {'c [m/s]':>10} {'ρ [kg/m³]':>12} {'α [dB/cm/MHz]':>15}")
    print("-" * 70)
    
    test_hu_values = [-1024, -800, -100, 0, 30, 40, 60, 80, 300, 1000, 2000]
    
    for hu in test_hu_values:
        # Create 1x1 array for testing
        hu_arr = np.array([[hu]])
        c, rho, alpha = hu_to_acoustic(hu_arr)
        
        tissue = get_tissue_label(hu)
        print(f"{hu:>8} {tissue:<20} {c[0,0]:>10.1f} {rho[0,0]:>12.1f} {alpha[0,0]:>15.3f}")
    
    print("=" * 70)
    print("\nLiterature References:")
    print("  - Duck, F.A. (1990). Physical Properties of Tissue")
    print("  - ICRU Report 61 (1998). Tissue Substitutes")
    print("  - Mast, T.D. (2000). Empirical acoustic relationships")


def plot_mapping_curves():
    """
    Generate visualization of HU → acoustic property curves.
    
    Requires matplotlib. Run as script to see plots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    # Generate HU range
    hu_range = np.linspace(-1024, 2213, 1000)
    hu_2d = hu_range.reshape(1, -1)
    
    # Get acoustic properties
    c, rho, alpha = hu_to_acoustic(hu_2d)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Speed of sound
    axes[0].plot(hu_range, c.flatten(), 'b-', linewidth=2)
    axes[0].set_ylabel('Speed of Sound [m/s]', fontsize=12)
    axes[0].set_title('CT Hounsfield Units → Acoustic Properties', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 4500])
    
    # Mark key tissue types
    tissue_marks = [
        (-1000, 'Air'),
        (-100, 'Fat'),
        (0, 'Water'),
        (30, 'Kidney'),
        (60, 'Liver'),
        (80, 'Blood'),
        (1000, 'Bone'),
    ]
    for hu, label in tissue_marks:
        axes[0].axvline(hu, color='gray', linestyle='--', alpha=0.5)
        axes[0].text(hu, axes[0].get_ylim()[1] * 0.95, label, 
                    rotation=90, va='top', ha='right', fontsize=9)
    
    # Density
    axes[1].plot(hu_range, rho.flatten(), 'g-', linewidth=2)
    axes[1].set_ylabel('Density [kg/m³]', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 2200])
    
    # Attenuation
    axes[2].plot(hu_range, alpha.flatten(), 'r-', linewidth=2)
    axes[2].set_ylabel('Attenuation [dB/cm/MHz]', fontsize=12)
    axes[2].set_xlabel('Hounsfield Units [HU]', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hu_to_acoustic_mapping.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: hu_to_acoustic_mapping.png")
    plt.show()


# =============================================================================
# Advanced: Differentiable Soft Segmentation
# =============================================================================

def soft_tissue_segmentation(hu: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute soft (probabilistic) tissue segmentation from HU values.
    
    Returns probability maps for each tissue type. All values in [0, 1].
    Sum of probabilities at each pixel may not equal 1 (overlapping distributions).
    
    This is useful for:
    - Uncertainty quantification
    - Soft blending of acoustic properties
    - Differentiable tissue classification
    
    Parameters
    ----------
    hu : np.ndarray
        Hounsfield Unit values
    
    Returns
    -------
    dict[str, np.ndarray]
        Probability maps for each tissue type:
        - 'air': HU < -900
        - 'lung': -900 to -500
        - 'fat': -100 to -50
        - 'water': -10 to 10
        - 'kidney': 20 to 40
        - 'muscle': 35 to 55
        - 'liver': 50 to 70
        - 'blood': 70 to 110
        - 'bone': > 200
    """
    # Define tissue HU ranges: (center, width)
    # Using Gaussian-like soft membership functions
    tissue_params = {
        'air':    (-1024, 100),
        'lung':   (-700, 200),
        'fat':    (-75, 25),
        'water':  (0, 10),
        'kidney': (30, 10),
        'muscle': (40, 10),
        'liver':  (60, 10),
        'blood':  (80, 20),
        'bone':   (1500, 500),
    }
    
    probs = {}
    for tissue, (center, width) in tissue_params.items():
        # Gaussian membership function
        probs[tissue] = np.exp(-0.5 * ((hu - center) / width) ** 2)
    
    return probs


def hu_to_acoustic_with_uncertainty(
    hu_slice: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert HU to acoustic properties with uncertainty quantification.
    
    Uses soft tissue segmentation to compute weighted average of properties
    from neighboring tissue types, providing uncertainty estimates.
    
    Returns
    -------
    c_map, rho_map, alpha_map : np.ndarray
        Acoustic properties
    
    uncertainty_map : np.ndarray
        Relative uncertainty [0, 1] based on tissue mixture.
        Higher values indicate HU values at tissue boundaries.
    """
    # Get base mapping
    c, rho, alpha = hu_to_acoustic(hu_slice)
    
    # Get soft tissue probabilities
    probs = soft_tissue_segmentation(hu_slice)
    
    # Compute entropy-based uncertainty
    # High uncertainty when multiple tissues have similar probabilities
    total_prob = sum(probs.values())
    entropy = np.zeros_like(hu_slice, dtype=np.float64)
    
    for p in probs.values():
        # Normalize and compute entropy
        p_norm = p / (total_prob + 1e-10)
        entropy -= p_norm * np.log(p_norm + 1e-10)
    
    # Normalize uncertainty to [0, 1]
    max_entropy = np.log(len(probs))  # Maximum possible entropy
    uncertainty = entropy / max_entropy
    
    return c, rho, alpha, uncertainty


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("HU → Acoustic Property Mapping")
    print("Based on published literature values\n")
    
    # Run validation
    validate_mapping()
    
    print("\n" + "=" * 70)
    print("Testing differentiability (gradient check)...")
    print("=" * 70)
    
    # Test gradient computation
    hu_test = np.linspace(-100, 100, 50).reshape(1, -1)
    c, rho, alpha = hu_to_acoustic(hu_test)
    
    # Compute numerical gradients
    dc_dhu = np.gradient(c.flatten())
    drho_dhu = np.gradient(rho.flatten())
    dalpha_dhu = np.gradient(alpha.flatten())
    
    print(f"Speed of sound gradient range: [{dc_dhu.min():.3f}, {dc_dhu.max():.3f}]")
    print(f"Density gradient range: [{drho_dhu.min():.3f}, {drho_dhu.max():.3f}]")
    print(f"Attenuation gradient range: [{dalpha_dhu.min():.6f}, {dalpha_dhu.max():.6f}]")
    
    # Check for discontinuities (gradient spikes)
    gradient_threshold = 10.0  # m/s per HU
    c_smooth = np.all(np.abs(dc_dhu) < gradient_threshold)
    print(f"\nSmooth transitions (|dc/dHU| < {gradient_threshold}): {c_smooth} ✓" if c_smooth else f"\nWARNING: Potential discontinuities detected")
    
    print("\nGenerating visualization...")
    plot_mapping_curves()
