# DPC-GNN-Acoustic Code Revisions

**Date:** 2026-03-18  
**Reviewer:** GLM Council (KimiCode)  
**Original Score:** 8/10 - Pass with Modifications  
**Status:** ✅ All Issues Fixed

---

## Summary of Changes

This document details all modifications made to the DPC-GNN-Acoustic codebase based on the GLM audit report findings.

---

## P2-1: Initial Conditions (HIGH PRIORITY) ✅ FIXED

### Problem
The original code incorrectly initialized pressure history for leapfrog integration:

```python
# ❌ WRONG: Simply cloning p0
p_prev = p0.clone()
p_curr = p0.clone()
```

This assumes zero initial velocity but fails to account for the Laplacian term in the Taylor expansion, causing numerical inaccuracy in the first few time steps.

### Solution
Implemented proper Taylor expansion initialization with Laplacian computation:

```python
# ✅ CORRECT: Taylor expansion with Laplacian
laplacian_0 = self.mp_layers[0](p0, edge_index, edge_attr)
c_squared = (c.mean() ** 2)
# Taylor expansion: p^{-1} = p^0 - 0.5*(c*dt)^2 * L(p^0) for v0=0
p_prev = p0 - 0.5 * (c_squared * self.dt**2) * laplacian_0
p_curr = p0.clone()
```

### Files Modified
- `src/models/acoustic_gnn.py`: `propagate()` method
- `src/models/wave_equation_mp.py`: `TimeStepping.initialize()` method with new signature

---

## P2-2: Physics Loss with Time Derivative (MEDIUM PRIORITY) ✅ FIXED

### Problem
The original physics loss only computed spatial Laplacian without the temporal derivative term:

```python
# ❌ WRONG: Missing time derivative
laplacian = self.mp_layers[-1](p, edge_index, edge_attr)
residual = c_squared * laplacian  # Not a complete wave equation residual
```

This violates the wave equation: ∂²p/∂t² = c²∇²p

### Solution
Implemented full physics loss with pressure history storage and central difference for time derivative:

```python
# ✅ CORRECT: Full wave equation residual
# Store pressure history during propagation
self.pressure_history = [p0.clone()]
for k in range(self.n_mp_layers):
    # ... time stepping ...
    self.pressure_history.append(p_curr.clone())

# Compute loss with time derivative
def compute_physics_loss(self):
    loss = 0
    for k in range(1, len(self.pressure_history)-1):
        p_prev, p_curr, p_next = self.pressure_history[k-1:k+2]
        # Central difference for second time derivative
        p_tt = (p_next - 2*p_curr + p_prev) / self.dt**2
        laplacian = self.mp_layers[k](p_curr, edge_index, edge_attr)
        residual = p_tt - c_squared * laplacian
        loss += (residual ** 2).mean()
    return loss
```

### Files Modified
- `src/models/acoustic_gnn.py`: 
  - Added `pressure_history` list storage
  - Added `_c_squared_physics` buffer
  - Added `edge_index_cache` and `edge_attr_cache`
  - Rewrote `compute_physics_loss()` method completely

---

## O1: TimeStepper Pre-computation (OPTIMIZATION) ✅ FIXED

### Problem
The original code recreated the TimeStepper object in every iteration:

```python
# ❌ INEFFICIENT: Creating TimeStepper every iteration
for k in range(self.n_mp_layers):
    c_field = c.mean() if c.numel() > 1 else c
    time_stepper = TimeStepping(self.dt, c_field)  # Created every loop!
    p_next = time_stepper.step(p_curr, p_prev, laplacian)
```

### Solution
Implemented caching mechanism to reuse TimeStepper objects:

```python
# ✅ EFFICIENT: Cache and reuse TimeStepper
def _get_or_create_time_stepper(self, c_field, device):
    c_key = f"{c_field.mean().item():.4f}_{str(device)}"
    if c_key not in self._time_stepper_cache:
        self._time_stepper_cache[c_key] = TimeStepping(self.dt, c_mean).to(device)
    return self._time_stepper_cache[c_key]

# In propagate():
time_stepper = self._get_or_create_time_stepper(c, device)  # Cached!
for k in range(self.n_mp_layers):
    p_next = time_stepper.step(p_curr, p_prev, laplacian)
```

### Files Modified
- `src/models/acoustic_gnn.py`: Added `_get_or_create_time_stepper()` method and `_time_stepper_cache` dictionary

---

## O2: Blood Tissue Properties (DATA) ✅ FIXED

### Problem
Missing blood tissue in acoustic properties, which is essential for medical ultrasound simulation.

### Solution
Added blood tissue to TISSUE_PROPERTIES:

```python
TISSUE_PROPERTIES = {
    'air':     (-2000, -900,    1.2,   343,   41.0,  2.0),
    'fat':     (-120,  -40,     916,   1450,  0.6,   1.0),
    'water':   (-20,   20,      1000,  1482,  0.002, 2.0),
    'blood':   (10,    50,      1025,  1570,  0.18,  1.0),  # ✅ ADDED
    'soft':    (20,    60,      1020,  1540,  0.5,   1.1),
    'liver':   (40,    80,      1050,  1570,  0.7,   1.1),
    'muscle':  (30,    100,     1050,  1580,  1.5,   1.0),
    'bone':    (200,   3000,    1908,  4080,  10.0,  1.0),
}
```

**Blood Properties (Reference Values @ 1 MHz):**
- Density: 1025 kg/m³
- Speed of sound: 1570 m/s
- Attenuation: 0.18 dB/cm
- Frequency power: 1.0

### Files Modified
- `src/physics/acoustic_properties.py`: Added 'blood' entry to TISSUE_PROPERTIES

---

## O3: CFL Check Returns Numerical Value (API) ✅ FIXED

### Problem
CFL check only returned boolean, making it hard to assess how close to stability limit:

```python
# ❌ OLD: Only returns boolean
def check_cfl_condition(...):
    return stable  # bool only
```

### Solution
Return both stability flag and CFL ratio:

```python
# ✅ NEW: Returns stability and ratio
def check_cfl_condition(...):
    cfl_ratio = dt / cfl_limit
    stable = cfl_ratio < 1.0
    return stable, cfl_ratio  # (bool, float)
```

**Usage:**
```python
stable, cfl_ratio = check_cfl_condition(dt, c_max, dx_min)
if cfl_ratio > 0.8:
    print("Warning: Close to CFL limit")
```

### Files Modified
- `src/models/wave_equation_mp.py`: Updated `check_cfl_condition()` return signature

---

## O4: Register Buffer Optimization (MEMORY) ✅ FIXED

### Problem
Using instance attributes instead of `register_buffer()` for non-trainable tensors that need device management.

### Solution
Used `register_buffer()` with `persistent=False` for temporary tensors:

```python
# ✅ OPTIMIZED: Using register_buffer for device-aware tensors
self.register_buffer('_c_squared_physics', c_squared, persistent=False)
```

This ensures:
1. Automatic device placement (CPU/GPU)
2. State dict exclusion (persistent=False)
3. Proper handling in model.to(device) calls

### Files Modified
- `src/models/acoustic_gnn.py`: Added `register_buffer` for `_c_squared_physics`

---

## Test Results

All self-tests pass after modifications:

```
wave_equation_mp.py — Self Test
✅ Test 1: WaveEquationMP forward pass successful
✅ Test 2: TimeStepping with Taylor expansion successful
✅ Test 3: Autograd working correctly
✅ Test 4: Multi-step propagation successful
✅ Test 5: Extended edge attributes working
✅ Test 6: CFL check working correctly

acoustic_gnn.py — Self Test
✅ Test 1: Model creation successful
✅ Test 2: Forward pass successful
✅ Test 3: Gradients flow through model
✅ Test 4: Physics loss with time derivative working
✅ Test 5: Acoustic properties computed
✅ Test 6: US renderer normalizes to [0, 1]
✅ Test 7: TimeStepper caching working correctly

acoustic_properties.py — Self Test
✅ Test 1: Basic mapping successful
✅ Test 2: Soft boundaries produce different results
✅ Test 3: Impedance ordering correct
✅ Test 4: Reflection coefficients reasonable
✅ Test 5: Attenuation increases with frequency
✅ Test 6: Batch processing successful
✅ Test 7: Gradients flow through mapper
```

---

## Files Changed

| File | Changes |
|------|---------|
| `src/models/wave_equation_mp.py` | TimeStepper initialization with Laplacian, CFL return tuple |
| `src/models/acoustic_gnn.py` | Initial conditions, physics loss, TimeStepper caching, buffers |
| `src/physics/acoustic_properties.py` | Added blood tissue properties |

---

## Backward Compatibility

**Breaking Changes:**
- `check_cfl_condition()` now returns `Tuple[bool, float]` instead of `bool`
- `compute_physics_loss()` no longer takes arguments (uses cached values)

**Migration:**
```python
# Old code
stable = check_cfl_condition(dt, c, dx)
physics_loss = model.compute_physics_loss(outputs, edge_index, edge_attr)

# New code
stable, cfl_ratio = check_cfl_condition(dt, c, dx)
physics_loss = model.compute_physics_loss()  # Call after forward()
```

---

## Verification

To verify the fixes, run:

```bash
cd /Users/taisenzhuang/workspace/DPC-GNN-Acoustic/src/models
python wave_equation_mp.py
python acoustic_gnn.py
python ../physics/acoustic_properties.py
```

All tests should pass with ✅ marks.
