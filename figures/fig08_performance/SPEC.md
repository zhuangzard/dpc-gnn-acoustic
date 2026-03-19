# Figure 8: Computational Performance Comparison

**Type:** Half-page (7.16 × 3.0 inches)
**Format:** SVG + PNG (300 DPI) + PDF

## Content Description
Two sub-figures showing speed advantage and scaling.

### Layout (side by side)

**(a) Speed Comparison** (log-scale horizontal bar chart)
| Method | Time per image | Type |
|--------|---------------|------|
| k-Wave OMP (CPU) | ~3.2s | Physics sim |
| k-Wave CUDA (GPU) | ~0.5s (est.) | Physics sim |
| j-Wave (JAX GPU) | ~1-2s (est.) | Differentiable sim |
| V3 PINN forward | ~0.7s | Neural + soft physics |
| **V4 forward (ours)** | **<100ms target** | Physics-as-Forward |
| V4 GNN only | ~5ms (est.) | GNN encoder only |
| Pix2Pix | ~50ms | Pure neural |

Annotate speedup factors: "30×+ vs k-Wave", "DAS vectorized 13×"

**(b) Scaling Analysis** (line plot)
- X: Grid resolution (128², 256², 512²)
- Y: Inference time (ms), log scale
- Lines: V4 forward, k-Wave, Pix2Pix
- Second Y axis: Memory (GB)

### Visual Style
- Log-scale where needed
- Color: our method in orange (#E67E22), k-Wave in blue (#1B3A5C), baselines in grey
- Speedup factors as bold annotations with arrows

### Caption
"Fig. 8. Computational performance. (a) Single-image inference time comparison (log scale). DPC-GNN-Acoustic V4 achieves XX× speedup over k-Wave while maintaining physics guarantees, unlike purely data-driven methods (Pix2Pix) which sacrifice physical fidelity for speed. The vectorized DAS beamformer provides a 13× speedup over the loop-based implementation. (b) Scaling behavior with grid resolution, showing V4 maintains favorable scaling characteristics."

## Generation Code Requirements
```python
# Save to: fig08_performance/code/benchmark_speed.py
# Run inference timing on GPU server
# Warm up + 100 iterations, report mean ± std
# Also profile k-Wave on same samples for direct comparison
```

## Data Files Needed
- `data/speed_benchmark.csv` — columns: method, resolution, time_ms_mean, time_ms_std, memory_gb
