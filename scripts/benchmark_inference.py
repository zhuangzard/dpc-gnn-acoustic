#!/usr/bin/env python3
"""
DPC-GNN-Acoustic V4: Inference Speed Benchmark

Measures:
  - Total inference time (mean ± std over 100 runs, 10 warmup)
  - Component breakdown: GNN Encoder | Leapfrog Propagator | DAS Beamformer
  - k-Wave OMP reference time (subprocess call)

Usage:
    python scripts/benchmark_inference.py --checkpoint checkpoints_v4/best.pt
    python scripts/benchmark_inference.py --checkpoint checkpoints_v4/best.pt --n_runs 200 --warmup 20
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dpc_gnn_acoustic_v4 import DPCGNNAcousticV4


# ---------------------------------------------------------------------------
# Patched model with timing hooks
# ---------------------------------------------------------------------------
class TimedDPCGNNAcousticV4(DPCGNNAcousticV4):
    """Wraps V4 model with per-component timing."""

    def forward_timed(self, ct: torch.Tensor):
        """
        Returns (outputs_dict, timing_dict_ms).
        All GPU syncs are explicit for accurate timing.
        """
        torch.cuda.synchronize()

        # 1. GNN Encoder
        t0 = time.perf_counter()
        c, alpha, sigma = self.encoder(ct)
        torch.cuda.synchronize()
        t_encoder = (time.perf_counter() - t0) * 1000.0

        # 2. Source generation
        source = self._default_source(ct.device, ct.size(0))

        # 3. Leapfrog Propagation
        t1 = time.perf_counter()
        sensor_data = self.propagator(c, alpha, sigma, source)
        torch.cuda.synchronize()
        t_leapfrog = (time.perf_counter() - t1) * 1000.0

        # 4. DAS Beamforming
        t2 = time.perf_counter()
        bmode = self.beamformer(sensor_data)
        torch.cuda.synchronize()
        t_beamform = (time.perf_counter() - t2) * 1000.0

        t_total = t_encoder + t_leapfrog + t_beamform

        outputs = {
            'bmode': bmode, 'c': c, 'alpha': alpha,
            'sigma': sigma, 'sensor_data': sensor_data,
        }
        timing = {
            'encoder_ms': t_encoder,
            'leapfrog_ms': t_leapfrog,
            'beamform_ms': t_beamform,
            'total_ms': t_total,
        }
        return outputs, timing


# ---------------------------------------------------------------------------
# k-Wave OMP benchmark (optional)
# ---------------------------------------------------------------------------
def benchmark_kwave_omp(n_runs: int = 5) -> dict:
    """
    Benchmark k-Wave OMP simulation via MATLAB/Octave subprocess.
    Returns timing stats or None if k-Wave is not available.
    """
    kwave_script = Path(__file__).resolve().parent.parent / 'data' / 'run_kwave_benchmark.m'
    if not kwave_script.exists():
        # Try to create a minimal MATLAB benchmark script
        print("[k-Wave] No benchmark script found. Skipping k-Wave OMP timing.")
        print(f"  To enable: create {kwave_script}")
        return None

    try:
        times = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            result = subprocess.run(
                ['matlab', '-batch', f"run('{kwave_script}')"],
                capture_output=True, text=True, timeout=600,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            if result.returncode == 0:
                times.append(elapsed)
            else:
                print(f"  [k-Wave] Run {i} failed: {result.stderr[:200]}")

        if times:
            return {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'n_runs': len(times),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[k-Wave] MATLAB not available or timed out: {e}")

    return None


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='DPC-GNN-Acoustic V4 Inference Benchmark')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_v4/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML (overrides checkpoint config)')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup runs (excluded from stats)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--kwave_runs', type=int, default=5,
                        help='Number of k-Wave OMP benchmark runs (0 to skip)')
    parser.add_argument('--output', type=str, default='benchmark_results.txt',
                        help='Output file for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get config from checkpoint or file
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in ckpt:
        config = ckpt['config']
    else:
        # Fallback to default
        with open('configs/v4_default.yaml', 'r') as f:
            config = yaml.safe_load(f)

    # --- Build model ---
    model = TimedDPCGNNAcousticV4(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    params = model.count_parameters()
    print(f"Parameters: {params}")

    # --- Prepare input ---
    ct_input = torch.rand(args.batch_size, 1, 256, 256, device=device)

    # --- Warmup ---
    print(f"\nWarming up ({args.warmup} runs)...")
    with torch.no_grad():
        for i in range(args.warmup):
            _, _ = model.forward_timed(ct_input)
    print("Warmup complete.")

    # --- Benchmark ---
    print(f"\nBenchmarking ({args.n_runs} runs, batch_size={args.batch_size})...")
    timings = {'encoder_ms': [], 'leapfrog_ms': [], 'beamform_ms': [], 'total_ms': []}

    with torch.no_grad():
        for i in range(args.n_runs):
            _, timing = model.forward_timed(ct_input)
            for k, v in timing.items():
                timings[k].append(v)
            if (i + 1) % 20 == 0:
                mean_total = np.mean(timings['total_ms'])
                print(f"  Run {i+1}/{args.n_runs}: total={timing['total_ms']:.2f} ms "
                      f"(running mean={mean_total:.2f} ms)")

    # --- Compute stats ---
    print(f"\n{'='*70}")
    print(f"DPC-GNN-Acoustic V4 Inference Benchmark Results")
    print(f"{'='*70}")
    print(f"Device:      {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Batch size:  {args.batch_size}")
    print(f"Grid:        256×256, {config['physics']['n_time_steps']} time steps")
    print(f"Parameters:  {params['total']:,} (encoder: {params['encoder']:,})")
    print(f"Runs:        {args.n_runs} (+ {args.warmup} warmup)")
    print(f"{'='*70}")

    results = {}
    for component in ['encoder_ms', 'leapfrog_ms', 'beamform_ms', 'total_ms']:
        arr = np.array(timings[component])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        median_val = np.median(arr)
        p95 = np.percentile(arr, 95)
        name = component.replace('_ms', '').upper()
        pct = (mean_val / np.mean(timings['total_ms'])) * 100 if 'total' not in component else 100.0
        print(f"  {name:12s}: {mean_val:8.2f} ± {std_val:5.2f} ms "
              f"(median={median_val:.2f}, p95={p95:.2f}) [{pct:5.1f}%]")
        results[component] = {'mean': mean_val, 'std': std_val,
                               'median': median_val, 'p95': p95}

    # --- k-Wave OMP comparison ---
    kwave_result = None
    if args.kwave_runs > 0:
        print(f"\n--- k-Wave OMP Benchmark ({args.kwave_runs} runs) ---")
        kwave_result = benchmark_kwave_omp(n_runs=args.kwave_runs)
        if kwave_result:
            speedup = kwave_result['mean_ms'] / results['total_ms']['mean']
            print(f"  k-Wave OMP:  {kwave_result['mean_ms']:.2f} ± {kwave_result['std_ms']:.2f} ms")
            print(f"  Speedup:     {speedup:.1f}× faster than k-Wave OMP")
        else:
            print("  k-Wave OMP benchmark not available (MATLAB not found or no script)")

    # --- Throughput ---
    fps = 1000.0 / results['total_ms']['mean'] * args.batch_size
    print(f"\n  Throughput:  {fps:.1f} samples/sec")

    # --- Save results ---
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write("DPC-GNN-Acoustic V4 Inference Benchmark\n")
        f.write("=" * 60 + "\n")
        f.write(f"Device: {device}\n")
        if device.type == 'cuda':
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Grid: 256x256, {config['physics']['n_time_steps']} steps\n")
        f.write(f"Parameters: {params['total']}\n")
        f.write(f"Runs: {args.n_runs} (+ {args.warmup} warmup)\n\n")
        for component in ['encoder_ms', 'leapfrog_ms', 'beamform_ms', 'total_ms']:
            r = results[component]
            name = component.replace('_ms', '').upper()
            f.write(f"{name}: {r['mean']:.2f} ± {r['std']:.2f} ms "
                    f"(median={r['median']:.2f}, p95={r['p95']:.2f})\n")
        f.write(f"\nThroughput: {fps:.1f} samples/sec\n")
        if kwave_result:
            f.write(f"\nk-Wave OMP: {kwave_result['mean_ms']:.2f} ± {kwave_result['std_ms']:.2f} ms\n")
            f.write(f"Speedup: {kwave_result['mean_ms'] / results['total_ms']['mean']:.1f}x\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
