#!/usr/bin/env python3
"""
evaluate.py — Evaluation script for DPC-GNN-Acoustic v2.

Loads a trained model and evaluates on test set with comprehensive metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/default_2d.yaml
    python evaluate.py --checkpoint checkpoints/best.pt --visualize
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.dpc_gnn_acoustic import DPCGNNAcousticV2, create_model
from data.kwave_dataset import KWaveDataset, create_dataloader
from utils.metrics import evaluate_sample, print_metrics


@torch.no_grad()
def evaluate(
    model: DPCGNNAcousticV2,
    test_loader,
    device: str = 'cuda',
    output_dir: str = 'results',
    visualize: bool = False,
):
    """Run evaluation on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        output_dir: Output directory for results
        visualize: Save visualization plots
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = []
    
    for i, batch in enumerate(test_loader):
        hu = batch['hu'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr = batch['edge_attr'].to(device)
        bmode_gt = batch['bmode_gt'].to(device)
        transducer_idx = batch['transducer_idx'].to(device)
        positions = batch['positions'].to(device)
        domain_size = batch['domain_size'].to(device)
        node_props = {k: v.to(device) for k, v in batch['node_props'].items()}
        
        # Forward
        outputs = model(
            hu, edge_index, edge_attr, node_props,
            transducer_idx, positions, domain_size,
        )
        
        # Metrics
        metrics = evaluate_sample(
            outputs['bmode'], bmode_gt,
            outputs.get('energy_history'),
        )
        metrics['sample_idx'] = i
        all_metrics.append(metrics)
        
        print(f"  Sample {i}: SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}dB, MAE={metrics['mae']:.6f}")
        
        # Visualization
        if visualize:
            _save_visualization(
                outputs['bmode'].cpu().numpy(),
                bmode_gt.cpu().numpy(),
                outputs.get('pressure_field', torch.zeros(1)).cpu().numpy(),
                os.path.join(output_dir, f'sample_{i:04d}.png'),
                metrics,
            )
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0]:
            if isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({len(all_metrics)} samples)")
        print(f"{'='*60}")
        for key, stats in avg_metrics.items():
            if key != 'sample_idx':
                print(f"  {key:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"{'='*60}")
        
        # Save results
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'per_sample': all_metrics,
                'aggregate': avg_metrics,
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}/metrics.json")
    
    return all_metrics


def _save_visualization(pred, target, pressure, save_path, metrics):
    """Save comparison visualization."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(target, cmap='gray', aspect='auto')
        axes[0].set_title('k-Wave GT')
        axes[0].axis('off')
        
        axes[1].imshow(pred, cmap='gray', aspect='auto')
        axes[1].set_title(f'DPC-GNN v2\nSSIM={metrics["ssim"]:.3f}, PSNR={metrics["psnr"]:.1f}dB')
        axes[1].axis('off')
        
        diff = np.abs(pred - target) if pred.shape == target.shape else pred
        axes[2].imshow(diff, cmap='hot', aspect='auto')
        axes[2].set_title(f'|Difference|\nMAE={metrics["mae"]:.4f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        pass  # No matplotlib


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPC-GNN-Acoustic v2")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_2d.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ Model loaded from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    
    # Test loader
    data_cfg = config.get('data', {})
    test_loader = create_dataloader(
        data_cfg.get('kwave_data_dir', 'data/kwave_gt'),
        split='test',
        grid_resolution=config.get('graph', {}).get('grid_resolution', 256),
        frequency=config.get('physics', {}).get('frequency', 5e6),
    )
    
    # Evaluate
    evaluate(model, test_loader, device, args.output, args.visualize)


if __name__ == '__main__':
    main()
