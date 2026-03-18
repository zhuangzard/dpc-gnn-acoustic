"""
kwave_gnn_evaluator.py — Comprehensive evaluation for k-Wave vs GNN comparison.

Metrics:
  1. Image Quality: SSIM, PSNR, MSE, MAE
  2. Energy Conservation: Energy error, decay rate
  3. Speckle Statistics: SNR, CNR, contrast
  4. Resolution: Axial/Lateral FWHM

Author: Taisen Zhuang & San Ya Research Team
Date: 2026-03-18
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Image quality thresholds
    ssim_threshold: float = 0.85
    psnr_threshold: float = 25.0  # dB
    
    # Energy conservation
    energy_tolerance: float = 0.1  # 10% max variation
    attenuation_rate_range: Tuple[float, float] = (0.0, 0.5)  # Expected dB/cm/MHz
    
    # Speckle analysis
    roi_size: int = 32  # Region of interest for speckle
    background_region: Tuple[int, int, int, int] = (0, 32, 0, 32)  # x1, x2, y1, y2
    
    # Resolution measurement
    fwhm_threshold: float = 10.0  # mm (max acceptable FWHM)
    

class KWaveGNNEvaluator:
    """Comprehensive evaluator for k-Wave GNN predictions."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.results = []
        
    def evaluate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Evaluate a single prediction.
        
        Args:
            pred: (H, W) or (B, H, W) predicted B-mode/pressure field
            target: (H, W) or (B, H, W) ground truth
            metadata: Optional metadata (frequency, depth, etc.)
        
        Returns:
            Dictionary of metrics
        """
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)
            
        # Ensure same size
        if pred.shape != target.shape:
            target = F.interpolate(
                target.unsqueeze(1), 
                size=pred.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        batch_metrics = []
        for i in range(pred.shape[0]):
            metrics = {}
            
            # 1. Image Quality
            metrics.update(self._compute_image_quality(pred[i], target[i]))
            
            # 2. Energy Conservation
            if metadata and 'energy_history' in metadata:
                metrics.update(self._compute_energy_error(metadata['energy_history']))
            
            # 3. Speckle Statistics
            metrics.update(self._compute_speckle_stats(pred[i], target[i]))
            
            # 4. Resolution (if point targets available)
            if metadata and 'point_targets' in metadata:
                metrics.update(self._compute_resolution(pred[i], metadata['point_targets']))
            
            metrics['sample_idx'] = i
            batch_metrics.append(metrics)
        
        self.results.extend(batch_metrics)
        return batch_metrics[0] if len(batch_metrics) == 1 else batch_metrics
    
    def _compute_image_quality(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute image quality metrics."""
        metrics = {}
        
        # MSE
        metrics['mse'] = F.mse_loss(pred, target).item()
        
        # MAE
        metrics['mae'] = F.l1_loss(pred, target).item()
        
        # PSNR
        max_val = max(pred.max().item(), target.max().item(), 1e-10)
        if metrics['mse'] < 1e-10:
            metrics['psnr'] = 100.0
        else:
            metrics['psnr'] = 10 * np.log10(max_val ** 2 / metrics['mse'])
        
        # SSIM
        metrics['ssim'] = self._compute_ssim(pred, target)
        
        # Normalized cross-correlation
        metrics['ncc'] = self._compute_ncc(pred, target)
        
        # Histogram similarity (Bhattacharyya coefficient)
        metrics['hist_similarity'] = self._compute_hist_similarity(pred, target)
        
        return metrics
    
    def _compute_ssim(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        window_size: int = 11,
    ) -> float:
        """Compute Structural Similarity Index."""
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        
        pad = window_size // 2
        
        mu1 = F.conv2d(pred, window, padding=pad)
        mu2 = F.conv2d(target, window, padding=pad)
        
        sigma1_sq = F.conv2d(pred ** 2, window, padding=pad) - mu1 ** 2
        sigma2_sq = F.conv2d(target ** 2, window, padding=pad) - mu2 ** 2
        sigma12 = F.conv2d(pred * target, window, padding=pad) - mu1 * mu2
        
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def _compute_ncc(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Normalized Cross-Correlation."""
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (target_centered ** 2).sum()
        )
        
        if denominator < 1e-10:
            return 0.0
        
        return (numerator / denominator).item()
    
    def _compute_hist_similarity(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        bins: int = 256,
    ) -> float:
        """Compute histogram similarity using Bhattacharyya coefficient."""
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        # Normalize to [0, 1]
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-10)
        target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-10)
        
        hist_pred, _ = np.histogram(pred_np, bins=bins, range=(0, 1), density=True)
        hist_target, _ = np.histogram(target_np, bins=bins, range=(0, 1), density=True)
        
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist_pred * hist_target))
        
        return float(bc)
    
    def _compute_energy_error(self, energy_history: List[float]) -> Dict[str, float]:
        """Compute energy conservation metrics.
        
        Args:
            energy_history: Energy at each time step
        """
        if len(energy_history) < 2:
            return {
                'energy_error_max': 0.0,
                'energy_error_final': 0.0,
                'energy_decay_rate': 0.0,
            }
        
        energies = np.array(energy_history)
        initial = energies[0]
        
        if abs(initial) < 1e-10:
            return {
                'energy_error_max': 0.0,
                'energy_error_final': 0.0,
                'energy_decay_rate': 0.0,
            }
        
        # Relative error at each step
        relative_errors = np.abs(energies - initial) / abs(initial)
        
        # Max error
        max_error = relative_errors.max()
        
        # Final error
        final_error = relative_errors[-1]
        
        # Decay rate (assuming exponential decay)
        if energies[-1] > 0 and initial > 0:
            decay_rate = -np.log(energies[-1] / initial) / len(energies)
        else:
            decay_rate = 0.0
        
        return {
            'energy_error_max': float(max_error),
            'energy_error_final': float(final_error),
            'energy_decay_rate': float(decay_rate),
        }
    
    def _compute_speckle_stats(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute speckle statistics.
        
        Metrics:
        - SNR: Signal-to-noise ratio
        - CNR: Contrast-to-noise ratio
        - Speckle contrast (Rayleigh test)
        """
        metrics = {}
        
        # Convert to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # ROI analysis (center region)
        h, w = pred_np.shape
        roi_size = min(self.config.roi_size, h // 4, w // 4)
        
        center_y, center_x = h // 2, w // 2
        roi_pred = pred_np[
            center_y - roi_size//2:center_y + roi_size//2,
            center_x - roi_size//2:center_x + roi_size//2
        ]
        roi_target = target_np[
            center_y - roi_size//2:center_y + roi_size//2,
            center_x - roi_size//2:center_x + roi_size//2
        ]
        
        # SNR = mean / std
        snr_pred = roi_pred.mean() / (roi_pred.std() + 1e-10)
        snr_target = roi_target.mean() / (roi_target.std() + 1e-10)
        metrics['snr_pred'] = float(snr_pred)
        metrics['snr_target'] = float(snr_target)
        metrics['snr_error'] = float(abs(snr_pred - snr_target) / (snr_target + 1e-10))
        
        # CNR = |mean_signal - mean_background| / std_background
        x1, x2, y1, y2 = self.config.background_region
        bg_pred = pred_np[y1:y2, x1:x2]
        bg_target = target_np[y1:y2, x1:x2]
        
        cnr_pred = abs(roi_pred.mean() - bg_pred.mean()) / (bg_pred.std() + 1e-10)
        cnr_target = abs(roi_target.mean() - bg_target.mean()) / (bg_target.std() + 1e-10)
        metrics['cnr_pred'] = float(cnr_pred)
        metrics['cnr_target'] = float(cnr_target)
        metrics['cnr_error'] = float(abs(cnr_pred - cnr_target) / (cnr_target + 1e-10))
        
        # Speckle contrast (should be ~0.5 for fully developed speckle in envelope)
        speckle_contrast_pred = roi_pred.std() / (roi_pred.mean() + 1e-10)
        speckle_contrast_target = roi_target.std() / (roi_target.mean() + 1e-10)
        metrics['speckle_contrast_pred'] = float(speckle_contrast_pred)
        metrics['speckle_contrast_target'] = float(speckle_contrast_target)
        metrics['speckle_contrast_error'] = float(
            abs(speckle_contrast_pred - speckle_contrast_target) / 
            (speckle_contrast_target + 1e-10)
        )
        
        return metrics
    
    def _compute_resolution(
        self, 
        image: torch.Tensor, 
        point_targets: List[Dict]
    ) -> Dict[str, float]:
        """Compute resolution metrics from point targets.
        
        Args:
            image: B-mode image
            point_targets: List of dicts with 'position' and 'type' (axial/lateral)
        
        Returns:
            Resolution metrics in mm
        """
        metrics = {
            'axial_resolution_mm': float('inf'),
            'lateral_resolution_mm': float('inf'),
        }
        
        image_np = image.cpu().numpy()
        
        axial_fwhms = []
        lateral_fwhms = []
        
        for target in point_targets:
            x, y = target['position']
            target_type = target.get('type', 'point')
            
            # Extract line profile
            if target_type in ['axial', 'point']:
                # Axial profile (vertical line)
                profile = image_np[:, int(x)]
                fwhm = self._measure_fwhm(profile)
                if fwhm > 0:
                    axial_fwhms.append(fwhm)
            
            if target_type in ['lateral', 'point']:
                # Lateral profile (horizontal line)
                profile = image_np[int(y), :]
                fwhm = self._measure_fwhm(profile)
                if fwhm > 0:
                    lateral_fwhms.append(fwhm)
        
        if axial_fwhms:
            metrics['axial_resolution_mm'] = float(np.mean(axial_fwhms))
        if lateral_fwhms:
            metrics['lateral_resolution_mm'] = float(np.mean(lateral_fwhms))
        
        return metrics
    
    def _measure_fwhm(self, profile: np.ndarray) -> float:
        """Measure FWHM from line profile (in pixels)."""
        profile = profile - profile.min()
        max_val = profile.max()
        
        if max_val < 1e-10:
            return 0.0
        
        half_max = max_val / 2
        
        # Find crossing points
        above_half = profile >= half_max
        if not above_half.any():
            return 0.0
        
        indices = np.where(above_half)[0]
        fwhm = indices[-1] - indices[0]
        
        return float(fwhm)
    
    def aggregate_results(self) -> Dict[str, Dict[str, float]]:
        """Aggregate all results with statistics."""
        if not self.results:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())
        all_keys.discard('sample_idx')
        
        aggregated = {}
        for key in all_keys:
            values = [r.get(key, float('nan')) for r in self.results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                }
        
        return aggregated
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        output = {
            'per_sample': self.results,
            'aggregated': self.aggregate_results(),
            'config': {
                'ssim_threshold': self.config.ssim_threshold,
                'psnr_threshold': self.config.psnr_threshold,
                'energy_tolerance': self.config.energy_tolerance,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")
    
    def print_summary(self):
        """Print summary of results."""
        aggregated = self.aggregate_results()
        
        if not aggregated:
            print("No results to summarize.")
            return
        
        print("\n" + "="*70)
        print("  K-WAVE GNN EVALUATION SUMMARY")
        print("="*70)
        print(f"  Total samples: {len(self.results)}")
        print("-"*70)
        
        # Group metrics
        image_quality_keys = ['ssim', 'psnr', 'mse', 'mae', 'ncc', 'hist_similarity']
        speckle_keys = ['snr_pred', 'snr_target', 'cnr_pred', 'cnr_target', 
                        'speckle_contrast_pred', 'speckle_contrast_target']
        energy_keys = ['energy_error_max', 'energy_error_final', 'energy_decay_rate']
        
        def print_group(title, keys):
            print(f"\n  {title}:")
            for key in keys:
                if key in aggregated:
                    stats = aggregated[key]
                    print(f"    {key:25s}: {stats['mean']:8.4f} ± {stats['std']:6.4f} "
                          f"[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print_group("Image Quality", image_quality_keys)
        print_group("Speckle Statistics", speckle_keys)
        print_group("Energy Conservation", energy_keys)
        
        print("\n" + "="*70)
    
    def generate_latex_table(
        self, 
        output_path: Optional[str] = None,
        caption: str = "Quantitative comparison of k-Wave and GNN predictions",
        label: str = "tab:quantitative_results"
    ) -> str:
        """Generate LaTeX table for paper."""
        aggregated = self.aggregate_results()
        
        if not aggregated:
            return ""
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"  \centering")
        latex.append(f"  \\caption{{{caption}}}")
        latex.append(f"  \\label{{{label}}}")
        latex.append(r"  \begin{tabular}{lcccc}")
        latex.append(r"    \toprule")
        latex.append(r"    Metric & Mean & Std & Min & Max \\")
        latex.append(r"    \midrule")
        
        # Key metrics
        key_metrics = [
            ('SSIM', 'ssim'),
            ('PSNR (dB)', 'psnr'),
            ('MSE', 'mse'),
            ('SNR (pred)', 'snr_pred'),
            ('CNR (pred)', 'cnr_pred'),
            ('Speckle Contrast', 'speckle_contrast_pred'),
        ]
        
        for display_name, key in key_metrics:
            if key in aggregated:
                stats = aggregated[key]
                latex.append(
                    f"    {display_name:20s} & "
                    f"{stats['mean']:.4f} & "
                    f"{stats['std']:.4f} & "
                    f"{stats['min']:.4f} & "
                    f"{stats['max']:.4f} \\\\"
                )
        
        latex.append(r"    \bottomrule")
        latex.append(r"  \end{tabular}")
        latex.append(r"\end{table}")
        
        latex_str = "\n".join(latex)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_str)
            print(f"📄 LaTeX table saved to {output_path}")
        
        return latex_str


# Convenience function
def evaluate_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a batch of predictions.
    
    Args:
        predictions: (B, H, W) predicted images
        targets: (B, H, W) ground truth images
        config: Evaluation configuration
    
    Returns:
        Aggregated metrics
    """
    evaluator = KWaveGNNEvaluator(config)
    
    for i in range(predictions.shape[0]):
        evaluator.evaluate(predictions[i], targets[i])
    
    return evaluator.aggregate_results()


if __name__ == "__main__":
    # Test the evaluator
    print("Testing KWaveGNNEvaluator...")
    
    config = EvaluationConfig()
    evaluator = KWaveGNNEvaluator(config)
    
    # Create test data
    pred = torch.rand(128, 128)
    target = pred + torch.randn(128, 128) * 0.1  # Add noise
    
    # Evaluate
    metrics = evaluator.evaluate(pred, target)
    print(f"\nSample metrics: {metrics}")
    
    # Aggregate
    evaluator.print_summary()
    
    # Generate LaTeX
    latex = evaluator.generate_latex_table()
    print("\nLaTeX Table:")
    print(latex)
