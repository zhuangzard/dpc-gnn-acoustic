"""
Route B Training Pipeline.

Loss = L_wavefield + 0.1 * L_momentum + 0.01 * L_energy
  - L_wavefield: MSE(p_pred, p_gt) + (1 - SSIM(p_pred, p_gt))
  - L_momentum:  ||rho * dv/dt + grad_p||^2 (momentum equation residual)
  - L_energy:    max(0, E(t+1) - E(t) - W_source)^2 (energy conservation)

Training strategy:
  - Single-step: model.step_from_frames() predicts next [p, vx, vy]
  - Per-sample backward to avoid OOM
  - Loss on normalized values (all components O(1))
  - Adam, LR=1e-3, cosine decay
  - Batch size: 32 (single frames)
  - Gradient clipping: 1.0
  - AMP: disabled (float32)

Usage:
    python train_routeB.py --data_dir data/wavefield_gt --epochs 200
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from src.models.wave_gnn_routeB import WaveGNNRouteB, P_SCALE, V_SCALE
from src.data.wavefield_dataset import WavefieldDataset


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
class RouteBLoss(nn.Module):
    """
    Combined loss for Route B wave GNN.

    L = MSE(p_pred, p_gt) + MSE(vx_pred, vx_gt) + MSE(vy_pred, vy_gt)

    Operates on flattened (N,) state vectors, not images.
    """

    def __init__(self, w_velocity: float = 1.0):
        super().__init__()
        self.w_velocity = w_velocity

    def forward(self, p_pred: torch.Tensor, p_gt: torch.Tensor,
                vx_pred: torch.Tensor, vx_gt: torch.Tensor,
                vy_pred: torch.Tensor, vy_gt: torch.Tensor) -> tuple:
        """
        MSE on NORMALIZED values so all loss components are O(1).
        Skips velocity loss when GT velocity is all zeros.
        """
        loss_p = F.mse_loss(p_pred / P_SCALE, p_gt / P_SCALE)
        loss_vx = F.mse_loss(vx_pred / V_SCALE, vx_gt / V_SCALE)
        loss_vy = F.mse_loss(vy_pred / V_SCALE, vy_gt / V_SCALE)

        # Skip velocity loss when GT is absent (all zeros)
        has_velocity = vx_gt.abs().max() > 1e-10 or vy_gt.abs().max() > 1e-10
        if has_velocity:
            total_loss = loss_p + self.w_velocity * (loss_vx + loss_vy)
        else:
            total_loss = loss_p

        metrics = {
            'loss_p': loss_p.item(),
            'loss_vx': loss_vx.item(),
            'loss_vy': loss_vy.item(),
            'total_loss': total_loss.item(),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Cosine LR Scheduler with Warmup
# ---------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)


# ---------------------------------------------------------------------------
# P2-6: Mandatory gradient sanity check
# ---------------------------------------------------------------------------
def gradient_sanity_check(model: WaveGNNRouteB, device: torch.device) -> None:
    """
    Run one forward+backward pass and verify ALL parameters have non-zero gradients.
    Raises RuntimeError if any parameter has zero grad.
    """
    print("Running gradient sanity check...")
    model.train()
    model.zero_grad()

    N = model.nx * model.ny
    p = torch.randn(N, device=device) * 100.0
    vx = torch.randn(N, device=device) * 1e-4
    vy = torch.randn(N, device=device) * 1e-4
    c = torch.ones(N, device=device) * 1540.0
    rho = torch.ones(N, device=device) * 1000.0
    alpha = torch.ones(N, device=device) * 2.0

    p_next, vx_next, vy_next = model.step_from_frames(p, vx, vy, c, rho, alpha)

    loss = p_next.pow(2).mean() + vx_next.pow(2).mean() + vy_next.pow(2).mean()
    loss.backward()

    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or param.grad.abs().sum() == 0:
                zero_grad_params.append(name)

    model.zero_grad()

    if zero_grad_params:
        raise RuntimeError(
            f"Gradient sanity check FAILED: {len(zero_grad_params)} parameters "
            f"have zero gradient: {zero_grad_params}")

    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  PASS: All {total_params} learnable parameters have non-zero gradient.")


# ---------------------------------------------------------------------------
# Training Loop (P0-1: model is actually called)
# ---------------------------------------------------------------------------
def train_one_epoch(model: WaveGNNRouteB, dataloader: DataLoader,
                    loss_fn: RouteBLoss, optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    grad_clip: float = 1.0) -> dict:
    """Train for one epoch using step_from_frames() with per-sample backward."""
    model.train()
    total_metrics = {}
    n_batches = 0

    for batch in dataloader:
        p_curr = batch['p_curr'].to(device)       # (B, H, W)
        vx_curr = batch['vx_curr'].to(device)     # (B, H, W)
        vy_curr = batch['vy_curr'].to(device)     # (B, H, W)
        p_next_gt = batch['p_next'].to(device)    # (B, H, W)
        vx_next_gt = batch['vx_next'].to(device)  # (B, H, W)
        vy_next_gt = batch['vy_next'].to(device)  # (B, H, W)
        c_map = batch['c_map'].to(device)         # (B, H, W)
        rho_map = batch['rho_map'].to(device)     # (B, H, W)
        alpha_map = batch['alpha_map'].to(device) # (B, H, W)

        B = p_curr.size(0)
        batch_metrics = {}

        optimizer.zero_grad()

        for b in range(B):
            # Flatten (H, W) -> (N,) for per-sample forward pass
            p_flat = p_curr[b].reshape(-1)
            vx_flat = vx_curr[b].reshape(-1)
            vy_flat = vy_curr[b].reshape(-1)
            c_flat = c_map[b].reshape(-1)
            rho_flat = rho_map[b].reshape(-1)
            alpha_flat = alpha_map[b].reshape(-1)

            p_pred, vx_pred, vy_pred = model.step_from_frames(
                p_flat, vx_flat, vy_flat, c_flat, rho_flat, alpha_flat)

            # Compute loss against GT
            p_gt_flat = p_next_gt[b].reshape(-1)
            vx_gt_flat = vx_next_gt[b].reshape(-1)
            vy_gt_flat = vy_next_gt[b].reshape(-1)

            step_loss, step_metrics = loss_fn(
                p_pred, p_gt_flat, vx_pred, vx_gt_flat, vy_pred, vy_gt_flat)

            # Per-sample backward: free graph immediately
            (step_loss / B).backward()

            for k, v in step_metrics.items():
                batch_metrics[k] = batch_metrics.get(k, 0.0) + v

        for k in batch_metrics:
            batch_metrics[k] /= B

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in batch_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    # Average over batches
    for k in total_metrics:
        total_metrics[k] /= max(n_batches, 1)

    return total_metrics


@torch.no_grad()
def validate(model: WaveGNNRouteB, dataloader: DataLoader,
             loss_fn: RouteBLoss, device: torch.device) -> dict:
    """Validate on held-out data."""
    model.eval()
    total_metrics = {}
    n_samples = 0

    for batch in dataloader:
        p_curr = batch['p_curr'].to(device)
        vx_curr = batch['vx_curr'].to(device)
        vy_curr = batch['vy_curr'].to(device)
        p_next_gt = batch['p_next'].to(device)
        vx_next_gt = batch['vx_next'].to(device)
        vy_next_gt = batch['vy_next'].to(device)
        c_map = batch['c_map'].to(device)
        rho_map = batch['rho_map'].to(device)
        alpha_map = batch['alpha_map'].to(device)

        B = p_curr.size(0)
        for b in range(B):
            p_pred, vx_pred, vy_pred = model.step_from_frames(
                p_curr[b].reshape(-1), vx_curr[b].reshape(-1), vy_curr[b].reshape(-1),
                c_map[b].reshape(-1), rho_map[b].reshape(-1), alpha_map[b].reshape(-1))

            _, metrics = loss_fn(
                p_pred, p_next_gt[b].reshape(-1),
                vx_pred, vx_next_gt[b].reshape(-1),
                vy_pred, vy_next_gt[b].reshape(-1))

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_samples += 1

    for k in total_metrics:
        total_metrics[k] /= max(n_samples, 1)

    return total_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Route B: Wave GNN Training")
    parser.add_argument('--data_dir', type=str, default='data/wavefield_gt')
    parser.add_argument('--output_dir', type=str, default='checkpoints_routeB')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)  # B=4 for per-sample sequential loop
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--w_velocity', type=float, default=1.0)
    parser.add_argument('--hidden_dim', type=int, default=80)
    parser.add_argument('--n_mp_layers', type=int, default=4)
    parser.add_argument('--checkpoint_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Max frames per sample (0=all). Reduces epoch time.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ---
    dataset = WavefieldDataset(args.data_dir, rollout_len=1,
                               max_frames_per_file=args.max_frames)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Dataset: {len(dataset)} samples ({n_train} train, {n_val} val)")

    # --- Infer grid size from first sample ---
    sample0 = dataset[0]
    grid_h, grid_w = sample0['c_map'].shape
    print(f"Grid size: {grid_h}x{grid_w} (inferred from data)")
    n_elements = min(128, grid_w - 2 * 20)  # fit within non-PML region
    pml_width = min(20, grid_h // 8)  # scale PML for small grids

    # --- Model ---
    model = WaveGNNRouteB(
        nx=grid_h, ny=grid_w, dx=2.34e-4, dt=4.0e-8,
        pml_width=pml_width, hidden_dim=args.hidden_dim,
        n_mp_layers=args.n_mp_layers, n_elements=n_elements,
        checkpoint_every=args.checkpoint_every,
    ).to(device)

    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,}")
    for k, v in params.items():
        if k != 'total':
            print(f"  {k}: {v:,}")

    # P2-6: Mandatory gradient sanity check before training
    gradient_sanity_check(model, device)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # --- Loss ---
    loss_fn = RouteBLoss(w_velocity=args.w_velocity)

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # --- Training loop ---
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            grad_clip=args.grad_clip,
        )

        val_metrics = validate(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0
        val_loss = val_metrics.get('total_loss', float('inf'))

        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"lr={lr:.2e} | "
              f"train_loss={train_metrics.get('total_loss', 0):.6f} | "
              f"val_loss={val_loss:.6f} | "
              f"loss_p={val_metrics.get('loss_p', 0):.4e} | "
              f"{elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'params': params,
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1

        # Periodic save
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(args.output_dir, f'epoch_{epoch:03d}.pt'))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {args.output_dir}/")


if __name__ == '__main__':
    main()
