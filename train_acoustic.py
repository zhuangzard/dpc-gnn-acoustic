#!/usr/bin/env python3
"""
train_acoustic.py — DPC-GNN-Acoustic Training Script
Physics-constrained GNN for acoustic wave propagation.
Wave equation on ultrasound phantom.
"""

import os, sys, math, time, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.acoustic_wave_gnn import AcousticWaveGNN, build_acoustic_graph, check_cfl_condition

# ═══════════════════════════════════════════════════
# 1. Data Generation
# ═══════════════════════════════════════════════════

def generate_acoustic_phantom(
    Lx=0.1, Ly=0.1, Lz=0.1,
    nx=20, ny=20, nz=20,
    device="cpu"
):
    """Generate 3D acoustic phantom with tissue properties.
    
    Args:
        Lx, Ly, Lz: Domain dimensions [m]
        nx, ny, nz: Grid resolution
        device: torch device
    
    Returns:
        dict with node positions, properties, graph, etc.
    """
    # Create regular grid
    x = torch.linspace(0, Lx, nx+1, device=device)
    y = torch.linspace(0, Ly, ny+1, device=device)
    z = torch.linspace(0, Lz, nz+1, device=device)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing='ij')
    
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).float()
    N = positions.shape[0]
    
    # Build graph using k-NN
    ei, ea = build_acoustic_graph(positions, k=8)
    
    # Assign acoustic properties based on position (simple phantom)
    # Background: soft tissue (c=1540 m/s, ρ=1000 kg/m³)
    # Sphere inclusion: bone-like (c=3000 m/s, ρ=1900 kg/m³)
    center = torch.tensor([Lx/2, Ly/2, Lz/2], device=device)
    radius = min(Lx, Ly, Lz) / 4
    
    dist_to_center = torch.norm(positions - center, dim=-1)
    is_inclusion = dist_to_center < radius
    
    # Acoustic properties
    c = torch.where(is_inclusion, 
                    torch.tensor(3000.0, device=device),
                    torch.tensor(1540.0, device=device))
    rho = torch.where(is_inclusion,
                      torch.tensor(1900.0, device=device),
                      torch.tensor(1000.0, device=device))
    alpha = torch.where(is_inclusion,
                        torch.tensor(10.0, device=device),  # Np/m
                        torch.tensor(0.5, device=device))
    
    # HU values (approximate)
    hu = torch.where(is_inclusion,
                     torch.tensor(1000.0, device=device),
                     torch.tensor(50.0, device=device))
    
    # Node features: [ρ, c, α, HU]
    nf = torch.stack([
        rho / 1000.0,      # normalized density
        c / 1540.0,        # normalized sound speed
        alpha / 10.0,      # normalized attenuation
        hu / 1000.0        # normalized HU
    ], dim=-1)
    
    # Transducer mask (front face y=0)
    transducer_mask = positions[:, 1] < 1e-6
    
    return {
        "positions": positions,
        "ei": ei,
        "ea": ea,
        "nf": nf,
        "c": c,
        "rho": rho,
        "alpha": alpha,
        "transducer_mask": transducer_mask,
        "is_inclusion": is_inclusion,
        "N": N,
        "Lx": Lx, "Ly": Ly, "Lz": Lz
    }


# ═══════════════════════════════════════════════════
# 2. Physics
# ═══════════════════════════════════════════════════

def initial_pressure(positions, frequency=5e6, amplitude=1.0):
    """Generate initial pressure field (Gaussian pulse).
    
    Args:
        positions: (N, 3) node positions
        frequency: Ultrasound frequency [Hz]
        amplitude: Pressure amplitude
    
    Returns:
        p0: (N, 1) initial pressure
    """
    # Gaussian pulse at center
    center = positions.mean(dim=0)
    sigma = 0.01  # 1cm width
    
    dist = torch.norm(positions - center, dim=-1, keepdim=True)
    p0 = amplitude * torch.exp(-dist**2 / (2 * sigma**2))
    
    return p0


def compute_physics_loss(model, nf, ei, ea, dt, c, p0):
    """Compute physics-informed loss (wave equation residual).
    
    Loss = ||∂²p/∂t² - c²∇²p||²
    
    Args:
        model: AcousticWaveGNN model
        nf, ei, ea: Graph data
        dt: Time step
        c: Sound speed
        p0: Initial pressure
    
    Returns:
        loss: Scalar physics loss
    """
    # Encode to hidden state
    h = model.enc(nf)
    
    # Compute Laplacian at initial state
    laplacian_0 = model.mps[0](h, ei, ea)
    
    # Initialize pressure history (Taylor expansion)
    c_sq = (c.float().mean() ** 2)
    p_prev = p0 - 0.5 * c_sq * (dt ** 2) * laplacian_0
    p_curr = p0.clone()
    
    # Compute residual at a few time steps
    loss = 0.0
    n_check = min(5, model.n_layers)
    
    for k in range(n_check):
        # Compute Laplacian
        laplacian = model.mps[k](p_curr, ei, ea)
        
        # Time step (leapfrog)
        p_next = 2.0 * p_curr - p_prev + c_sq * (dt ** 2) * laplacian
        
        # Wave equation residual
        # ∂²p/∂t² ≈ (p^{n+1} - 2*p^n + p^{n-1}) / dt²
        p_tt = (p_next - 2 * p_curr + p_prev) / (dt ** 2)
        
        # Residual: p_tt - c² * ∇²p
        residual = p_tt - c_sq * laplacian
        
        loss = loss + (residual ** 2).mean()
        
        # Update
        p_prev = p_curr
        p_curr = p_next
    
    return loss / n_check


# ═══════════════════════════════════════════════════
# 3. Training
# ═══════════════════════════════════════════════════

def train_medium(medium, c_val, rho_val, alpha_val, epochs=500, lr=1e-3,
                 hdim=64, n_layers=6, nx=20, ny=20, nz=20, device=None):
    """Train on a single acoustic medium (类比 train_tissue).
    
    Args:
        medium: Medium name (e.g., 'liver', 'fat')
        c_val: Sound speed [m/s]
        rho_val: Density [kg/m³]
        alpha_val: Attenuation [Np/m]
        epochs: Number of training epochs
        lr: Learning rate
        hdim: Hidden dimension
        n_layers: Number of MP layers
        nx, ny, nz: Grid resolution
        device: torch device
    
    Returns:
        results: Training results dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Time step (CFL condition)
    dx_min = 0.1 / nx  # minimum grid spacing
    dt = 0.5 * dx_min / (c_val * math.sqrt(3))  # CFL with safety factor
    
    print(f"\n{'='*70}")
    print(f"  MEDIUM: {medium.upper()} | c={c_val:.0f} m/s | ρ={rho_val:.0f} kg/m³ | α={alpha_val:.1f} Np/m")
    print(f"  Grid: {nx}x{ny}x{nz} | Device: {device} | Epochs: {epochs}")
    print(f"  Time step: {dt:.2e} s (CFL)")
    print(f"{'='*70}")
    
    # Generate phantom
    phantom = generate_acoustic_phantom(nx=nx, ny=ny, nz=nz, device=device)
    N = phantom["N"]
    ei = phantom["ei"]
    ea = phantom["ea"]
    positions = phantom["positions"]
    
    # Set medium properties
    c = torch.full((N,), c_val, device=device)
    rho = torch.full((N,), rho_val, device=device)
    alpha = torch.full((N,), alpha_val, device=device)
    hu = torch.zeros(N, device=device)
    
    # Node features
    nf = torch.stack([
        rho / 1000.0,
        c / 1540.0,
        alpha / 10.0,
        hu / 1000.0
    ], dim=-1)
    
    # Initial pressure
    p0 = initial_pressure(positions)
    
    # Augment edge attributes with acoustic properties
    src, dst = ei
    Z = rho * c
    Z_ratio = Z[dst] / (Z[src] + 1e-8)
    
    distance = ea[:, 3:4]
    alpha_avg = (alpha[src] + alpha[dst]) / 2
    atten_factor = torch.exp(-alpha_avg.unsqueeze(-1) * distance)
    
    ea_full = torch.cat([
        ea,                      # (E, 4) [r_vec, distance]
        Z_ratio.unsqueeze(-1),   # (E, 1)
        atten_factor             # (E, 1)
    ], dim=-1)  # (E, 6)
    
    print(f"  Nodes: {N} | Edges: {ei.shape[1]}")
    
    # Model
    model = AcousticWaveGNN(hdim=hdim, n_layers=n_layers, node_dim=4, edge_dim=6).to(device)
    print(f"  Params: {model.count_params():,}")
    
    # Optimizer
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    warmup = min(50, epochs//10)
    
    best_loss, best_ep, best_state = float('inf'), 0, None
    history = []
    t0 = time.time()
    
    print(f"\n  {'Ep':>5} | {'Loss':>12} | {'Physics':>12} | {'P_max':>10} | {'lr':>9}")
    print(f"  {'-'*60}")
    
    for ep in range(1, epochs+1):
        if ep <= warmup:
            for pg in opt.param_groups:
                pg['lr'] = lr * ep / warmup
        
        opt.zero_grad()
        
        # Forward pass
        p = model(nf, ei, ea_full, dt, c.unsqueeze(-1))
        
        # Physics loss
        physics_loss = compute_physics_loss(model, nf, ei, ea_full, dt, c, p0)
        
        # Data loss (MSE with initial pressure at transducer)
        transducer_mask = phantom["transducer_mask"]
        data_loss = (p[transducer_mask] - p0[transducer_mask]).pow(2).mean()
        
        # Total loss
        loss = physics_loss + 0.1 * data_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if ep > warmup:
            sched.step()
        
        lv = loss.item()
        p_max = p.abs().max().item()
        
        rec = {
            "ep": ep,
            "loss": lv,
            "physics_loss": physics_loss.item(),
            "data_loss": data_loss.item(),
            "p_max": p_max
        }
        history.append(rec)
        
        if lv < best_loss:
            best_loss, best_ep = lv, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if ep <= 5 or ep % 25 == 0 or ep == epochs:
            print(f"  {ep:5d} | {lv:12.6e} | {physics_loss.item():12.6e} | "
                  f"{p_max:10.4e} | {opt.param_groups[0]['lr']:9.2e}")
        
        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep}! Reverting to best (ep {best_ep})")
            if best_state:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                for pg in opt.param_groups:
                    pg['lr'] *= 0.1
            else:
                break
    
    dt_train = time.time() - t0
    
    # Final eval
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    with torch.no_grad():
        p_final = model(nf, ei, ea_full, dt, c.unsqueeze(-1))
    
    results = {
        "medium": medium,
        "c_ms": c_val,
        "rho_kgm3": rho_val,
        "alpha_Npm": alpha_val,
        "epochs": epochs,
        "best_epoch": best_ep,
        "best_loss": best_loss,
        "final_p_max": p_final.abs().max().item(),
        "training_time_s": dt_train,
        "N_nodes": N,
        "params": model.count_params()
    }
    
    print(f"\n  {'='*70}")
    print(f"  ✅ {medium.upper()} COMPLETE")
    print(f"  Best loss: {best_loss:.6e} (epoch {best_ep})")
    print(f"  Training time: {dt_train:.1f}s ({dt_train/60:.1f}min)")
    print(f"  {'='*70}\n")
    
    # Save checkpoint
    ckpt_dir = f"/root/results/acoustic_{medium}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state:
        torch.save(best_state, f"{ckpt_dir}/best_model.pt")
    with open(f"{ckpt_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{ckpt_dir}/history.json", "w") as f:
        json.dump(history, f)
    
    return results


# ═══════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════

MEDIA = {
    "liver":   {"c": 1540, "rho": 1050, "alpha": 0.5},
    "fat":     {"c": 1450, "rho": 950,  "alpha": 0.3},
    "muscle":  {"c": 1580, "rho": 1050, "alpha": 0.8},
    "bone":    {"c": 3000, "rho": 1900, "alpha": 10.0},
    "water":   {"c": 1480, "rho": 1000, "alpha": 0.02},
    "blood":   {"c": 1570, "rho": 1060, "alpha": 0.2},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium", type=str, default=None, help="Single medium to train")
    parser.add_argument("--c", type=float, default=None, help="Sound speed [m/s]")
    parser.add_argument("--rho", type=float, default=None, help="Density [kg/m³]")
    parser.add_argument("--alpha", type=float, default=None, help="Attenuation [Np/m]")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hdim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"\n{'='*70}")
    print(f"  DPC-GNN-Acoustic Training")
    print(f"  Device: {device}")
    print(f"  Hidden dim: {args.hdim} | Layers: {args.n_layers}")
    print(f"{'='*70}\n")
    
    if args.medium:
        # Single medium
        if args.c is not None:
            props = {"c": args.c, "rho": args.rho or 1000, "alpha": args.alpha or 0.5}
        else:
            props = MEDIA[args.medium]
        
        train_medium(
            args.medium, props["c"], props["rho"], props["alpha"],
            epochs=args.epochs, lr=args.lr, hdim=args.hdim,
            n_layers=args.n_layers, nx=args.nx, ny=args.nx, nz=args.nx,
            device=device
        )
    else:
        # All media
        all_results = {}
        for medium, props in MEDIA.items():
            results = train_medium(
                medium, props["c"], props["rho"], props["alpha"],
                epochs=args.epochs, lr=args.lr, hdim=args.hdim,
                n_layers=args.n_layers, nx=args.nx, ny=args.nx, nz=args.nx,
                device=device
            )
            all_results[medium] = results
        
        # Summary
        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETE - SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Medium':<12} | {'c [m/s]':>10} | {'Best Loss':>12} | {'Time [s]':>10}")
        print(f"  {'-'*55}")
        for medium, res in all_results.items():
            print(f"  {medium:<12} | {res['c_ms']:>10.0f} | {res['best_loss']:>12.6e} | {res['training_time_s']:>10.1f}")
        print(f"{'='*70}\n")
        
        # Save summary
        with open("/root/results/acoustic_all_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
