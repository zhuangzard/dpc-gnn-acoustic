# Paper Outline: DPC-GNN-Acoustic

## Title Options
1. "DPC-GNN-Acoustic: Differentiable Hard-Physics Ultrasound Simulation"
2. "Wave-GNN: Graph Neural Networks with Hard Wave Equation Constraints"
3. "Differentiable Ultrasound: Physics-Constrained GNN for Real-time CT-to-US"

## Target Venues
- MICCAI 2026 (Method track)
- IEEE TMI (Transactions on Medical Imaging)
- IPMI 2025 (Information Processing in Medical Imaging)

## Abstract Structure

### Background
- CT-to-US synthesis is crucial for surgical navigation
- Physical simulation is accurate but slow (40+ minutes)
- Deep learning is fast but lacks physical correctness

### Problem
- Existing methods: either accurate OR fast, not both
- PINNs use soft constraints (loss function) → may violate physics
- DPC-GNN uses hard constraints but not differentiable

### Method
- **DPC-GNN-Acoustic**: Hard physics + Differentiability
- Wave equation as differentiable Message Passing layer
- End-to-end trainable: CT → US (physical correct + real-time)

### Results
- Speed: 40min → 100ms (×24,000 faster)
- Accuracy: SSIM 0.85 vs physical simulation
- Physical correctness: Guaranteed by design

## Paper Structure

### 1. Introduction (1 page)
- Motivation: Real-time US for surgical navigation
- Gap: No method is both fast and physically correct
- Contribution: DPC-GNN-Acoustic paradigm

### 2. Related Work (1 page)
- Physical simulation: k-Wave, Field II, SIMUS
- Deep learning: CycleGAN, Diffusion models
- Hybrid: PINNs, DiffUS
- **Gap**: No hard-physics + differentiable method

### 3. Method (3 pages)

#### 3.1 Acoustic Wave Equation
- Mathematical formulation
- Discretization for GNN

#### 3.2 WaveEquationMP Layer
- Message passing formulation
- Differentiability proof
- Hard constraint satisfaction

#### 3.3 DPC-GNN-Acoustic Architecture
- Encoder: CT → initial pressure
- Physics core: WaveEquationMP layers
- Decoder: pressure → US image

#### 3.4 Training Strategy
- Supervised: Simulated data
- Self-supervised: Physical consistency loss
- End-to-end differentiable

### 4. Experiments (2 pages)

#### 4.1 Dataset
- SimUS v3.1 generated data
- Real US for validation

#### 4.2 Baselines
- Physical: k-Wave, SIMUS
- Learning: CycleGAN, Diffusion
- Hybrid: PINNs, DiffUS

#### 4.3 Metrics
- Image quality: SSIM, PSNR
- Physical correctness: Wave equation residual
- Speed: Inference time

#### 4.4 Results
- Quantitative comparison table
- Qualitative visualization
- Ablation study

### 5. Discussion (1 page)
- Limitations
- Future work: Multi-frequency, 3D, real patient data

### 6. Conclusion (0.5 page)

## Key Figures

1. **Architecture diagram**: CT → Encoder → WaveEquationMP → Decoder → US
2. **WaveEquationMP detail**: Message passing with physics
3. **Comparison**: Physical (slow) vs PINNs (soft) vs Ours (hard+fast)
4. **Results**: Speed-accuracy trade-off plot

## Timeline

- **Week 1-2**: Implement WaveEquationMP, verify differentiability
- **Week 3-4**: Train on SimUS data, tune hyperparameters
- **Week 5-6**: Real US validation, clinical evaluation
- **Week 7-8**: Paper writing, submission preparation

## Related Papers to Cite

1. DPC-GNN (our previous work)
2. DiffUS (MICCAI 2025)
3. PINNs for ultrasound (Ultrasonics 2023)
4. MeshGraphNets (DeepMind, ICML 2021)
5. k-Wave (standard physical simulator)
