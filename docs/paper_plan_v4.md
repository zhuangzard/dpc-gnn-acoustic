# DPC-GNN-Acoustic: Paper Plan (V4 Architecture)

**Working Title:**  
*Physics-as-Forward: Differentiable Acoustic Simulation via Graph Neural Networks for CT-to-Ultrasound Image Synthesis*

**Authors:** Taisen Zhuang, Hao Liu  
**Affiliation:** Embodied AI & Surgical Robotics Lab, Hands Robotics  
**Date:** 2026-03-19

---

## 1. Core Contributions (7 Points)

### C1. Physics-as-Forward Paradigm (L3) — The Central Thesis
We introduce a new level in the physics-informed learning hierarchy: **Physics-as-Forward (L3)**, where the entire forward pass *is* the physics simulation — not a regularization term (L1/PINN), not a hard constraint on architecture (L2). The neural network learns only material properties; wave propagation and image formation are executed by parameter-free differentiable solvers. This yields **zero spurious solutions** — every output is physically realizable.

**Why top-tier:** Reframes the entire PINN/physics-informed debate with a clean taxonomy (L0–L3). Broadly applicable beyond acoustics.

### C2. L0–L3 Taxonomy of Physics-Informed Learning
We formalize four levels of physics integration in learned simulators:
| Level | Name | Physics Role | Learnable Physics? | Example |
|-------|------|-------------|-------------------|---------|
| L0 | Data-only | None | N/A | U-Net, Pix2Pix |
| L1 | Physics-as-Loss (PINN) | Regularization in loss | Yes (implicit) | Raissi et al. 2019 |
| L2 | Physics-as-Architecture | Hard constraints in network | Partial | HNN, LNN, DPC-GNN-solid |
| L3 | Physics-as-Forward | Entire forward pass | No — only material properties | **This work** |

**Why top-tier:** Provides a unifying framework the community can adopt. Taxonomies drive citations.

### C3. Cross-Domain Transfer of Antisymmetric Message Passing
The DPC-GNN architecture, originally designed for solid mechanics (force balance F_ij = −F_ji), transfers to acoustic wave propagation via reinterpretation of antisymmetric message passing as **spatial acoustic reciprocity** (p_ij = −p_ji at interfaces). Same architecture, different physics — demonstrating that the inductive bias is more general than its original domain.

**Why top-tier:** Shows GNN architectural priors generalize across PDE families, not just parameter regimes.

### C4. Differentiable Pulse-Echo Ultrasound Pipeline
We implement a fully differentiable acoustic imaging pipeline:
- **Leapfrog wave equation** (1754 time steps, CFL=0.206, PML boundaries) — 0 learnable parameters
- **Delay-and-Sum (DAS) beamformer** — 0 learnable parameters
- Gradient flows through all 1754 steps back to the GNN encoder

This is, to our knowledge, the first end-to-end differentiable pulse-echo ultrasound simulator trained against ground-truth B-mode images.

**Why top-tier:** Enables gradient-based optimization of any upstream module (segmentation, registration) through a physics-faithful acoustic model.

### C5. Residual Material Property Prediction
The GNN predicts material properties as residuals over a physics-informed prior:
```
c(x,y) = c_table(CT_HU) + c_residual(GNN),  |c_residual| ≤ 150 m/s
```
The lookup table `c_table` encodes known HU-to-speed-of-sound relationships; the GNN learns patient-specific corrections. This decomposition:
- Guarantees physically plausible speed-of-sound (1400–1700 m/s)
- Reduces learning burden (GNN only models deviations)
- Provides interpretable outputs (residual maps reveal what CT alone cannot capture)

**Why top-tier:** Principled fusion of domain knowledge and learned correction — applicable to any physics parameter estimation task.

### C6. Extreme Parameter Efficiency
The entire learnable component is a 253K-parameter GNN encoder. Compare:
| Method | Parameters | Physics Guarantee |
|--------|-----------|-------------------|
| Pix2Pix (L0) | ~54M | None |
| CycleGAN (L0) | ~28M | None |
| PINN-style V3 | ~500K | Soft (loss only) |
| **DPC-GNN-Acoustic V4** | **253K** | **Hard (forward pass)** |

With 100–200× fewer parameters than data-driven baselines, V4 matches V3's best SSIM in 2 epochs vs 30, suggesting dramatically better sample efficiency.

**Why top-tier:** Practical for clinical deployment — small model, fast inference (<100ms target), interpretable outputs.

### C7. Convergence Superiority of Physics-as-Forward
V4 reaches SSIM=0.518 at epoch 2, matching V3's best (SSIM=0.520 at epoch 30) — a **15× convergence speedup**. This is not just faster training; it demonstrates that when the forward model is physically correct, the optimizer has a vastly smoother loss landscape because every gradient direction corresponds to a physically meaningful change in material properties.

**Why top-tier:** Empirically validates the theoretical advantage of L3 over L1 with a clean apples-to-apples comparison.

---

## 2. Paper Structure

### Abstract (~250 words)
- Problem: CT-to-ultrasound synthesis for surgical planning/navigation
- Gap: Data-driven methods lack physics guarantees; PINNs offer soft constraints only
- Contribution: Physics-as-Forward (L3) — GNN predicts material properties, differentiable wave equation + DAS beamformer produce B-mode images with zero learnable parameters in the physics pipeline
- Results: 15× faster convergence than PINN-style, 253K params, SSIM>0.9 target
- Impact: New paradigm for physics-informed image synthesis

### 1. Introduction (2 pages)
- **Hook:** Clinical need for CT-to-US synthesis (surgical planning, training, registration)
- **Problem statement:** Existing approaches are either physics-free (hallucinate) or physics-soft (PINNs — violate physics at test time)
- **Key insight:** What if the neural network doesn't simulate physics at all — it only predicts *what the medium looks like*, and physics does the rest?
- **Contribution summary:** 7 points (see above)
- **Paper organization**

### 2. Related Work (2 pages)

#### 2.1 CT-to-Ultrasound Synthesis
- Data-driven: Pix2Pix, CycleGAN, diffusion models for cross-modal synthesis
- Simulation-based: k-Wave, Field II, MUST — accurate but non-differentiable
- Hybrid: Learning + simulation components

#### 2.2 Physics-Informed Neural Networks
- PINNs (Raissi et al.): PDE residual as loss
- Hard-constrained networks: Hamiltonian NN, Lagrangian NN
- Neural operators: FNO, DeepONet — learn PDE solutions but not guaranteed to satisfy PDEs

#### 2.3 Graph Neural Networks for Physics
- GNS (Sanchez-Gonzalez et al.): Learned simulators
- MeshGraphNets: FEM-like learned solvers
- **DPC-GNN (original):** Antisymmetric MP for solid mechanics — our direct predecessor

#### 2.4 Differentiable Simulation
- Differentiable rendering (NeRF, 3DGS)
- Differentiable physics (DiffTaichi, Brax, Warp)
- Gap: No differentiable pulse-echo ultrasound pipeline exists

### 3. Method (4–5 pages)

#### 3.1 Problem Formulation
- Input: CT image I_CT ∈ ℝ^{256×256}
- Output: B-mode ultrasound image I_US ∈ ℝ^{128×128}
- Approach: Learn mapping CT → material properties → (physics) → B-mode

#### 3.2 L0–L3 Taxonomy
- Formal definitions of each level
- Mathematical characterization of what is learned vs. fixed
- Table comparing guarantees

#### 3.3 GNN Encoder: CT → Material Properties
- Graph construction from CT pixel grid
- Antisymmetric message passing: m_ij = −m_ji
  - Physical motivation: acoustic reciprocity
  - Connection to original force-balance interpretation
- Output heads: c(x,y), α(x,y), σ(x,y)
- Residual parameterization: c = c_table(HU) + c_residual

#### 3.4 Differentiable Wave Equation Solver
- 2D acoustic wave equation with attenuation
- Leapfrog time integration (second-order accurate)
- CFL condition: CFL = c_max · Δt / Δx = 0.206
- PML absorbing boundary conditions
- Source injection: pulse-echo configuration
- Gradient stability through 1754 time steps

#### 3.5 Differentiable DAS Beamformer
- Delay calculation from transducer geometry
- Coherent summation across receive channels
- Envelope detection and log compression
- Vectorized implementation (13× speedup)

#### 3.6 Loss Function
- L = L1(I_pred, I_GT) + λ(1 − SSIM(I_pred, I_GT))
- Why SSIM: perceptual quality matters for clinical use
- No physics loss needed — physics is in the forward pass

### 4. Experiments (3–4 pages)

#### 4.1 Dataset and Ground Truth
- CT phantoms / anatomical models
- k-Wave simulation as ground truth B-mode generator
- Data splits, augmentation

#### 4.2 Baselines
- **L0:** Pix2Pix, U-Net (pure data-driven)
- **L1:** V3 PINN-style (physics loss)
- **L2:** Physics-constrained architecture (if applicable baseline exists)
- **L3:** V4 Physics-as-Forward (ours)

#### 4.3 Quantitative Results
- SSIM, PSNR, LPIPS vs. k-Wave ground truth
- Convergence curves (epochs to target SSIM)
- Parameter count comparison
- Inference time comparison

#### 4.4 Ablation Studies
- With/without residual parameterization (c_table + c_residual vs. direct prediction)
- With/without antisymmetric MP (vs. standard MP)
- With/without PML boundaries
- Number of Leapfrog time steps (convergence vs. accuracy)
- Loss components (L1 only vs. L1+SSIM)

#### 4.5 Qualitative Analysis
- B-mode image comparison (GT / V4 / V3 / Pix2Pix)
- Predicted material property maps vs. ground truth
- Error maps and failure cases

#### 4.6 Computational Analysis
- Training time comparison (V3 30 epochs ≈ V4 2 epochs in SSIM)
- Inference speed: GNN + Leapfrog + DAS vs. k-Wave
- Memory footprint

### 5. Discussion (1.5 pages)

#### 5.1 Why Physics-as-Forward Works
- Loss landscape smoothness: every gradient is physically meaningful
- No mode collapse: physics constrains output space
- Implicit regularization: fewer learnable parameters = less overfitting

#### 5.2 Limitations
- 2D only (3D extension requires GPU memory solutions)
- Fixed transducer geometry (not yet learnable)
- Homogeneous density assumption
- Current SSIM gap to target (if not yet >0.9 at submission)

#### 5.3 Clinical Implications
- Surgical planning: preview US images from preoperative CT
- Training: generate realistic US training data
- Registration: differentiable CT→US enables gradient-based registration

#### 5.4 Broader Impact of the L0–L3 Framework
- Applicable to any domain with known forward physics
- Examples: MRI simulation, optical coherence tomography, radar

### 6. Conclusion (0.5 page)
- Summary of contributions
- Key result: Physics-as-Forward achieves faster convergence, fewer parameters, and physical guarantees
- Future work: 3D extension, clinical validation, learnable transducer optimization

---

## 3. Figure Plan

### Figure 1: Architecture Overview (Full Page)
**Content:** The complete DPC-GNN-Acoustic V4 pipeline.
```
┌──────────────────────────────────────────────────────────────┐
│  CT Image (256×256)                                          │
│       ↓                                                      │
│  ┌─────────────────────┐                                     │
│  │  GNN Encoder (253K) │  ← Only learnable component        │
│  │  Antisymmetric MP   │                                     │
│  └──────┬──────────────┘                                     │
│         ↓                                                    │
│  c(x,y)  α(x,y)  σ(x,y)   ← Material property maps         │
│         ↓                                                    │
│  ┌─────────────────────┐                                     │
│  │  Leapfrog Wave Eq.  │  ← 0 learnable params              │
│  │  1754 steps, PML    │     Gradient flows through          │
│  └──────┬──────────────┘                                     │
│         ↓                                                    │
│  p(x,y,t) pressure field                                     │
│         ↓                                                    │
│  ┌─────────────────────┐                                     │
│  │  DAS Beamformer     │  ← 0 learnable params              │
│  └──────┬──────────────┘                                     │
│         ↓                                                    │
│  B-mode image (128×128)                                      │
│         ↓                                                    │
│  Loss: L1 + (1-SSIM) vs k-Wave GT                           │
└──────────────────────────────────────────────────────────────┘
```
**Style:** Clean vector diagram, blue/orange color scheme. Learnable block highlighted in orange, physics blocks in blue. Gradient arrow shown flowing backwards through entire pipeline.

### Figure 2: L0–L3 Taxonomy (Half Page)
**Content:** Four-panel conceptual diagram showing each level.
- **L0 (Data-only):** Neural network black box, CT→US direct mapping. Red ✗ for physics.
- **L1 (Physics-as-Loss):** Neural network + PDE residual loss branch. Yellow ⚠ for soft physics.
- **L2 (Physics-as-Architecture):** Network with built-in conservation structure. Yellow-green for partial physics.
- **L3 (Physics-as-Forward):** Small NN → full physics pipeline. Green ✓ for guaranteed physics.

Each panel shows: (a) schematic, (b) what is learned, (c) physics guarantee level.
Accompanying table with formal mathematical definitions.

### Figure 3: Antisymmetric Message Passing — Cross-Domain Transfer (Half Page)
**Content:** Side-by-side comparison:
- **Left:** Original DPC-GNN (solid mechanics): nodes = material points, edges = bonds, m_ij = −m_ji represents Newton's third law (F_ij = −F_ji)
- **Right:** DPC-GNN-Acoustic: nodes = spatial points, edges = neighbor connections, m_ij = −m_ji represents acoustic reciprocity
- **Center:** Shared mathematical structure, different physical interpretation

**Style:** Graph visualization with force/pressure vectors on edges. Color-coded to show antisymmetry.

### Figure 4: Convergence Comparison — V3 vs V4 (Half Page)
**Content:** 
- **Main plot:** SSIM vs. Epoch for V4 (Physics-as-Forward) and V3 (PINN-style)
  - V4: steep rise, SSIM=0.518 at epoch 2
  - V3: gradual rise, SSIM=0.520 at epoch 30
  - Annotation: "V4 matches V3 in 2/30 = 6.7% of training time"
- **Inset:** Same data, log-scale x-axis to emphasize early convergence
- **Second y-axis or subplot:** Training loss curves

**Style:** Publication-quality matplotlib, no grid clutter. V4 in bold orange, V3 in dashed blue.

### Figure 5: B-mode Image Comparison (Full Page)
**Content:** Multi-row comparison grid. Each row = different test case (3–4 cases).
Columns:
1. CT input (256×256)
2. k-Wave ground truth B-mode
3. V4 predicted B-mode (with SSIM annotation)
4. V3 predicted B-mode (with SSIM annotation)
5. Pix2Pix predicted B-mode (with SSIM annotation, if baseline available)
6. Error map (|GT − predicted|, jet colormap)

**Style:** Tight subplot grid, consistent windowing/contrast. SSIM values overlaid in corner of each prediction.

### Figure 6: Predicted Material Property Maps (Half Page)
**Content:** For a representative test case:
- Row 1: **Speed of sound c(x,y)**
  - (a) c_table(CT) — lookup table baseline
  - (b) c_residual(GNN) — learned correction (±150 m/s range, diverging colormap)
  - (c) c_total = c_table + c_residual — final prediction
  - (d) Ground truth c (from k-Wave phantom definition)
- Row 2: **Attenuation α(x,y)** — predicted vs. ground truth
- Row 3: **Reflectivity σ(x,y)** — predicted vs. ground truth

**Style:** Scientific colormaps (viridis for c, magma for α, coolwarm for residuals). Colorbars with physical units (m/s, dB/cm/MHz).

### Figure 7: Ablation Study Results (Half Page)
**Content:** Bar chart or table with SSIM for each ablation:
1. Full V4 model (baseline)
2. Without residual parameterization (direct c prediction)
3. Without antisymmetric MP (standard symmetric MP)
4. Without PML (reflective boundaries)
5. Fewer time steps (877 steps = half)
6. L1 loss only (no SSIM component)
7. Without attenuation (α=0)

Each bar annotated with ΔSSIM relative to full model.

**Style:** Horizontal bar chart, sorted by impact. Full model at top with dashed reference line.

### Figure 8: Computational Performance (Half Page)
**Content:** Two sub-figures:
- **(a) Speed comparison** (log-scale bar chart):
  - k-Wave (MATLAB, CPU): reference time
  - k-Wave (GPU): faster
  - V4 forward pass (GPU): target <100ms
  - DAS vectorized vs. loop-based: 13× annotation
- **(b) Scaling analysis:**
  - Inference time vs. grid resolution (128², 256², 512²)
  - Memory usage vs. time steps

**Style:** Log-scale where appropriate. Speedup factors annotated.

---

## 4. Supplementary Material Plan

### S1. Mathematical Details
- Full derivation of Leapfrog discretization
- PML formulation and parameter choices
- CFL stability analysis
- DAS beamforming equations

### S2. Additional Visualizations
- Pressure field snapshots at selected time steps (wave propagation animation frames)
- More test cases for B-mode comparison
- Training dynamics: material property maps at different epochs

### S3. Implementation Details
- Hyperparameter table
- Graph construction details
- Training hardware and time

### S4. Extended Ablations
- Sensitivity to CFL number
- Number of GNN message passing layers
- Residual bound sensitivity (±50, ±100, ±150, ±200 m/s)

---

## 5. Target Journals (Ranked)

### Tier 1 — Top Choice
**1. IEEE Transactions on Medical Imaging (TMI)**
- IF: ~10.6
- Fit: ★★★★★ — Core audience is medical image synthesis and ultrasound
- Why: CT-to-US synthesis is directly in scope; strong methods + clinical relevance
- Review time: 3–6 months
- **Recommended first submission target**

### Tier 2 — High Impact Alternatives
**2. Medical Image Analysis (MedIA)**
- IF: ~10.9
- Fit: ★★★★☆ — More methods-focused, less US-specific audience
- Why: Novel methodology (L0–L3 taxonomy) appeals to their audience
- Risk: May want more clinical validation

**3. Nature Communications**
- IF: ~16.6
- Fit: ★★★★☆ — Broad impact, paradigm-shifting framing
- Why: The L0–L3 taxonomy + cross-domain transfer story is general enough
- Risk: Need stronger experimental results (SSIM>0.9 required)
- Strategy: Frame as "physics-as-forward" general paradigm paper, acoustics as demonstration

### Tier 3 — Specialty / Backup
**4. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control (TUFFC)**
- IF: ~3.0
- Fit: ★★★★★ for ultrasound community, but lower impact
- Why: Differentiable ultrasound simulation is directly novel for this community

**5. Computer Methods in Applied Mechanics and Engineering (CMAME)**
- IF: ~6.9
- Fit: ★★★★☆ — GNN + physics simulation angle
- Why: DPC-GNN cross-domain transfer, differentiable simulation methodology

### Conference Alternative (Fast Publication)
**6. MICCAI 2026/2027** (if timing works)
- 8-page paper, fast review cycle
- Good for establishing priority while preparing journal version
- Strategy: Submit condensed version to MICCAI, full version to TMI

---

## 6. Key Experiments Still Needed

Before submission, the following must be completed:

| # | Experiment | Purpose | Priority |
|---|-----------|---------|----------|
| 1 | Train V4 to convergence (SSIM>0.85) | Core result | 🔴 Critical |
| 2 | L0 baselines (Pix2Pix, U-Net) | Comparison table | 🔴 Critical |
| 3 | All 7 ablations (Fig 7) | Ablation study | 🔴 Critical |
| 4 | Multiple test cases (≥5) | Generalization evidence | 🟡 Important |
| 5 | Inference speed benchmarking | Practical value claim | 🟡 Important |
| 6 | 3+ anatomical regions | Robustness | 🟡 Important |
| 7 | Statistical significance (5 random seeds) | Reviewer defense | 🟢 Nice-to-have |

---

## 7. Writing Timeline (Suggested)

| Week | Milestone |
|------|-----------|
| W1 | Complete V4 training to convergence; run L0 baselines |
| W2 | Run all ablations; generate all figure data |
| W3 | Draft Method section (§3) + generate Figures 1–3 |
| W4 | Draft Experiments (§4) + generate Figures 4–8 |
| W5 | Draft Introduction (§1) + Related Work (§2) |
| W6 | Draft Discussion (§5) + Conclusion (§6) + Abstract |
| W7 | Internal review, polish, supplementary materials |
| W8 | Submit to IEEE TMI |

---

*Plan created 2026-03-19. Update as experimental results evolve.*
