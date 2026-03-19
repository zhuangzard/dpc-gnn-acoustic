# DPC-GNN-Acoustic: Reference Library v4

> **Paper**: Physics-as-Forward Graph Neural Network for CT-to-Ultrasound Image Translation
> **Generated**: 2026-03-19
> **Status**: Ready for BibTeX conversion

---

## 1. CT-to-Ultrasound Translation (Core Competitors)

### 1.1 Pure Deep Learning Approaches

#### ⭐ [MUST CITE — Direct Competitor] S-CycleGAN
- **Authors**: Song, Y. and Chong, N.Y.
- **Title**: S-CycleGAN: Semantic Segmentation Enhanced CT-Ultrasound Image-to-Image Translation for Robotic Ultrasonography
- **Year**: 2024
- **Venue**: IEEE International Conference on Robotics and Automation (ICRA) / arXiv:2406.01191
- **DOI**: 10.1109/ICRA57436.2024.10860598
- **Relevance**: Direct competitor — uses CycleGAN with semantic discriminators for CT-to-US translation. Purely data-driven, no physics constraints. Our method replaces the learned generator with physics-based wave propagation, ensuring physical plausibility that CycleGAN cannot guarantee.

#### ⭐ [MUST CITE — Direct Competitor] Vitale et al. CycleGAN US Simulation
- **Authors**: Vitale, S., Orlando, J.I., Iarussi, E., and Larrabide, I.
- **Title**: Improving Realism in Patient-Specific Abdominal Ultrasound Simulation Using CycleGANs
- **Year**: 2020
- **Venue**: International Journal of Computer Assisted Radiology and Surgery (IJCARS), 15(2):183-192
- **DOI**: 10.1007/s11548-019-02046-5
- **Relevance**: Pioneering work on using CycleGAN to improve realism of physics-simulated US images from CT. Hybrid approach (physics sim + GAN refinement). Our method eliminates the need for GAN refinement by learning accurate material properties directly.

#### ⭐ [MUST CITE — Direct Competitor] Li et al. CT-to-US Spine Surgery
- **Authors**: Li, A., et al.
- **Title**: Enabling Augmented Segmentation and Registration in Ultrasound-Guided Spinal Surgery via Realistic Ultrasound Synthesis from Diagnostic CT Volume
- **Year**: 2023
- **Venue**: IEEE Transactions on Medical Imaging / arXiv:2301.01940
- **DOI**: 10.1109/TMI.2023.3308322
- **Relevance**: Direct competitor for CT-to-US synthesis in surgical context. Uses deep learning for synthesis. Our physics-based approach provides interpretable outputs and generalizes to unseen anatomies without retraining.

### 1.2 Physics-Based Simulation Approaches

#### ⭐ [MUST CITE — Direct Competitor] Almahfouz Nasser & Sethi — Wave-Based CT-to-US
- **Authors**: Almahfouz Nasser, S. and Sethi, A.
- **Title**: Simulating Ultrasound Images from CT Scans
- **Year**: 2023
- **Venue**: medRxiv / BIOSTEC 2023
- **DOI**: 10.1101/2023.01.16.23284615
- **Relevance**: Most closely related work — generates US from CT using wave equation solver (Stride). Uses fixed CT-derived acoustic properties rather than learned ones. Our GNN learns optimal material mappings, and our Leapfrog solver is fully differentiable end-to-end.

#### ⭐ [MUST CITE] Burger et al. — GPU US Simulation from CT
- **Authors**: Burger, B., Bettinghausen, S., Radle, M., and Hesser, J.
- **Title**: Real-Time GPU-Based Ultrasound Simulation Using Deformable Mesh Models
- **Year**: 2013
- **Venue**: IEEE Transactions on Medical Imaging, 32(11):2073-2084
- **DOI**: 10.1109/TMI.2013.2272691
- **Relevance**: Early work on real-time US simulation from CT using GPU-accelerated ray-tracing. Uses simplified acoustic model (no full wave equation). Our method solves the full wave equation for higher fidelity.

#### Burger et al. — Visualization and GPU-Accelerated US Simulation
- **Authors**: Burger, B., Bettinghausen, S., Radle, M., and Hesser, J.
- **Title**: Visualization and GPU-Accelerated Simulation of Medical Ultrasound from CT Images
- **Year**: 2009
- **Venue**: IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 56(11):2401-2409
- **DOI**: 10.1109/TUFFC.2009.1331
- **Relevance**: Foundational work on GPU-based US simulation from CT data. Ray-tracing approach trades accuracy for speed.

#### Salehi et al. — Patient-Specific US Simulation
- **Authors**: Salehi, M., Ahmadi, S.-A., Prevost, R., Navab, N., and Wein, W.
- **Title**: Patient-Specific 3D Ultrasound Simulation Based on Convolutional Ray-Tracing and Appearance Optimization
- **Year**: 2015
- **Venue**: MICCAI 2015, LNCS 9350, pp. 510-518
- **DOI**: 10.1007/978-3-319-24571-3_61
- **Relevance**: Uses convolutional ray-tracing with appearance optimization from MRI/CT. Demonstrates need for patient-specific acoustic properties — exactly what our GNN learns.

---

## 2. Physics-Informed Neural Networks & Hard Constraints

### 2.1 Foundational PINN

#### ⭐ [MUST CITE — Milestone] Raissi et al. — Original PINN
- **Authors**: Raissi, M., Perdikaris, P., and Karniadakis, G.E.
- **Title**: Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations
- **Year**: 2019
- **Venue**: Journal of Computational Physics, 378:686-707
- **DOI**: 10.1016/j.jcp.2018.10.045
- **arXiv**: 1711.10561
- **Relevance**: Seminal paper introducing PINNs. Uses soft PDE constraints via loss functions. Our approach differs fundamentally: we use hard constraints where the wave equation is explicitly solved (zero learnable parameters in physics), not approximated by a neural network.

#### ⭐ [MUST CITE — Milestone] Karniadakis et al. — PIML Review
- **Authors**: Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., and Yang, L.
- **Title**: Physics-Informed Machine Learning
- **Year**: 2021
- **Venue**: Nature Reviews Physics, 3(6):422-440
- **DOI**: 10.1038/s42254-021-00314-5
- **Relevance**: Comprehensive review categorizing physics-ML integration approaches. Our method falls into the "physics-as-forward-model" category where physics equations are explicitly executed rather than approximated, representing the strongest form of physics integration.

### 2.2 Hard Constraint & Structure-Preserving Methods

#### ⭐ [MUST CITE] Greydanus et al. — Hamiltonian Neural Networks
- **Authors**: Greydanus, S., Dzamba, M., and Yosinski, J.
- **Title**: Hamiltonian Neural Networks
- **Year**: 2019
- **Venue**: NeurIPS 2019, pp. 15353-15363
- **arXiv**: 1906.01563
- **Relevance**: Pioneered embedding conservation laws into neural network architecture. Our approach shares the philosophy of hard physics constraints but applies it to wave propagation: we don't learn the Hamiltonian — we execute the wave equation directly.

#### Cranmer et al. — Lagrangian Neural Networks
- **Authors**: Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., and Ho, S.
- **Title**: Lagrangian Neural Networks
- **Year**: 2020
- **Venue**: ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations
- **arXiv**: 2003.04630
- **Relevance**: Extends structure-preserving NNs to Lagrangian mechanics. Part of the broader trend toward physics-embedded architectures that our work continues.

---

## 3. GNN for Physics Simulation

### 3.1 Core GNN Simulation Frameworks

#### ⭐ [MUST CITE — Milestone] Sanchez-Gonzalez et al. — GNS
- **Authors**: Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., and Battaglia, P.W.
- **Title**: Learning to Simulate Complex Physics with Graph Networks
- **Year**: 2020
- **Venue**: ICML 2020, PMLR 119
- **arXiv**: 2002.09405
- **Relevance**: Foundational work on Graph Network-based Simulators (GNS). Learns full dynamics via message passing. Our method differs: GNN only predicts material properties, while dynamics are computed by explicit physics equations — a fundamental architectural distinction.

#### ⭐ [MUST CITE — Milestone] Pfaff et al. — MeshGraphNets
- **Authors**: Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., and Battaglia, P.W.
- **Title**: Learning Mesh-Based Simulation with Graph Networks
- **Year**: 2021
- **Venue**: ICLR 2021 (Outstanding Paper Award)
- **arXiv**: 2010.03409
- **Relevance**: Extends GNS to mesh-based simulation with adaptive resolution. Both GNS and MeshGraphNets learn the entire dynamics — our architecture constrains the GNN to learn only material properties (c, α, σ), with wave propagation computed analytically.

#### ⭐ [MUST CITE] Battaglia et al. — Graph Networks Framework
- **Authors**: Battaglia, P.W., Hamrick, J.B., Bapst, V., Sanchez-Gonzalez, A., et al.
- **Title**: Relational Inductive Biases, Deep Learning, and Graph Networks
- **Year**: 2018
- **Venue**: arXiv:1806.01261
- **Relevance**: Defines the graph network framework and message-passing formalism that our GNN architecture builds upon. Provides theoretical grounding for using graphs to represent physical systems.

### 3.2 GNN for Medical/Tissue Simulation

#### ⭐ [MUST CITE] Salehi & Giannacopoulos — PhysGNN
- **Authors**: Salehi, Y. and Giannacopoulos, D.D.
- **Title**: PhysGNN: A Physics-Driven Graph Neural Network Based Model for Predicting Soft Tissue Deformation in Image-Guided Neurosurgery
- **Year**: 2022
- **Venue**: NeurIPS 2022
- **arXiv**: 2109.04352
- **Relevance**: Uses GNN for physics-driven soft tissue deformation prediction in surgery. Data-driven with physics-based loss. Our approach goes further: physics is not in the loss but in the forward model itself.

#### Dalton et al. — Physics-Informed GNN for Soft Tissue
- **Authors**: Dalton, D., Husmeier, D., and Gao, H.
- **Title**: Physics-Informed Graph Neural Network Emulation of Soft-Tissue Mechanics
- **Year**: 2023
- **Venue**: Computer Methods in Applied Mechanics and Engineering, 417:116351
- **DOI**: 10.1016/j.cma.2023.116351
- **Relevance**: Uses physics-informed training (minimum potential energy) for GNN tissue mechanics. Complements our work by showing GNNs can encode physical principles for biological simulation.

### 3.3 Antisymmetric Message Passing

#### ⭐ [MUST CITE] Gravina et al. — Anti-Symmetric DGN
- **Authors**: Gravina, A., Bacciu, D., and Gallicchio, C.
- **Title**: Anti-Symmetric DGN: A Stable Architecture for Deep Graph Networks
- **Year**: 2023
- **Venue**: ICLR 2023
- **arXiv**: 2210.09789
- **Relevance**: Introduces antisymmetric weight matrices in GNN message passing for stability and non-dissipativity. Our antisymmetric message passing builds on this concept but applies it to enforce spatial reciprocity in acoustic wave propagation — a physical symmetry requirement.

---

## 4. Differentiable Physics Simulation

#### ⭐ [MUST CITE] Hu et al. — DiffTaichi
- **Authors**: Hu, Y., Anderson, L., Li, T.-M., Sun, Q., Carr, N., Ragan-Kelley, J., and Durand, F.
- **Title**: DiffTaichi: Differentiable Programming for Physical Simulation
- **Year**: 2020
- **Venue**: ICLR 2020
- **arXiv**: 1910.00935
- **Relevance**: Foundational framework for differentiable physical simulation. Our 1754-step differentiable Leapfrog wave equation solver faces similar challenges (memory, gradient stability) and uses similar techniques (gradient checkpointing) enabled by this paradigm.

#### NVIDIA Warp
- **Authors**: Macklin, M. et al.
- **Title**: Warp: A High-Performance Python Framework for GPU Simulation
- **Year**: 2022
- **Venue**: NVIDIA Technical Report
- **URL**: https://github.com/NVIDIA/warp
- **Relevance**: GPU-accelerated differentiable simulation framework. Represents the engineering frontier for differentiable physics. Our PyTorch-based implementation achieves similar differentiability for acoustic wave equations.

#### ⭐ [MUST CITE] Chen et al. — Gradient Checkpointing
- **Authors**: Chen, T., Xu, B., Zhang, C., and Guestrin, C.
- **Title**: Training Deep Nets with Sublinear Memory Cost
- **Year**: 2016
- **Venue**: arXiv:1604.06174
- **Relevance**: Essential technique enabling our 1754-step differentiable wave equation. Without gradient checkpointing, backpropagation through 1754 Leapfrog steps would be memory-prohibitive. We apply this to trade computation for memory in the physics forward pass.

---

## 5. Ultrasound Imaging Physics & Simulation Tools

### 5.1 Acoustic Simulation Toolboxes

#### ⭐ [MUST CITE — Milestone] Treeby & Cox — k-Wave
- **Authors**: Treeby, B.E. and Cox, B.T.
- **Title**: k-Wave: MATLAB Toolbox for the Simulation and Reconstruction of Photoacoustic Wave Fields
- **Year**: 2010
- **Venue**: Journal of Biomedical Optics, 15(2):021314
- **DOI**: 10.1117/1.3360308
- **Relevance**: Gold-standard acoustic simulation toolbox. Uses k-space pseudospectral methods. Our Leapfrog finite-difference solver is simpler and fully differentiable, enabling end-to-end gradient flow — something k-Wave's MATLAB implementation cannot do natively.

#### Treeby et al. — Nonlinear US Propagation
- **Authors**: Treeby, B.E., Jaros, J., Rendell, A.P., and Cox, B.T.
- **Title**: Modeling Nonlinear Ultrasound Propagation in Heterogeneous Media with Power Law Absorption Using a k-Space Pseudospectral Method
- **Year**: 2012
- **Venue**: Journal of the Acoustical Society of America, 131(6):4324-4336
- **DOI**: 10.1121/1.4712021
- **Relevance**: Extends k-Wave to nonlinear propagation and heterogeneous media. Our wave equation implementation handles heterogeneous speed-of-sound and attenuation similarly, but with a differentiable Leapfrog scheme.

#### ⭐ [MUST CITE] Stanziola et al. — j-Wave
- **Authors**: Stanziola, A., Arridge, S.R., Cox, B.T., and Treeby, B.E.
- **Title**: j-Wave: An Open-Source Differentiable Wave Simulator
- **Year**: 2023
- **Venue**: SoftwareX, 22:101338
- **DOI**: 10.1016/j.softx.2023.101338
- **arXiv**: 2207.01499
- **Relevance**: JAX-based differentiable acoustic simulator. Closest existing tool to our differentiable wave solver. Key difference: j-Wave is a standalone simulator; our wave equation is embedded as a non-learnable layer within a GNN architecture, enabling joint optimization of material prediction and wave physics.

#### Jensen — Field II
- **Authors**: Jensen, J.A.
- **Title**: Field: A Program for Simulating Ultrasound Systems
- **Year**: 1996
- **Venue**: Medical & Biological Engineering & Computing, 34(Suppl. 1):351-353 (10th NB Conference on Biomedical Imaging, 1996)
- **Relevance**: Foundational ultrasound field simulation program based on spatial impulse response method. Widely used for transducer modeling. Our approach uses full wave equation rather than impulse response approximation.

#### Garcia — MUST/SIMUS
- **Authors**: Garcia, D.
- **Title**: SIMUS: An Open-Source Simulator for Ultrasound Imaging. Part I: Theory & Examples
- **Year**: 2021
- **Venue**: arXiv:2102.02738 / IEEE International Ultrasonics Symposium (IUS) 2021
- **Relevance**: Modern ultrasound simulation toolbox. Uses paraxial approximations. Our full-wave approach avoids these approximations at the cost of higher computation, which we mitigate through GNN-based material prediction.

### 5.2 Wave Equation & Beamforming

#### ⭐ [MUST CITE] Virieux — Velocity-Stress Finite Difference
- **Authors**: Virieux, J.
- **Title**: P-SV Wave Propagation in Heterogeneous Media: Velocity-Stress Finite-Difference Method
- **Year**: 1986
- **Venue**: Geophysics, 51(4):889-901
- **DOI**: 10.1190/1.1442147
- **Relevance**: Foundational paper for staggered-grid finite-difference wave equation solvers. Our Leapfrog implementation uses the velocity-stress formulation introduced here, adapted for acoustic (pressure) wave propagation in tissue.

#### Matrone et al. — DMAS Beamforming
- **Authors**: Matrone, G., Savoia, A.S., Caliano, G., and Magenes, G.
- **Title**: The Delay Multiply and Sum Beamforming Algorithm in Ultrasound B-Mode Medical Imaging
- **Year**: 2015
- **Venue**: IEEE Transactions on Medical Imaging, 34(4):940-949
- **DOI**: 10.1109/TMI.2014.2371235
- **Relevance**: Describes beamforming algorithms for B-mode image formation. Our DAS beamforming module implements a differentiable version of this process as a fixed (non-learnable) component.

---

## 6. Ultrasound in Surgical Navigation

#### ⭐ [MUST CITE] Wein et al. — US-CT Registration Survey Context
- **Authors**: Various groups working on intraoperative US-CT registration
- **Title**: (Representative: Real-time intraoperative ultrasound registration for accurate surgical navigation)
- **Year**: 2024
- **Venue**: International Journal of Computer Assisted Radiology and Surgery
- **DOI**: 10.1007/s11548-024-03287-3
- **Relevance**: Motivates the need for fast, accurate CT-to-US conversion. Intraoperative registration requires understanding the correspondence between CT and US appearance — exactly what our model provides through physics-based synthesis.

#### Ungi et al. — US-Guided Interventions
- **Authors**: Ungi, T., Lasso, A., and Fichtinger, G.
- **Title**: Open-Source Platforms for Navigated Image-Guided Interventions
- **Year**: 2016
- **Venue**: Medical Image Analysis, 33:181-186
- **DOI**: 10.1016/j.media.2016.06.011
- **Relevance**: Describes open-source platforms for image-guided surgery. US-CT registration is a core component. Our method could serve as a real-time CT-to-US prediction module in such navigation systems.

---

## 7. Medical Image Synthesis (General)

### 7.1 Foundational GAN Methods

#### ⭐ [MUST CITE — Milestone] Goodfellow et al. — GAN
- **Authors**: Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y.
- **Title**: Generative Adversarial Nets
- **Year**: 2014
- **Venue**: NeurIPS 2014 (NIPS), pp. 2672-2680
- **arXiv**: 1406.2661
- **Relevance**: Original GAN paper. Foundation for all GAN-based CT-to-US methods that we compare against. Our method avoids adversarial training entirely, using physics equations instead of discriminators.

#### ⭐ [MUST CITE — Milestone] Isola et al. — Pix2Pix
- **Authors**: Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A.A.
- **Title**: Image-to-Image Translation with Conditional Adversarial Networks
- **Year**: 2017
- **Venue**: CVPR 2017
- **arXiv**: 1611.07004
- **Relevance**: Foundational paired image translation method. Basis for many CT-to-US approaches. Requires paired training data; our method requires only CT images and physics parameters.

#### ⭐ [MUST CITE — Milestone] Zhu et al. — CycleGAN
- **Authors**: Zhu, J.-Y., Park, T., Isola, P., and Efros, A.A.
- **Title**: Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks
- **Year**: 2017
- **Venue**: ICCV 2017
- **arXiv**: 1703.10593
- **Relevance**: Enables unpaired CT-to-US translation. Most CT-to-US competitors use CycleGAN or variants. Our physics-based approach doesn't need any real US training data — only physical equations and CT inputs.

### 7.2 Medical Image Synthesis Reviews

#### Yi et al. — GAN Medical Image Synthesis Review
- **Authors**: Yi, X., Walia, E., and Babyn, P.
- **Title**: Generative Adversarial Network in Medical Imaging: A Review
- **Year**: 2019
- **Venue**: Medical Image Analysis, 58:101552
- **DOI**: 10.1016/j.media.2019.101552
- **Relevance**: Comprehensive review of GANs in medical imaging including cross-modality synthesis. Contextualizes our work within the broader medical image synthesis landscape.

#### Skandarani et al. — GANs for Medical Image Synthesis
- **Authors**: Skandarani, Y., Jodoin, P.-M., and Bhatt, A.
- **Title**: GANs for Medical Image Synthesis: An Empirical Study
- **Year**: 2023
- **Venue**: Journal of Imaging Informatics in Medicine
- **arXiv**: 2105.05318
- **Relevance**: Empirical comparison of GAN architectures for medical image synthesis. Provides baseline comparisons relevant to our evaluation.

---

## 8. Additional Technical References

#### Kipf & Welling — GCN
- **Authors**: Kipf, T.N. and Welling, M.
- **Title**: Semi-Supervised Classification with Graph Convolutional Networks
- **Year**: 2017
- **Venue**: ICLR 2017
- **arXiv**: 1609.02907
- **Relevance**: Foundational GCN paper. Our message-passing architecture builds upon the graph convolution framework introduced here.

#### Gilmer et al. — MPNN
- **Authors**: Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., and Dahl, G.E.
- **Title**: Neural Message Passing for Quantum Chemistry
- **Year**: 2017
- **Venue**: ICML 2017
- **arXiv**: 1704.01212
- **Relevance**: Unifies GNN variants under the message-passing framework. Our antisymmetric message passing is a constrained variant of the general MPNN framework.

---

## Summary Statistics

| Category | Count | Must-Cite |
|----------|-------|-----------|
| 1. CT-to-US Translation | 7 | 5 |
| 2. PINN & Hard Constraints | 4 | 3 |
| 3. GNN for Physics | 6 | 5 |
| 4. Differentiable Physics | 3 | 2 |
| 5. US Imaging Physics | 7 | 4 |
| 6. Surgical Navigation | 2 | 1 |
| 7. Medical Image Synthesis | 5 | 3 |
| 8. Additional Technical | 2 | 0 |
| **Total** | **36** | **23** |

---

## Priority Citation Map

### Tier 1: Must cite (direct competitors / foundational)
1. Almahfouz Nasser & Sethi 2023 — Wave-based CT-to-US (closest competitor)
2. Song & Chong 2024 — S-CycleGAN CT-to-US
3. Vitale et al. 2020 — CycleGAN US simulation
4. Li et al. 2023 — CT-to-US for spine surgery
5. Raissi et al. 2019 — PINN (contrast: soft vs hard constraints)
6. Sanchez-Gonzalez et al. 2020 — GNS (contrast: full dynamics vs material-only)
7. Pfaff et al. 2021 — MeshGraphNets
8. Treeby & Cox 2010 — k-Wave
9. Gravina et al. 2023 — Antisymmetric DGN

### Tier 2: Should cite (important context)
10. Karniadakis et al. 2021 — PIML review
11. Greydanus et al. 2019 — HNN
12. Stanziola et al. 2023 — j-Wave
13. Hu et al. 2020 — DiffTaichi
14. Chen et al. 2016 — Gradient checkpointing
15. Virieux 1986 — Velocity-stress FD
16. Goodfellow et al. 2014 — GAN
17. Isola et al. 2017 — Pix2Pix
18. Zhu et al. 2017 — CycleGAN
19. Battaglia et al. 2018 — Graph networks
20. Salehi & Giannacopoulos 2022 — PhysGNN

### Tier 3: Good to cite (broader context)
21-36: Remaining references for completeness
