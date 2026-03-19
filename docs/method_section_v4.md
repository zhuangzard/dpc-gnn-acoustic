# Method Section Draft: DPC-GNN-Acoustic V4

## 3. Methodology

### 3.1 Physics-as-Forward Paradigm

We introduce the **Physics-as-Forward** paradigm for CT-to-ultrasound synthesis: the neural network learns *only* the material property mapping (CT → acoustic parameters), while wave dynamics are executed deterministically through the physical wave equation during the forward pass. This fundamentally differs from PINN-style approaches that incorporate physics as soft constraints in the loss function.

**Definition (Physics-as-Forward)**. A physics-informed neural network satisfies the Physics-as-Forward property if and only if:
1. All governing equations are executed as deterministic operations in the forward pass
2. The neural network outputs only material/constitutive parameters, never dynamical states
3. No physics-related terms appear in the training loss function
4. Physical residuals are identically zero up to numerical discretization error

This definition extends the DPC-GNN framework [our previous work] from solid mechanics (Neo-Hookean constitutive model + Leapfrog integration) to acoustic wave propagation.

### 3.2 Architecture Overview

The DPC-GNN-Acoustic model consists of three sequential modules:

$$\mathbf{I}_{US} = \mathcal{B}\left(\mathcal{P}\left(\mathcal{E}_\theta(\mathbf{I}_{CT}), \mathbf{s}(t)\right)\right)$$

where:
- $\mathcal{E}_\theta$: **GNN Material Encoder** (learnable, ~253K parameters) — maps CT image to acoustic parameter fields
- $\mathcal{P}$: **Deterministic Leapfrog Propagator** (0 learnable parameters) — executes the wave equation
- $\mathcal{B}$: **Deterministic DAS Beamformer** (0 learnable parameters) — forms the B-mode image
- $\theta$: the only trainable parameters in the entire model

### 3.3 GNN Material Encoder $\mathcal{E}_\theta$

Given a CT image $\mathbf{I}_{CT} \in \mathbb{R}^{N \times N}$ (with $N = 256$), the encoder predicts three spatially-varying acoustic parameter fields:

$$\mathcal{E}_\theta(\mathbf{I}_{CT}) = \left(c(\mathbf{x}),\; \alpha(\mathbf{x}),\; \sigma(\mathbf{x})\right)$$

where $c$ is the speed of sound [m/s], $\alpha$ is the attenuation coefficient [Np/m], and $\sigma$ is the scattering strength [dimensionless].

#### 3.3.1 Physics-Prior Structure

Rather than learning the CT-to-acoustic mapping from scratch, we decompose the speed of sound prediction into a known physical prior and a learned residual:

$$c(\mathbf{x}) = c_{\text{table}}(\text{HU}(\mathbf{x})) + \Delta c_\theta(\mathbf{x})$$

where $c_{\text{table}}$ is a piecewise-linear lookup table derived from established tissue acoustics literature:

$$c_{\text{table}}(\text{HU}) = \begin{cases}
343 & \text{HU} < -900 \text{ (air)} \\
1430 + 0.1375 \times (\text{HU} + 900) & -900 \leq \text{HU} < -100 \\
1540 + 0.6 \times \text{HU} & -100 \leq \text{HU} < 100 \\
1600 + 1.89 \times (\text{HU} - 100) & 100 \leq \text{HU} < 1000
\end{cases}$$

The residual $\Delta c_\theta$ captures spatial context effects, sub-resolution microstructure, and patient-specific variations that the lookup table cannot represent.

**Output constraints** ensure physical validity:
- $c(\mathbf{x}) \in [c_{\min}, c_{\max}] = [1400, 1700]$ m/s via sigmoid activation
- $\alpha(\mathbf{x}) \geq 0$ via softplus activation
- $\sigma(\mathbf{x}) \in [0, 1]$ via sigmoid activation

#### 3.3.2 CNN-GNN-CNN Architecture

The encoder follows a downsample-process-upsample structure:

1. **CNN Downsampler**: $\mathbf{I}_{CT} \in \mathbb{R}^{256 \times 256} \xrightarrow{\text{Conv}} \mathbf{F} \in \mathbb{R}^{64 \times 64 \times d}$
2. **GNN Processor**: $L = 5$ message passing layers operating on 4,096 nodes with $k = 8$ nearest neighbors
3. **CNN Upsampler**: $\mathbb{R}^{64 \times 64 \times d} \xrightarrow{\text{ConvTranspose}} \mathbb{R}^{256 \times 256 \times 3}$

#### 3.3.3 Antisymmetric Message Passing

Following the DPC-GNN framework, we employ antisymmetric message passing to enforce spatial consistency of the predicted parameter fields. For nodes $i$ and $j$, the message is computed as:

$$\mathbf{m}_{ij} = \tanh\left((\mathbf{h}_i - \mathbf{h}_j) \cdot \mathbf{W}_{\text{anti}}\right), \quad \mathbf{W}_{\text{anti}} = \mathbf{W} - \mathbf{W}^\top$$

This construction guarantees $\mathbf{m}_{ij} = -\mathbf{m}_{ji}$, which in the acoustic context enforces that the influence of node $i$ on $j$'s parameter estimate equals the influence of $j$ on $i$. This corresponds to the self-adjoint property of the wave equation's spatial operator.

**Comparison with DPC-GNN for soft tissue**:

| Property | DPC-GNN (Soft Tissue) | DPC-GNN-Acoustic |
|----------|----------------------|-------------------|
| Physical law | Newton's 3rd: $F_{ij} = -F_{ji}$ | Wave eq. self-adjointness |
| Predicted quantity | Deformation gradient | Acoustic parameters |
| Antisymmetry ensures | Momentum conservation | Spatial parameter consistency |
| Mathematical form | $\mathbf{W}_{\text{anti}} = \mathbf{W} - \mathbf{W}^\top$ | Same |

### 3.4 Deterministic Leapfrog Propagator $\mathcal{P}$

The wave equation with spatially-varying coefficients:

$$\frac{\partial^2 p}{\partial t^2} + 2\alpha(\mathbf{x}) \frac{\partial p}{\partial t} = c(\mathbf{x})^2 \nabla^2 p + \sigma(\mathbf{x}) s(t)$$

is discretized using the Leapfrog (central difference) scheme:

$$p^{n+1}_{i,j} = \frac{1}{1 + \alpha_{i,j}\frac{\Delta t}{2}} \left[ 2p^n_{i,j} - \left(1 - \alpha_{i,j}\frac{\Delta t}{2}\right) p^{n-1}_{i,j} + \Delta t^2 \left( c_{i,j}^2 \nabla^2_h p^n_{i,j} + \sigma_{i,j} s^n \right) \right]$$

where $\nabla^2_h$ is the discrete Laplacian:

$$\nabla^2_h p_{i,j} = \frac{p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j}}{\Delta x^2}$$

**Key properties**:
- **Zero learnable parameters**: The propagator is purely deterministic given $(c, \alpha, \sigma)$
- **Fully differentiable**: All operations (addition, multiplication, convolution) support autograd
- **Gradient checkpointing**: Intermediate states saved every 20 steps to manage memory
- **CFL stability**: Guaranteed by constraining $c \in [1400, 1700]$ m/s:

$$\text{CFL} = c_{\max} \frac{\Delta t}{\Delta x} \sqrt{2} = 1700 \times \frac{2 \times 10^{-8}}{2.34 \times 10^{-4}} \times \sqrt{2} = 0.206 < 1 \quad \checkmark$$

**PML Absorbing Boundary**: A 20-cell PML layer with cubic damping profile $d(x) = d_0 (x/L_{\text{PML}})^3$ eliminates boundary reflections.

### 3.5 Differentiable DAS Beamformer $\mathcal{B}$

Sensor data is extracted at the top boundary ($y = 0$) and processed through the standard ultrasound imaging pipeline, implemented entirely with differentiable operations:

1. **Delay-and-Sum (DAS)**: For each image pixel $(x_p, z_p)$ and sensor element $k$:
   $$I(x_p, z_p) = \frac{1}{K} \sum_{k=1}^{K} \text{RF}_k\left(\frac{d_{\text{tx}}(x_p, z_p) + d_{\text{rx}}(x_p, z_p, k)}{c_0}\right)$$
   where $d_{\text{tx}}$ and $d_{\text{rx}}$ are transmit and receive distances, and RF interpolation uses differentiable linear interpolation.

2. **Envelope Detection**: Hilbert transform via FFT:
   $$\text{env}(t) = |p(t) + j\mathcal{H}[p](t)| = |\mathcal{F}^{-1}[2 \cdot \mathbf{1}_{f>0} \cdot \mathcal{F}[p]]|$$

3. **Log Compression**:
   $$I_{\text{dB}} = 20 \log_{10}\left(\frac{\text{env}}{\max(\text{env})} + \epsilon\right), \quad \epsilon = 10^{-6}$$

### 3.6 Training Objective

The training loss operates exclusively on the final B-mode image:

$$\mathcal{L}(\theta) = \|\mathbf{I}^{\text{pred}}_{US} - \mathbf{I}^{\text{GT}}_{US}\|_1 + \left(1 - \text{SSIM}(\mathbf{I}^{\text{pred}}_{US}, \mathbf{I}^{\text{GT}}_{US})\right)$$

**Critically, there is no physics loss term.** The wave equation is satisfied by construction through the Leapfrog propagator. We monitor the physics residual:

$$r_{\text{phys}} = \left\| \frac{p^{n+1} - 2p^n + p^{n-1}}{\Delta t^2} - c^2 \nabla^2_h p^n + \alpha \frac{p^{n+1} - p^{n-1}}{2\Delta t} - \sigma s^n \right\|_2$$

as a verification metric (not a training signal). By construction, $r_{\text{phys}} = O(\Delta t^2 + \Delta x^2)$, confirming physical self-consistency.

**Proposition 1 (Physics-weight optimality)**. For the DPC-GNN-Acoustic architecture, any nonzero physics weight $\lambda > 0$ in the loss:
$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda \cdot r_{\text{phys}}^2$$
is redundant, because $r_{\text{phys}}$ is identically at numerical precision regardless of $\theta$. The gradient $\nabla_\theta (\lambda \cdot r_{\text{phys}}^2) \approx \mathbf{0}$ contributes only numerical noise to the optimization. Therefore, $\lambda = 0$ is not a hyperparameter choice but an architectural necessity.

### 3.7 Relationship to DPC-GNN for Soft Tissue

| Aspect | DPC-GNN (Soft Tissue) | DPC-GNN-Acoustic |
|--------|----------------------|-------------------|
| Domain | Solid mechanics | Wave propagation |
| GNN learns | Material properties (μ, λ) | Acoustic parameters (c, α, σ) |
| Physics engine | Neo-Hookean + Leapfrog | Wave equation + Leapfrog |
| Conservation law | Newton's 3rd (F_ij = -F_ji) | Acoustic reciprocity |
| MP structure | Antisymmetric | Antisymmetric |
| Physics in loss? | **No** | **No** |
| Learnable dynamics? | **No** | **No** |

This parallel structure demonstrates that the **Physics-as-Forward paradigm is domain-agnostic**: the same architectural principle — "network learns constitutive parameters, physics executes dynamics" — applies across fundamentally different physical domains.
