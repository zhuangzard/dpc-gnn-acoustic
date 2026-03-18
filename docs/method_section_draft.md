# Method Section Draft: DPC-GNN-Acoustic

This document provides a draft for the Method section of the DPC-GNN-Acoustic paper, formatted for direct inclusion in academic manuscripts.

---

## 3. Methodology

### 3.1 Acoustic Wave Propagation

Acoustic wave propagation in biological tissue is governed by the second-order wave equation derived from linearized fluid dynamics. Let $\Omega \subset \mathbb{R}^3$ denote the spatial domain representing the tissue volume, and $p: \Omega \times [0, T] \rightarrow \mathbb{R}$ denote the acoustic pressure field. The governing equation is:

$$
\frac{\partial^2 p(\mathbf{x}, t)}{\partial t^2} = c(\mathbf{x})^2 \nabla^2 p(\mathbf{x}, t) + s(\mathbf{x}, t)
$$

where $c(\mathbf{x})$ is the spatially varying speed of sound and $s(\mathbf{x}, t)$ represents acoustic sources. For ultrasound simulation, we consider the initial value problem with:

$$
\begin{aligned}
p(\mathbf{x}, 0) &= p_0(\mathbf{x}) \\
\frac{\partial p}{\partial t}(\mathbf{x}, 0) &= v_0(\mathbf{x})
\end{aligned}
$$

**Heterogeneous Medium**. In realistic tissue modeling, the acoustic properties vary spatially. The wave equation generalizes to:

$$
\frac{\partial^2 p}{\partial t^2} = c(\mathbf{x})^2 \nabla^2 p + \nabla \ln \rho(\mathbf{x}) \cdot \nabla p
$$

where $\rho(\mathbf{x})$ is the tissue density. This accounts for both speed of sound variations and acoustic impedance mismatches at tissue interfaces.

**Boundary Conditions**. For computational tractability, we employ absorbing boundary conditions via the Perfectly Matched Layer (PML) technique:

$$
\frac{\partial^2 p}{\partial t^2} + 2\sigma(\mathbf{x}) \frac{\partial p}{\partial t} + \sigma(\mathbf{x})^2 p = c^2 \nabla^2 p
$$

where $\sigma(\mathbf{x})$ is a smoothly increasing damping coefficient within the PML region that eliminates artificial reflections.

### 3.2 Graph-based Spatial Discretization

Rather than employing traditional grid-based discretization, we represent the continuous domain as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes $\mathbf{x}_i \in \mathcal{V}$ correspond to spatial sampling points and edges $(i, j) \in \mathcal{E}$ encode local connectivity.

**Node Features**. Each node $i$ maintains the following state variables:
- $p_i \in \mathbb{R}$: acoustic pressure
- $\mathbf{v}_i \in \mathbb{R}^3$: particle velocity vector
- $c_i \in \mathbb{R}$: local speed of sound
- $\rho_i \in \mathbb{R}$: local density

**Graph Construction**. The graph topology is determined by the geometry of the computational domain. For Cartesian grids, we employ a 6-connected neighborhood in 3D (or 4-connected in 2D). For unstructured meshes derived from CT data, connectivity follows the mesh topology with edge weights:

$$
w_{ij} = \frac{1}{\|\mathbf{x}_i - \mathbf{x}_j\|^2}
$$

**Graph Laplacian Operator**. The continuous Laplacian $\nabla^2$ is approximated by the weighted graph Laplacian $\mathbf{L} \in \mathbb{R}^{N \times N}$:

$$
(\mathbf{L} \mathbf{p})_i = \sum_{j \in \mathcal{N}(i)} w_{ij} (p_j - p_i)
$$

where $\mathcal{N}(i)$ denotes the neighborhood of node $i$. In matrix form:

$$
\mathbf{L} = \mathbf{D} - \mathbf{W}
$$

with $\mathbf{W}$ being the weighted adjacency matrix and $\mathbf{D} = \text{diag}(\sum_j w_{ij})$ the degree matrix.

**Consistency Analysis**. For uniform grid spacing $h$, the graph Laplacian approximates the continuous operator with second-order accuracy:

$$
(\mathbf{L} \mathbf{p})_i = \nabla^2 p(\mathbf{x}_i) + \mathcal{O}(h^2)
$$

### 3.3 Hard Physics Constraints via Message Passing

The core innovation of DPC-GNN-Acoustic lies in embedding the wave equation directly into the message passing architecture, transforming the GNN into a **physics-constrained numerical integrator**.

**Wave Equation Message Passing (WaveEquation-MP)**. Each message passing step computes the next pressure state according to the discrete wave equation:

$$
\mathbf{p}^{n+1} = 2\mathbf{p}^n - \mathbf{p}^{n-1} - \Delta t^2 \cdot \mathbf{C} \mathbf{L} \mathbf{p}^n
$$

where $\mathbf{C} = \text{diag}(c_1^2, \ldots, c_N^2)$ encodes spatially varying sound speeds. This is implemented as:

$$
\boxed{
\begin{aligned}
\mathbf{m}_{j \rightarrow i} &= w_{ij} (p_j^n - p_i^n) \cdot c_i^2 \Delta t^2 \\
\mathbf{p}_i^{n+1} &= 2p_i^n - p_i^{n-1} - \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \rightarrow i}
\end{aligned}
}
\tag{1}
$$

**Architecture Design**. The message passing layer (Eq. 1) is implemented as a custom PyTorch module with the following structure:

```python
# Pseudo-code representation
class WaveEquationMP(MessagePassing):
    def forward(self, p_curr, p_prev, dt):
        # Aggregate messages: computes (L * p)
        laplacian_p = self.propagate(edge_index, x=p_curr, edge_weight=weights)
        # Apply wave equation update
        p_next = 2 * p_curr - p_prev - dt**2 * c_squared * laplacian_p
        return p_next
```

**Equivariance Properties**. The WaveEquation-MP layer is:
1. **Translation equivariant**: Shifting the input shifts the output
2. **Rotation equivariant** (with appropriate edge weight construction)
3. **Permutation equivariant**: Node ordering does not affect results

These properties ensure that the learned dynamics respect the underlying physical symmetries.

### 3.4 Differentiable Time Integration

DPC-GNN-Acoustic implements a **fully differentiable time integration scheme** that enables end-to-end training through the temporal domain. This is achieved by unrolling the wave equation update over $K$ time steps.

**Unrolled Computation Graph**. Given initial conditions $(\mathbf{p}^0, \mathbf{p}^1)$, the network computes:

$$
\mathbf{p}^{k+1} = \text{WaveEquationMP}(\mathbf{p}^k, \mathbf{p}^{k-1}; \theta), \quad k = 1, \ldots, K-1
$$

where $\theta$ represents learnable parameters (e.g., material properties, source terms).

**Memory-Efficient Backpropagation**. For long trajectories, we employ checkpointing to trade computation for memory:

$$
\text{Memory} = \mathcal{O}(N \sqrt{K}) \quad \text{instead of} \quad \mathcal{O}(NK)
$$

**Gradient Flow Analysis**. The Jacobian of the leapfrog update is:

$$
\frac{\partial \mathbf{p}^{n+1}}{\partial \mathbf{p}^n} = 2\mathbf{I} - \Delta t^2 \mathbf{C} \mathbf{L}
$$

The eigenvalues of this Jacobian determine gradient stability during backpropagation through time (BPTT). Under the CFL condition, eigenvalues remain bounded, ensuring stable gradient flow.

### 3.5 Loss Functions and Training

**Physics-Informed Loss**. The training objective combines multiple terms:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}
$$

**Data Fidelity Term** (supervised learning):

$$
\mathcal{L}_{\text{data}} = \frac{1}{K} \sum_{k=1}^{K} \|\mathbf{p}^k - \mathbf{p}^k_{\text{ground truth}}\|_2^2
$$

**Physics Consistency Term**:

$$
\mathcal{L}_{\text{phys}} = \frac{1}{K-1} \sum_{k=1}^{K-1} \left\| \frac{\mathbf{p}^{k+1} - 2\mathbf{p}^k + \mathbf{p}^{k-1}}{\Delta t^2} + \mathbf{C} \mathbf{L} \mathbf{p}^k \right\|_2^2
$$

This term vanishes identically for the WaveEquation-MP layer, serving as a validation of the hard constraint implementation.

**Boundary Condition Term**:

$$
\mathcal{L}_{\text{bc}} = \frac{1}{|\partial\Omega|} \sum_{i \in \partial\Omega} |p_i - g_D(\mathbf{x}_i)|^2 + \frac{1}{|\partial\Omega|} \sum_{i \in \partial\Omega} \left| \frac{\partial p}{\partial \mathbf{n}}_i - g_N(\mathbf{x}_i) \right|^2
$$

**Energy Conservation Regularization** (optional):

$$
\mathcal{L}_{\text{energy}} = |E^{K} - E^{0}|
$$

where $E^k = \frac{1}{2} \mathbf{p}^k \cdot \mathbf{M} \mathbf{p}^k + \frac{1}{2} \dot{\mathbf{p}}^k \cdot \mathbf{M} \dot{\mathbf{p}}^k$ is the discrete total energy.

### 3.6 CT-to-Ultrasound Synthesis Pipeline

The complete inference pipeline for real-time CT-to-US synthesis comprises:

**Step 1: Preprocessing**
- Extract acoustic properties from CT Hounsfield units:
  $$
  c(\text{HU}) = c_{\text{water}} + \alpha \cdot \text{HU}
  $$
  $$
  \rho(\text{HU}) = \rho_{\text{water}} + \beta \cdot \text{HU}
  $$

**Step 2: Graph Construction**
- Subsample CT volume to simulation resolution
- Construct graph with edge weights based on Euclidean distance
- Identify source positions (transducer locations)

**Step 3: Wave Propagation**
- Initialize pressure field with impulse response
- Iterate WaveEquation-MP for $K$ time steps
- Record pressure at receiver positions (A-line synthesis)

**Step 4: Image Formation**
- Apply envelope detection and log compression
- Map to B-mode image space

---

## Key Equations Summary

| Equation | Description | Reference |
|----------|-------------|-----------|
| (1) | WaveEquation Message Passing | Section 3.3 |
| (2) | CFL Stability Condition | Section 3.2 |
| (3) | Discrete Energy Conservation | Section 3.5 |

---

*This draft is intended for integration into the main manuscript.*
