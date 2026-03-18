# Mathematical Foundations of DPC-GNN-Acoustic

This document provides the rigorous mathematical foundations underlying the Differentiable Physics-Constrained Graph Neural Network for Acoustic Wave Propagation (DPC-GNN-Acoustic).

---

## 1. Acoustic Wave Equation

### 1.1 Continuous Formulation

The propagation of acoustic waves in a homogeneous medium is governed by the **second-order hyperbolic partial differential equation**:

$$
\frac{\partial^2 p(\mathbf{x}, t)}{\partial t^2} = c^2 \nabla^2 p(\mathbf{x}, t)
$$

where:
- $p(\mathbf{x}, t) \in \mathbb{R}$ is the acoustic pressure field at position $\mathbf{x} \in \Omega \subset \mathbb{R}^d$ ($d = 2, 3$) and time $t \geq 0$
- $c > 0$ is the speed of sound in the medium
- $\nabla^2 = \sum_{i=1}^{d} \frac{\partial^2}{\partial x_i^2}$ is the Laplace operator

#### Conservation Form

The wave equation can be derived from the linearized Euler equations, expressing conservation of mass and momentum:

$$
\begin{aligned}
\frac{\partial \mathbf{v}}{\partial t} &= -\frac{1}{\rho_0} \nabla p \\
\frac{\partial p}{\partial t} &= -\rho_0 c^2 \nabla \cdot \mathbf{v}
\end{aligned}
$$

where $\rho_0$ is the equilibrium density and $\mathbf{v}$ is the particle velocity.

### 1.2 Initial and Boundary Conditions

#### Initial Conditions (Cauchy Problem)

Given initial pressure distribution $p_0$ and initial velocity $v_0$:

$$
\begin{aligned}
p(\mathbf{x}, 0) &= p_0(\mathbf{x}), \quad \forall \mathbf{x} \in \Omega \\
\frac{\partial p}{\partial t}(\mathbf{x}, 0) &= v_0(\mathbf{x}), \quad \forall \mathbf{x} \in \Omega
\end{aligned}
$$

#### Boundary Conditions

**Dirichlet Boundary Conditions** (pressure-specified boundaries):

$$
p(\mathbf{x}, t) = g_D(\mathbf{x}, t), \quad \forall \mathbf{x} \in \partial\Omega_D, \quad t > 0
$$

**Neumann Boundary Conditions** (velocity-specified boundaries):

$$
\frac{\partial p}{\partial \mathbf{n}}(\mathbf{x}, t) = g_N(\mathbf{x}, t), \quad \forall \mathbf{x} \in \partial\Omega_N, \quad t > 0
$$

where $\mathbf{n}$ is the outward unit normal vector and $\partial\Omega = \partial\Omega_D \cup \partial\Omega_N$.

**Absorbing Boundary Conditions** (PML - Perfectly Matched Layer):

For open-domain problems, we use the unsplit PML formulation:

$$
\frac{\partial^2 p}{\partial t^2} + 2\sigma \frac{\partial p}{\partial t} + \sigma^2 p = c^2 \nabla^2 p
$$

where $\sigma(\mathbf{x})$ is the damping profile that increases smoothly from the interior boundary.

### 1.3 Energy Conservation

The total acoustic energy is defined as:

$$
E(t) = \frac{1}{2} \int_{\Omega} \left( \frac{1}{\rho_0 c^2} p^2 + \rho_0 |\mathbf{v}|^2 \right) d\mathbf{x}
$$

For a lossless medium with appropriate boundary conditions, energy is conserved:

$$
\frac{dE}{dt} = 0
$$

---

## 2. Spatial Discretization: Graph Laplacian

### 2.1 Graph Representation

Consider a weighted graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$ where:
- $\mathcal{V} = \{1, \ldots, N\}$ is the set of nodes (spatial discretization points)
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of edges (connectivity)
- $\mathbf{W} \in \mathbb{R}^{N \times N}$ is the weighted adjacency matrix with $W_{ij} > 0$ if $(i,j) \in \mathcal{E}$

### 2.2 Graph Laplacian

The **unnormalized graph Laplacian** is defined as:

$$
\mathbf{L} = \mathbf{D} - \mathbf{W}
$$

where $\mathbf{D} = \text{diag}(d_1, \ldots, d_N)$ with $d_i = \sum_{j} W_{ij}$ is the degree matrix.

The **normalized symmetric graph Laplacian** is:

$$
\mathbf{L}_{\text{sym}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}
$$

### 2.3 Approximation of Continuous Laplacian

For a regular grid with spacing $h$, the standard 5-point stencil in 2D corresponds to:

$$
W_{ij} = \begin{cases}
\frac{1}{h^2} & \text{if } j \in \mathcal{N}(i) \\
0 & \text{otherwise}
\end{cases}
$$

Then the discrete Laplacian approximation becomes:

$$
(\mathbf{L} \mathbf{p})_i = \sum_{j \in \mathcal{N}(i)} W_{ij} (p_j - p_i) \approx \nabla^2 p(\mathbf{x}_i)
$$

### 2.4 Spectral Properties

The graph Laplacian has the following properties:
1. **Positive semi-definiteness**: $\mathbf{L} \succeq 0$ (all eigenvalues $\lambda_k \geq 0$)
2. **Zero eigenvalue**: $\lambda_0 = 0$ with eigenvector $\mathbf{1}$ (constant function)
3. **Spectral gap**: $\lambda_1 > 0$ if and only if $\mathcal{G}$ is connected

The eigenvalue problem:

$$
\mathbf{L} \mathbf{u}_k = \lambda_k \mathbf{u}_k, \quad k = 0, \ldots, N-1
$$

provides a discrete analog to the continuous Laplacian eigenfunctions.

---

## 3. Time Discretization: Leapfrog Scheme

### 3.1 Explicit Leapfrog Integration

For the semi-discrete system:

$$
\frac{d^2 \mathbf{p}}{dt^2} = -c^2 \mathbf{L} \mathbf{p}
$$

The leapfrog (second-order central difference) scheme uses:

$$
\frac{\mathbf{p}^{n+1} - 2\mathbf{p}^n + \mathbf{p}^{n-1}}{\Delta t^2} = -c^2 \mathbf{L} \mathbf{p}^n
$$

Solving for $\mathbf{p}^{n+1}$:

$$
\mathbf{p}^{n+1} = 2\mathbf{p}^n - \mathbf{p}^{n-1} - c^2 \Delta t^2 \mathbf{L} \mathbf{p}^n
$$

### 3.2 Velocity-Verlet Formulation

Equivalently, using first-order system:

$$
\begin{aligned}
\mathbf{p}^{n+1/2} &= \mathbf{p}^{n-1/2} + \Delta t \cdot \mathbf{v}^n \\
\mathbf{v}^{n+1} &= \mathbf{v}^n - c^2 \Delta t \cdot \mathbf{L} \mathbf{p}^{n+1/2}
\end{aligned}
$$

This formulation maintains second-order accuracy and symplectic structure.

### 3.3 Stability Analysis: CFL Condition

**Von Neumann Stability Analysis**:

Assume a plane wave solution $\mathbf{p}^n = \mathbf{u}_k e^{i\omega n \Delta t}$. Substituting into the leapfrog scheme:

$$
e^{i\omega \Delta t} + e^{-i\omega \Delta t} = 2 - c^2 \Delta t^2 \lambda_k
$$

Using $e^{i\omega \Delta t} + e^{-i\omega \Delta t} = 2\cos(\omega \Delta t)$:

$$
\cos(\omega \Delta t) = 1 - \frac{c^2 \Delta t^2 \lambda_k}{2}
$$

For stability, we require $|\cos(\omega \Delta t)| \leq 1$:

$$
-1 \leq 1 - \frac{c^2 \Delta t^2 \lambda_k}{2} \leq 1
$$

The right inequality is always satisfied. The left inequality gives:

$$
c^2 \Delta t^2 \lambda_{\max} \leq 4
$$

**Courant-Friedrichs-Lewy (CFL) Condition**:

For a regular grid with spacing $h$ in $d$ dimensions, $\lambda_{\max} \approx \frac{4d}{h^2}$:

$$
\Delta t \leq \frac{h}{c\sqrt{d}}
$$

In 2D: $\Delta t \leq \frac{h}{c\sqrt{2}}$

In 3D: $\Delta t \leq \frac{h}{c\sqrt{3}}$

### 3.4 Dispersion Analysis

The numerical dispersion relation is:

$$
\omega_{\text{num}} = \frac{1}{\Delta t} \arccos\left(1 - \frac{c^2 \Delta t^2 \lambda_k}{2}\right)
$$

For small $\Delta t$, this approximates the continuous dispersion $\omega = c\sqrt{\lambda_k}$.

---

## 4. GNN Approximation Theory

### 4.1 Message Passing as Discrete Differential Operator

The **Graph Convolutional Network (GCN)** message passing rule:

$$
\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\hat{d}_i \hat{d}_j}} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)}\right)
$$

where $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ and $\hat{\mathbf{D}}$ is its degree matrix.

**Connection to Laplacian**:

For small updates with $\sigma(x) \approx x$:

$$
\mathbf{h}^{(l+1)} - \mathbf{h}^{(l)} \approx -\mathbf{L}_{\text{sym}} \mathbf{W}^{(l)} \mathbf{h}^{(l)}
$$

This resembles a discretization of the heat equation $\frac{\partial u}{\partial t} = -\mathcal{L}u$.

### 4.2 Universal Approximation Theorem for GNNs

**Theorem** (Maron et al., 2019; Keriven & Peyré, 2019):

Let $f: \mathbb{R}^{N \times d_{in}} \rightarrow \mathbb{R}^{N \times d_{out}}$ be a continuous graph function that is equivariant to graph isomorphisms. For any $\epsilon > 0$ and any compact set $\mathcal{K} \subset \mathbb{R}^{N \times d_{in}}$, there exists a message-passing neural network (MPNN) with sufficiently many layers and sufficiently large hidden dimensions such that:

$$
\sup_{\mathbf{X} \in \mathcal{K}} \|f(\mathbf{X}) - \text{MPNN}(\mathbf{X})\| < \epsilon
$$

### 4.3 Convergence Analysis

Consider a sequence of graphs $\{\mathcal{G}_n\}$ converging to a manifold $\mathcal{M}$ in the sense of graphons or metric measure spaces.

**Theorem** (Belkin & Niyogi, 2008; Burago et al., 2014):

Let $u: \mathcal{M} \rightarrow \mathbb{R}$ be a smooth function. Under appropriate conditions on the graph construction (e.g., $k$-nearest neighbors with $k \sim n^{2/(d+2)}$ or $\epsilon$-neighborhood with $\epsilon \sim n^{-1/(d+4)}$):

$$
\lim_{n \rightarrow \infty} \mathbf{L}_n \mathbf{u}_n = \Delta_{\mathcal{M}} u
$$

where $\mathbf{L}_n$ is the graph Laplacian on $\mathcal{G}_n$ and $\Delta_{\mathcal{M}}$ is the Laplace-Beltrami operator on $\mathcal{M}$.

**Pointwise Convergence Rate**:

For points away from the boundary:

$$
|(\mathbf{L}_n \mathbf{u})_i - \Delta u(\mathbf{x}_i)| = \mathcal{O}\left(h^2 + \frac{1}{nh^d}\right)
$$

where $h$ is the characteristic spacing. The optimal rate is achieved when $h \sim n^{-1/(d+4)}$.

### 4.4 Spectral Convergence

**Theorem** (Von Luxburg et al., 2008):

Under the same conditions, the eigenvalues and eigenvectors of $\mathbf{L}_n$ converge to those of $\Delta_{\mathcal{M}}$:

$$
\begin{aligned}
\lambda_k(\mathbf{L}_n) &\rightarrow \lambda_k(\Delta_{\mathcal{M}}) \\
\mathbf{u}_k^{(n)} &\rightarrow u_k \quad \text{(in } L^2\text{ sense)}
\end{aligned}
$$

This ensures that the GNN-based wave propagation maintains the correct spectral properties of the continuous problem.

---

## 5. Summary of Key Mathematical Results

| Aspect | Continuous | Discrete (Graph) |
|--------|-----------|------------------|
| Domain | $\Omega \subset \mathbb{R}^d$ | $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ |
| Laplacian | $\nabla^2$ | $\mathbf{L} = \mathbf{D} - \mathbf{W}$ |
| Wave Equation | $\partial_t^2 p = c^2 \nabla^2 p$ | $\ddot{\mathbf{p}} = -c^2 \mathbf{L} \mathbf{p}$ |
| Eigenproblem | $-\nabla^2 \phi_k = \lambda_k \phi_k$ | $\mathbf{L} \mathbf{u}_k = \lambda_k \mathbf{u}_k$ |
| Stability | N/A | $\Delta t \leq h/(c\sqrt{d})$ |
| Energy | $E = \frac{1}{2}\int (p^2/c^2 + |\nabla p|^2) d\mathbf{x}$ | $E_h = \frac{1}{2}\mathbf{p}^T \mathbf{M} \mathbf{p} + \frac{1}{2}\mathbf{v}^T \mathbf{M} \mathbf{v}$ |

---

## References

See `references.bib` for the complete bibliography.

---

*Document Version: 1.0*  
*Last Updated: March 2026*
