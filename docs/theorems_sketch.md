# Theorem Proofs and Sketch: DPC-GNN-Acoustic

This document provides formal statements and proof sketches for the key theoretical results underlying DPC-GNN-Acoustic.

---

## Theorem 1: WaveEquation-MP Satisfies Discrete Wave Equation

### Statement

**Theorem 1** (Discrete Wave Equation Satisfaction). Let $\mathbf{p}^n \in \mathbb{R}^N$ be the pressure field at time step $n$ computed by the WaveEquation-MP layer. Then for any graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$ with graph Laplacian $\mathbf{L}$, time step $\Delta t$, and sound speed matrix $\mathbf{C} = \text{diag}(c_1^2, \ldots, c_N^2)$, the sequence $\{\mathbf{p}^n\}_{n=0}^{K}$ satisfies:

$$
\frac{\mathbf{p}^{n+1} - 2\mathbf{p}^n + \mathbf{p}^{n-1}}{\Delta t^2} + \mathbf{C} \mathbf{L} \mathbf{p}^n = 0
$$

for all $n = 1, \ldots, K-1$.

### Proof

**Proof.** By construction, the WaveEquation-MP layer implements the following update rule:

$$
\mathbf{p}^{n+1} = 2\mathbf{p}^n - \mathbf{p}^{n-1} - \Delta t^2 \cdot \mathbf{C} \mathbf{L} \mathbf{p}^n
$$

Rearranging terms:

$$
\mathbf{p}^{n+1} - 2\mathbf{p}^n + \mathbf{p}^{n-1} = -\Delta t^2 \cdot \mathbf{C} \mathbf{L} \mathbf{p}^n
$$

Dividing by $\Delta t^2$:

$$
\frac{\mathbf{p}^{n+1} - 2\mathbf{p}^n + \mathbf{p}^{n-1}}{\Delta t^2} = -\mathbf{C} \mathbf{L} \mathbf{p}^n
$$

Moving all terms to the left-hand side:

$$
\frac{\mathbf{p}^{n+1} - 2\mathbf{p}^n + \mathbf{p}^{n-1}}{\Delta t^2} + \mathbf{C} \mathbf{L} \mathbf{p}^n = 0
$$

This is precisely the central difference approximation of the wave equation with the discrete graph Laplacian replacing the continuous Laplacian. **Q.E.D.**

### Corollary: Time Reversibility

**Corollary 1.1** (Time Reversibility). The WaveEquation-MP scheme is time-reversible, i.e., if $\{\mathbf{p}^n\}$ is a solution, then so is $\{\mathbf{p}^{K-n}\}$ with reversed initial velocity.

**Proof Sketch.** The leapfrog scheme is symmetric in time. Replacing $n \rightarrow K-n$ preserves the equation structure. **Q.E.D.**

---

## Theorem 2: Convergence to Continuous Solution

### Statement

**Theorem 2** (Convergence to Continuous Solution). Let $\Omega \subset \mathbb{R}^d$ be a bounded Lipschitz domain with smooth boundary. Let $p(\mathbf{x}, t)$ be the solution to the continuous wave equation:

$$
\begin{cases}
\partial_t^2 p = c^2 \nabla^2 p & \text{in } \Omega \times [0, T] \\
p(\mathbf{x}, 0) = p_0(\mathbf{x}), \quad \partial_t p(\mathbf{x}, 0) = v_0(\mathbf{x}) & \text{in } \Omega \\
p(\mathbf{x}, t) = 0 & \text{on } \partial\Omega \times [0, T]
\end{cases}
$$

Consider a sequence of graphs $\{\mathcal{G}_N\}_{N=1}^{\infty}$ constructed as follows:
- Nodes $\{\mathbf{x}_i^{(N)}\}_{i=1}^N$ are sampled uniformly from $\Omega$
- Edges connect nodes within distance $h_N = \mathcal{O}(N^{-1/d})$
- Edge weights $w_{ij}^{(N)} = h_N^{-(d+2)} \phi\left(\frac{\|\mathbf{x}_i - \mathbf{x}_j\|}{h_N}\right)$ for kernel $\phi$

Let $\mathbf{p}_N^n$ be the DPC-GNN-Acoustic solution on $\mathcal{G}_N$ with $\Delta t_N = \mathcal{O}(h_N)$ satisfying the CFL condition. Then:

$$
\lim_{N \rightarrow \infty} \max_{0 \leq n \leq T/\Delta t_N} \| \mathbf{p}_N^n - \mathbf{p}(\cdot, n\Delta t_N) \|_{L^2(\Omega)} = 0
$$

### Proof Sketch

**Step 1: Consistency of Spatial Discretization**

For smooth $u$, the weighted graph Laplacian converges to the continuous Laplacian (Burago et al., 2014):

$$
\lim_{N \rightarrow \infty} (\mathbf{L}_N \mathbf{u})_i = \nabla^2 u(\mathbf{x}_i)
$$

The convergence rate is $\mathcal{O}(h_N^2)$ for interior points and $\mathcal{O}(h_N)$ for boundary points.

**Step 2: Stability via CFL Condition**

The eigenvalues of $\mathbf{L}_N$ satisfy (Coifman & Lafon, 2006):

$$
\lambda_k(\mathbf{L}_N) \leq \frac{C}{h_N^2}
$$

With $\Delta t_N = \alpha h_N$ and $\alpha < c^{-1}\sqrt{d^{-1}}$, the CFL condition ensures:

$$
\| \mathbf{p}_N^{n+1} \| \leq C \| \mathbf{p}_N^n \|
$$

for some constant $C$ independent of $N$ and $n$.

**Step 3: Lax Equivalence Theorem**

For linear evolution equations with consistent and stable discretizations, consistency + stability implies convergence (Lax Equivalence Theorem). 

**Step 4: Convergence Rate**

Under sufficient regularity assumptions ($p_0, v_0 \in H^4(\Omega)$), the error satisfies:

$$
\| \mathbf{p}_N^n - p(\cdot, n\Delta t) \|_{L^2} \leq C_1 h_N^2 + C_2 \Delta t_N^2
$$

**Q.E.D.**

### Remarks

1. The convergence rate $\mathcal{O}(h^2 + \Delta t^2)$ matches standard FDTD methods.
2. For irregular meshes derived from medical imaging, the rate may degrade to $\mathcal{O}(h)$ near boundaries.
3. The result extends to heterogeneous media with $c(\mathbf{x}) \in C^1(\bar{\Omega})$.

---

## Theorem 3: Gradient Flow Preserves Physical Conservation Laws

### Statement

**Theorem 3** (Physical Conservation in Gradient Flow). Consider training the DPC-GNN-Acoustic model with the physics-informed loss:

$$
\mathcal{L}(\theta) = \sum_{k=1}^{K} \| \mathbf{p}^k(\theta) - \mathbf{p}^k_{\text{target}} \|^2
$$

where $\mathbf{p}^k(\theta)$ is computed by iterating the WaveEquation-MP layer with parameters $\theta$ (e.g., sound speed field). The gradient descent update:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}(\theta^{(t)})
$$

satisfies the following properties:

**(a) Energy Dissipation**: The discrete energy $E^k = \frac{1}{2} \mathbf{p}^k \cdot \mathbf{M} \mathbf{p}^k + \frac{1}{2} \dot{\mathbf{p}}^k \cdot \mathbf{M} \dot{\mathbf{p}}^k$ remains bounded throughout training.

**(b) Symplectic Structure Preservation**: The update preserves the symplectic form $\omega = d\mathbf{p} \wedge d\mathbf{v}$ in the phase space $(\mathbf{p}, \mathbf{v})$.

**(c) Causal Gradient Flow**: For any parameter $\theta_i$ affecting only the solution at time $t \leq t_0$, the gradient $\partial \mathcal{L} / \partial \theta_i$ depends only on $\mathbf{p}^k$ for $k \geq t_0/\Delta t$.

### Proof Sketch

**(a) Energy Dissipation**

The leapfrog scheme is a symplectic integrator (Hairer et al., 2006). For the Hamiltonian system with:

$$
H(\mathbf{p}, \mathbf{v}) = \frac{1}{2} \mathbf{v}^T \mathbf{M} \mathbf{v} + \frac{c^2}{2} \mathbf{p}^T \mathbf{L} \mathbf{p}
$$

the leapfrog scheme preserves a modified Hamiltonian $\tilde{H} = H + \mathcal{O}(\Delta t^2)$ exactly. Therefore, energy oscillates around the true value with amplitude $\mathcal{O}(\Delta t^2)$, ensuring boundedness.

**(b) Symplectic Structure**

The leapfrog update can be written as:

$$
\begin{pmatrix} \mathbf{p}^{n+1} \\ \mathbf{v}^{n+1} \end{pmatrix} = \mathbf{J}_{\text{LF}} \begin{pmatrix} \mathbf{p}^n \\ \mathbf{v}^n \end{pmatrix}
$$

where $\mathbf{J}_{\text{LF}}$ is the leapfrog Jacobian. Direct computation shows:

$$
\mathbf{J}_{\text{LF}}^T \mathbf{\Omega} \mathbf{J}_{\text{LF}} = \mathbf{\Omega}
$$

where $\mathbf{\Omega} = \begin{pmatrix} \mathbf{0} & -\mathbf{M} \\ \mathbf{M} & \mathbf{0} \end{pmatrix}$ is the symplectic matrix. This proves symplecticity.

**(c) Causal Gradient Flow**

By the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} = \sum_{k=1}^{K} 2(\mathbf{p}^k - \mathbf{p}^k_{\text{target}})^T \frac{\partial \mathbf{p}^k}{\partial \theta_i}
$$

The Jacobian $\partial \mathbf{p}^k / \partial \theta_i$ depends on $\theta_i$ only through $\mathbf{p}^j$ for $j \leq k$. Therefore, if $\theta_i$ affects only $t \leq t_0$, the gradient at time $k \u003c t_0/\Delta t$ vanishes.

**Q.E.D.**

---

## Lemma Collection

### Lemma A: Graph Laplacian Spectral Bounds

**Lemma A.1**. For a $k$-regular graph with $N$ nodes, the eigenvalues of the normalized Laplacian satisfy:

$$
0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{N-1} \leq 2
$$

**Proof.** The normalized Laplacian $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ is positive semi-definite with eigenvalues in $[0, 2]$ (Chung, 1997). **Q.E.D.**

### Lemma B: CFL Condition Sufficiency

**Lemma B.1**. The leapfrog scheme is stable if and only if:

$$
\Delta t \leq \frac{2}{c \sqrt{\lambda_{\max}(\mathbf{L})}}
$$

**Proof.** From the Von Neumann analysis, stability requires $|2 - c^2 \Delta t^2 \lambda| \leq 2$ for all eigenvalues $\lambda$. This gives $c^2 \Delta t^2 \lambda \leq 4$, or $\Delta t \leq 2/(c\sqrt{\lambda_{\max}})$. **Q.E.D.**

### Lemma C: Backpropagation Through Time

**Lemma C.1**. The gradient of the loss with respect to initial conditions satisfies:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{p}^0} = \sum_{k=1}^{K} \mathbf{J}_{0,k}^T \frac{\partial \mathcal{L}}{\partial \mathbf{p}^k}
$$

where $\mathbf{J}_{0,k} = \prod_{j=1}^{k} \frac{\partial \mathbf{p}^j}{\partial \mathbf{p}^{j-1}}$ is the product of Jacobian matrices.

**Proof.** Direct application of the chain rule for composed functions. **Q.E.D.**

---

## Proposition: Hard Constraints vs. Soft Constraints

**Proposition**. Let $\mathcal{F}_{\text{hard}}$ be the function class of networks with hard physics constraints (WaveEquation-MP embedded) and $\mathcal{F}_{\text{soft}}$ be networks with soft constraints (physics loss penalty). For any $\epsilon > 0$:

$$
\inf_{f \in \mathcal{F}_{\text{soft}}} \mathbb{E}[\mathcal{L}_{\text{physics}}(f)] \geq 0
$$

with equality achieved by any $f \in \mathcal{F}_{\text{hard}}$.

**Proof Sketch.** Hard constraints satisfy $\mathcal{L}_{\text{physics}}(f) = 0$ by construction (Theorem 1). Soft constraints can only approach zero asymptotically. Therefore, the infimum over soft constraints is non-negative and strictly positive unless the network learns to satisfy the constraint exactly. **Q.E.D.**

---

## Future Work: Open Problems

The following theoretical questions remain open and are subjects of ongoing research:

1. **Optimal Graph Construction**: What is the optimal graph topology for a given error tolerance $\epsilon$ and computational budget?

2. **Generalization Bounds**: Derive PAC-Bayesian bounds for the generalization error of DPC-GNN-Acoustic.

3. **Long-Time Stability**: Prove stability for arbitrarily long time horizons $T \rightarrow \infty$.

4. **Nonlinear Extensions**: Extend convergence results to nonlinear wave equations (e.g., Westervelt equation for high-intensity ultrasound).

---

## References to Full Proofs

For detailed proofs of supporting results, see:

- Burago et al. (2014): Graph Laplacian convergence
- Hairer et al. (2006): Symplectic integration
- Lax & Richtmyer (1956): Lax Equivalence Theorem
- Chung (1997): Spectral Graph Theory

See `references.bib` for complete citations.

---

*Theorem Document Version: 1.0*  
*Last Updated: March 2026*
