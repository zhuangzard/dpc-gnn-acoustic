# Baseline Comparison Table

> Last updated: 2026-03-19
> Context: CT-to-US synthesis and physics-based ultrasound simulation baselines for DPC-GNN / US-DPC-GNN project.

## Summary Table

| Method | Type | Params | SSIM | PSNR (dB) | Speed | Physics | Dataset | Source |
|--------|------|--------|------|-----------|-------|---------|---------|--------|
| **j-Wave** (Stanziola 2023) | Physics sim (JAX) | N/A (solver) | N/A (waveform sim) | N/A | ~seconds/frame (GPU, JAX JIT) | ✅ Full wave (k-space, FDTD) | Generic acoustic | arXiv:2207.01499; SoftwareX 2023 |
| **FNO** (Li et al. 2021) | Neural operator | ~6.6M (4-layer) | N/A | N/A | 1000× faster than PDE solver | ❌ Learned physics | Burgers/Darcy/Navier-Stokes | arXiv:2010.08895, Table 1–3 |
| **Almahfouz Nasser 2023** | Physics sim (wave-based) | N/A (solver) | N/A (no image-level metric) | N/A | Minutes–hours per volume | ✅ Ray-tracing + wave propagation | CT→US (abdomen/liver) | medRxiv:2023.01.16.23284615 |
| **S-CycleGAN** (Song & Chong 2024) | GAN (semantic) | ~CycleGAN-scale | Not reported† | Not reported† | Real-time inference | ❌ None | CT→US (abdomen) | arXiv:2406.01191 |
| **Li et al. 2023** | Physics sim + DL seg | N/A (hybrid pipeline) | N/A (task-oriented) | N/A | Near real-time (seg) | ✅ In-silico bone US sim | CT→US (spine) | arXiv:2301.01940 |
| **Pix2Pix** (Isola 2017) | cGAN (paired) | ~54.4M (U-Net gen) | 0.85–0.92‡ | 25–32‡ | ~50ms/frame (GPU) | ❌ None | Various medical | Various; see notes |
| **CycleGAN** (Zhu 2017) | GAN (unpaired) | ~11.4M×2 (ResNet gen) | 0.78–0.88‡ | 22–28‡ | ~80ms/frame (GPU) | ❌ None | Various medical | Various; see notes |
| **Diffusion (B-Maps)** (Domínguez/Velikova 2024) | Diffusion (physics-inspired) | ~UNet-based (GD/SDM) | Reported (see notes)§ | Reported (see notes)§ | ~minutes/sample | ⚠️ Physics-inspired scheduler | Thyroid (SegThy), Cardiac (CAMUS), Liver | MICCAI 2024; arXiv:2407.05428 |

**Legend:**
- ✅ = explicit physics modeling; ⚠️ = physics-inspired but not full simulation; ❌ = no physics
- N/A = metric not applicable or not reported for this method type
- † = authors explicitly state "Suitable metrics that comprehensively evaluate the effectiveness of ultrasound image synthesis in a numerical manner are still absent"
- ‡ = typical range across medical image synthesis literature (see Detailed Notes)
- § = table in paper (Table 2); exact numbers in images/PDF only

---

## Detailed Notes

### 1. j-Wave (Stanziola et al. 2023)
- **Paper:** "j-Wave: An open-source differentiable wave simulator" — SoftwareX, 2023. arXiv:2207.01499
- **Type:** Full-wave acoustic simulator built on JAX. Supports k-space pseudo-spectral methods (time-domain). Fully differentiable via JAX autodiff.
- **Performance:** j-Wave is a PDE solver, not an image synthesis model — it does not report SSIM/PSNR. Its value is in enabling **differentiable physics** for gradient-based optimization.
- **Speed:** JAX JIT compilation enables GPU-accelerated simulation. The paper benchmarks against k-Wave (MATLAB); j-Wave achieves comparable accuracy with differentiability. Typical 2D simulation: seconds on GPU.
- **Params:** Not applicable (solver, not neural network).
- **Physics:** Full wave equation (Helmholtz/time-domain). Supports heterogeneous media, absorption, nonlinearity.
- **Key insight for our work:** j-Wave can be used as a differentiable physics backbone. Our DPC-GNN approach complements it by learning the CT→acoustic-property mapping that feeds into such solvers.
- **Source:** Section 3 (Benchmarks), validated against k-Wave analytical solutions.

### 2. FNO — Fourier Neural Operator (Li et al. 2021)
- **Paper:** "Fourier Neural Operator for Parametric Partial Differential Equations" — ICLR 2021. arXiv:2010.08895
- **Type:** Neural operator that learns mappings between function spaces. Parameterizes integral kernel in Fourier space.
- **Key results (from Table 1–3 in paper):**
  - **Burgers' equation:** Relative L2 error = **0.0018** (resolution 8192)
  - **Darcy Flow (r=421):** Relative L2 error = **0.0108**
  - **Navier-Stokes (ν=1e-3, T=50):** Relative L2 error = **0.0086**
  - **Navier-Stokes (ν=1e-4, T=30):** Relative L2 error = **0.0098**
  - **Navier-Stokes (ν=1e-5, T=20):** Relative L2 error = **0.0820** (turbulent regime)
- **Speed:** Up to **1000× faster** than traditional PDE solvers (stated in abstract). Inference: single forward pass ~0.005s per instance.
- **Params:** ~6.6M for 4-layer FNO (12 Fourier modes).
- **Physics:** No explicit physics encoding — learns PDE solution operator from data.
- **Relevance:** FNO shows neural operators can approximate PDE solutions with ~1% relative error. However, **not directly applied to acoustic/US simulation**. The wave equation is structurally similar to Navier-Stokes but FNO has not been benchmarked on acoustic imaging tasks specifically.
- **Source:** Tables 1–3 in the ICLR 2021 paper.

### 3. Almahfouz Nasser 2023 — Wave-based CT-to-US
- **Paper:** "Simulating Ultrasound Images from CT Scans" — BIOIMAGING 2023 (SciTePress)
- **Type:** Physics-based simulation pipeline: CT → tissue property extraction → wave propagation → B-mode image formation.
- **Performance:** The paper focuses on **qualitative realism** of simulated US from CT. No SSIM/PSNR/FID scores are reported. Evaluation is primarily visual comparison + radiologist assessment.
- **Speed:** Physics simulation is computationally expensive (minutes to hours per volume depending on resolution and wave model).
- **Physics:** ✅ Full — ray-tracing for tissue interfaces, wave propagation model for acoustic interaction, B-mode image formation pipeline.
- **Dataset:** CT scans (abdomen/liver), simulated US output.
- **Key insight:** This is the most directly relevant prior work for CT→US physics-based synthesis. The lack of quantitative metrics is a gap our work can fill.
- **Source:** SciTePress Proceedings, BIOIMAGING 2023.

### 4. S-CycleGAN (Song & Chong 2024)
- **Paper:** "Simulating Realistic Ultrasound Images from CT Volumes via S-CycleGAN" — arXiv:2406.01191
- **Type:** Modified CycleGAN with semantic label guidance for CT→US translation.
- **Performance:** Authors explicitly note in Section 5: *"Suitable metrics that comprehensively evaluate the effectiveness of ultrasound image synthesis in a numerical manner are still absent."* They do NOT report SSIM, PSNR, or FID. Evaluation is **qualitative only** (visual comparison, expert review).
- **Architecture:** CycleGAN backbone with semantic consistency loss. Uses paired CT-US data from the same patient with semantic organ labels.
- **Dataset:** Abdominal CT-US pairs.
- **Physics:** ❌ No physics modeling — purely image-to-image translation.
- **Key insight:** Even in 2024, CT→US synthesis papers struggle with quantitative evaluation. This reinforces the need for physics-based metrics (which our approach provides).
- **Source:** arXiv:2406.01191v2, Section 5 (Evaluation).

### 5. Li et al. 2023 — CT-to-US Spine Surgery
- **Paper:** "Enabling Augmented Segmentation and Registration in Ultrasound-Guided Spinal Surgery via Realistic Ultrasound Synthesis from Diagnostic CT Volume" — Submitted to IEEE T-ASE. arXiv:2301.01940
- **Type:** Hybrid pipeline: in-silico bone US simulation from CT + lightweight ViT for bone segmentation.
- **Key results (from abstract/paper):**
  - **Bone segmentation Chamfer distance:** 0.599 mm
  - **CT-US registration Dice:** **0.93**
  - **Registration accuracy:** 0.13–3.37 mm (point cloud based)
- **Note:** SSIM/PSNR not reported — evaluation is task-oriented (segmentation accuracy, registration error), not image-quality oriented.
- **Physics:** ✅ In-silico simulation based on CT-derived acoustic properties (tissue impedance, reflection at interfaces).
- **Dataset:** Spinal CT volumes → simulated US for training; validated on real US.
- **Key insight:** Task-oriented evaluation (Dice, Chamfer distance) may be more meaningful than pixel-level SSIM for surgical applications. Our DPC-GNN should report both.
- **Source:** arXiv:2301.01940, Abstract + Results section.

### 6. Pix2Pix / CycleGAN in Medical Image Synthesis
- **Pix2Pix** (Isola et al. 2017): Conditional GAN with U-Net generator + PatchGAN discriminator.
- **CycleGAN** (Zhu et al. 2017): Unpaired image translation with cycle consistency loss.

**Typical SSIM/PSNR ranges in medical image synthesis** (aggregated from literature):

| Task | Method | SSIM | PSNR (dB) | Source |
|------|--------|------|-----------|--------|
| CBCT → CT | CycleGAN | 0.87 ± 0.02 | ~28 | Frontiers Oncol. 2021 |
| CBCT → CT | Double-chain-CycleGAN | Improved over baseline | MAE: 32.05 | PubMed:37244147 |
| MR → CT (brain) | Pix2Pix | 0.88–0.92 | 28–32 | Multiple surveys |
| MR → CT (pelvis) | CycleGAN | 0.82–0.87 | 24–28 | Multiple surveys |
| CT → MR (brain) | DC-CycleGAN | ~0.85 | ~26 | CMIG 2023 |
| CT → US | Pix2Pix (typical) | 0.55–0.75* | 18–24* | Limited reports |
| CT → US | CycleGAN (typical) | 0.50–0.70* | 16–22* | Limited reports |

*Note: CT→US is fundamentally harder than CT↔MR due to the drastically different image formation physics. SSIM values for CT→US are significantly lower than CT↔MR. These ranges are estimated from the limited literature.*

- **Params:** Pix2Pix ~54.4M (U-Net256); CycleGAN ~11.4M×2 (ResNet-9blocks).
- **Speed:** Both support real-time inference (~50-80ms/frame on GPU).

### 7. Diffusion Models for US Synthesis
- **Paper:** "Diffusion as Sound Propagation: Physics-inspired Model for Ultrasound Image Generation" — Domínguez, Velikova et al. MICCAI 2024. arXiv:2407.05428
- **Type:** Modified DDPM with "B-Maps" noise scheduler inspired by US wave attenuation.
- **Key results (Table 1 — FID, and Table 2 — LPIPS/SSIM/PSNR):**
  - Table 2 reports LPIPS, SSIM, PSNR for Thyroid (SegThy) and CAMUS datasets
  - Results show B-Maps method **consistently lower LPIPS** and **higher PSNR** than baseline GD/SDM
  - SSIM shows "minimal differences from baseline methods — indicating comparable structural integrity"
  - **Exact numbers are in Table 2 of the paper** (embedded as image in PDF; not extractable from HTML)
  - FID (Table 1): B-Maps significantly lower than baselines across all datasets using 1st and 2nd max-pooling features
- **Note:** This is **not CT→US synthesis** — it's unconditional/semantic-conditioned US image generation (generating new US images from noise/labels). Datasets: SegThy (thyroid, 2250 images), CAMUS (cardiac, 1600+augmented), Liver (6900 slices).
- **Speed:** Diffusion models are slow at inference (~minutes per sample with 2000 diffusion steps).
- **Physics:** Physics-*inspired* only — the B-Maps scheduler mimics attenuation behavior but does not solve wave equations.
- **Params:** Based on Guided-Diffusion / SDM architectures (UNet backbone).
- **Source:** MICCAI 2024, LNCS 15004, pp. 613–623. Tables 1 & 2.

---

## Key Gaps in Existing Literature (Opportunities for DPC-GNN)

1. **No method combines differentiable physics + learned tissue mapping + quantitative evaluation** — j-Wave has physics but no learning; FNO has learning but no physics grounding in US domain; GANs have learning but no physics.

2. **CT→US quantitative metrics are severely lacking** — S-CycleGAN (2024) explicitly admits no good metrics exist; Almahfouz Nasser reports only qualitative results; Li et al. use task-oriented metrics only.

3. **Speed vs. fidelity trade-off is unaddressed** — Physics sims (j-Wave, Almahfouz) are slow but accurate; GANs are fast but physics-blind; no method achieves both.

4. **Graph-based approaches are entirely absent** — No prior work uses GNNs for tissue property propagation in CT→US synthesis. This is a unique contribution of DPC-GNN.

5. **Diffusion models show promise but are not physics-grounded** — B-Maps (MICCAI 2024) is a step toward physics but still fundamentally a noise scheduler trick, not true wave simulation.

---

## Quick Reference: Metric Definitions

| Metric | Range | Better | Notes |
|--------|-------|--------|-------|
| SSIM | [0, 1] | Higher | Structural similarity; 1 = perfect match |
| PSNR | [0, ∞) dB | Higher | Peak signal-to-noise ratio; >30 dB generally good |
| FID | [0, ∞) | Lower | Fréchet Inception Distance; measures distribution similarity |
| LPIPS | [0, 1] | Lower | Learned Perceptual Image Patch Similarity |
| Relative L2 Error | [0, ∞) | Lower | ‖u_pred - u_true‖₂ / ‖u_true‖₂ |
| Chamfer Distance | [0, ∞) mm | Lower | Point cloud distance metric |
| Dice | [0, 1] | Higher | Segmentation overlap; 1 = perfect |
