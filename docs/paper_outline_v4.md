# Paper Outline: DPC-GNN-Acoustic V4

## Title
**"DPC-GNN-Acoustic: Physics-as-Forward Graph Neural Network for Real-time CT-to-Ultrasound Synthesis"**

备选:
- "From CT to Ultrasound in Milliseconds: Hard-Physics GNN with Zero Learnable Dynamics"
- "Physics is Forward, Not Loss: DPC-GNN for Differentiable Acoustic Simulation"

## Target Venues
- **Medical Image Analysis** (首选，与DPC-GNN软组织版同系列)
- IEEE TMI (Transactions on Medical Imaging)
- MICCAI 2026 (Method track)

## 核心创新（一句话）

**GNN只学材料属性（CT→声速/衰减/散射），波动力学由确定性Leapfrog硬约束执行——物理不是loss，物理就是forward pass。**

这是DPC-GNN统一框架从软组织力学到声学波传播的首次扩展，证明"Physics-as-Forward"范式可以跨物理域迁移。

---

## Abstract Structure

### Background
- CT-to-Ultrasound synthesis对手术导航至关重要（术中实时配准）
- k-Wave等物理仿真精确但慢（256×256 ~3.2s/sample）
- 现有深度学习方法（CycleGAN、Diffusion）快但无物理保证
- **PINN方法的根本缺陷**：将波动方程放入loss作为软约束（physics_weight可调），物理变成"鼓励遵守"而非"必须遵守"

### Method
- **DPC-GNN-Acoustic V4**: 对齐DPC-GNN软组织版的"Physics-as-Forward"哲学
- GNN编码器（~253K参数）：CT→声学参数场 c(x,y), α(x,y), σ(x,y)
- 确定性Leapfrog传播器（0可学习参数）：严格执行波动方程
- 确定性DAS波束成形（0可学习参数）：标准超声成像算法
- 物理先验：HU→c查表 + GNN残差修正
- 反对称消息传递保证空间一致性

### Results (目标)
- SSIM > 0.9 vs k-Wave GT
- 推理 < 100ms（vs k-Wave 3.2s = **30×+加速**）
- 物理残差 ≈ 0（数值精度级别，证明物理自治）
- 消融：反对称MP vs 普通MP，残差结构 vs 直接预测

### Significance
- 首次将DPC-GNN"Physics-as-Forward"范式从固体力学扩展到波动力学
- 证明physics_weight=0是唯一正确选择（不是调参问题，是架构问题）
- 为手术导航提供实时、物理正确的超声仿真

---

## Paper Structure

### 1. Introduction (1.5 pages)
- **动机**: 术中超声-CT配准需要实时仿真
- **现有方法的三个层次**:
  - L1: 纯物理仿真（k-Wave）— 精确但慢
  - L2: 纯学习（CycleGAN/Diffusion）— 快但无物理保证
  - L3: PINN/混合（DiffUS等）— 软约束，physics_weight是假的物理
- **我们的位置: L4 — Physics-as-Forward**:
  - 物理在前向传播中严格执行
  - GNN只学材料属性，不碰动力学
  - 这不是"又一个PINN变体"，是范式转换
- **贡献**:
  1. DPC-GNN框架从软组织到声学的跨域扩展
  2. 证明"物理即前向"范式在波动方程中同样成立
  3. 253K参数模型实现SSIM>0.9 + <100ms推理

### 2. Related Work (1 page)
- **物理仿真**: k-Wave, Field II, MUST
- **学习方法**: CycleGAN CT-to-US, Diffusion US synthesis
- **PINN/软约束方法**: 为什么physics_weight是根本错误（不是调参问题）
- **DPC-GNN系列**: 软组织版的核心思想，本文如何扩展
- **关键区别表**: 

| 方法类别 | 物理约束 | 可学习参数 | 速度 | 物理保证 |
|---------|---------|-----------|------|---------|
| k-Wave | 全部硬约束 | 0 | 慢(3.2s) | ✅ 完全 |
| CycleGAN | 无 | ~10M | 快(10ms) | ❌ |
| PINN | 软约束(loss) | ~1M | 中 | ⚠️ 近似 |
| **Ours** | **硬约束(forward)** | **253K** | **快(<100ms)** | **✅ 完全** |

### 3. Method (3 pages) — **论文核心**

#### 3.1 Problem Formulation
- 给定CT图像I_CT，预测B-mode超声图像I_US
- 中间物理量：声速场c(x,y)、衰减系数α(x,y)、散射强度σ(x,y)
- 波动方程严格约束压力场演化

#### 3.2 DPC-GNN-Acoustic Architecture
**三段式架构（图1）**:

```
[可学习] GNN编码器: I_CT → (c, α, σ)
         ↓
[确定性] Leapfrog传播器: (c, α, σ, source) → p(x,y,t)
         ↓  
[确定性] DAS波束成形: p → I_US
```

##### 3.2.1 GNN Material Encoder
- CNN下采样(256→64) + 5层反对称MP(dim=96) + CNN上采样(64→256)
- **物理先验结构**: c = c_table(HU) + Δc_GNN
  - c_table: 分段线性查表（已知物理）
  - Δc_GNN: 空间上下文残差（GNN学习）
- **输出约束**: c∈[1400,1700], α≥0, σ∈[0,1]
- **反对称MP**: W_anti = W - W^T, m_ij = tanh((h_i-h_j) @ W_anti)
  - 保证空间一致性（来自波方程的自伴随性）

##### 3.2.2 Deterministic Leapfrog Propagator
- 波动方程: ∂²p/∂t² = c²∇²p - α·∂p/∂t + σ·f(t)
- Leapfrog离散: p^{n+1} = [2p^n - (1-αΔt/2)p^{n-1} + Δt²(c²∇²p + σf)] / (1+αΔt/2)
- **零可学习参数** — 纯物理
- PML吸收边界（20格，3次多项式）
- CFL条件由c范围自动保证
- Gradient checkpointing（每20步）

##### 3.2.3 Differentiable DAS Beamformer
- Delay-and-Sum波束成形（线性插值保可微）
- Hilbert变换包络检测（FFT实现）
- 平滑对数压缩: log(envelope + ε)

#### 3.3 Training Strategy
- **损失函数**: L = L1(I_pred, I_gt) + (1 - SSIM(I_pred, I_gt))
- **没有physics_weight** — 物理在forward中自动满足
- physics_residual仅作为监控指标（应≈0）
- AdamW, lr=1e-4, cosine schedule, gradient clipping

#### 3.4 Theoretical Analysis
- **命题1**: 如果Leapfrog传播器正确实现，physics_residual ≡ O(Δt² + Δx²)（数值截断误差）
- **命题2**: 梯度可以从B-mode loss穿过Leapfrog和DAS回传到GNN参数
- **推论**: physics_weight=0是理论最优（不是超参数选择，是架构必然）

### 4. Experiments (2 pages)

#### 4.1 Dataset
- k-Wave GT: 300样本（100均匀+100分层+100包体）
- 256×256网格，dx=2.34e-4m, ~1754时间步
- 分割: 240训练 + 30验证 + 30测试

#### 4.2 Baselines
| 基线 | 类型 | 特点 |
|------|------|------|
| k-Wave | 物理 | Gold standard, 3.2s/sample |
| V3 (PINN式) | 软约束 | physics_weight=0.01, 202K params |
| U-Net直接预测 | 纯学习 | 无物理, 端到端 |
| CNN+物理loss | PINN | 标准PINN方式 |

#### 4.3 Main Results Table
| 方法 | SSIM↑ | L1↓ | Physics Res.↓ | 推理时间 | 参数量 |
|------|-------|-----|--------------|---------|--------|
| k-Wave | 1.0 | 0.0 | 0 | 3.2s | 0 |
| V3(pw=0.01) | ? | ? | ~1e-4 | ~50ms | 202K |
| V3(pw=0.1) | ? | ? | ~1e-3 | ~50ms | 202K |
| U-Net | ? | ? | large | ~10ms | ~1M |
| **V4(ours)** | **>0.9** | **<0.05** | **≈0** | **<100ms** | **253K** |

#### 4.4 Ablation Study
1. **反对称MP vs 普通MP** — 空间一致性贡献
2. **物理先验(c_table+残差) vs 直接预测** — 先验结构的价值
3. **模型大小**: 50K / 150K / 253K / 500K
4. **时间步数**: 100 / 200 / 500步
5. **physics_weight消融**: 0 vs 0.01 vs 0.1（证明0最优）

#### 4.5 Physics Correctness Analysis
- physics_residual在训练过程中始终≈0（vs V3不为0）
- 能量守恒验证
- 声速场可视化（GNN学到的c与解剖结构对应）

### 5. Discussion (1 page)
- **为什么Physics-as-Forward优于PINN**: 不是程度问题，是范式问题
- **DPC-GNN统一框架**: 从软组织到声学，证明可跨域迁移
- **局限性**: 2D（3D扩展方向）、单一探头模型、频率依赖衰减简化
- **临床应用前景**: 术中导航、超声引导介入、训练模拟器

### 6. Conclusion (0.5 page)
- Physics-as-Forward是CT-to-US仿真的正确范式
- physics_weight=0不是超参数选择，是架构必然
- DPC-GNN框架可推广到更多物理域

---

## Key Figures

1. **图1: 架构总览** — 三段式：GNN编码器 / Leapfrog / DAS
2. **图2: Physics-as-Forward vs PINN对比** — 架构层面的根本区别
3. **图3: 反对称MP详解** — W_anti = W - W^T 的物理意义
4. **图4: B-mode结果对比** — GT / V4 / V3 / U-Net
5. **图5: 声速场可视化** — GNN预测的c(x,y)与CT解剖对应
6. **图6: 训练曲线** — SSIM + physics_residual随epoch变化
7. **图7: 消融热力图** — 各组件贡献

## Key Arguments（论文叙事线）

**开篇**: "物理仿真和深度学习之间不应该是权衡，而应该是融合。但融合的方式决定了一切。"

**核心论点**: "将物理约束放入loss函数（PINN范式）在概念上就是错误的——它把'必须遵守的定律'降级为'鼓励遵守的偏好'。DPC-GNN的'Physics-as-Forward'是唯一自洽的融合方式。"

**结尾**: "这不仅是一个超声仿真方法，更是一种将物理定律正确嵌入神经网络的范式。DPC-GNN从软组织力学到声学波传播的成功迁移，暗示这种范式可能适用于更广泛的物理域。"
