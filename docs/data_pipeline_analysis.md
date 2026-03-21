# DPC-GNN-Acoustic 数据管线深度解析

**作者**: 三丫（研究助手）  
**日期**: 2026-03-21  
**版本**: v1.0  
**项目**: DPC-GNN-Acoustic — 用GNN替代k-Wave超声波仿真求解器  
**数据集**: TRUSTED（48个肾脏CT + 59个配对超声）

---

## 摘要

本报告深入解析DPC-GNN-Acoustic项目的数据管线，重点回答两个核心技术问题：

1. **HU→声学参数映射**：为什么需要这个映射？CT图像到底提供了什么信息？
2. **CT-超声配准**：CT生成的3D体积如何与超声图像的物理位置对应？

这两个问题直接决定了训练数据的质量上界，理解其局限性是正确设计GNN训练策略的前提。

---

## 目录

1. [背景：物理场景与数据流](#1-背景)
2. [问题一：HU声学映射详解](#2-hu声学映射详解)
3. [问题二：CT-超声配准问题](#3-ct-超声配准问题)
4. [完整数据管线流程图](#4-完整数据管线流程图)
5. [对GNN训练策略的影响](#5-对gnn训练策略的影响)
6. [结论与下一步建议](#6-结论与下一步建议)
7. [参考文献](#7-参考文献)

---

## 1. 背景

### 1.1 项目目标

用图神经网络（GNN）替代k-Wave传统有限差分波场求解器。训练GNN需要大量"输入→物理响应"数据对：

```
输入: 声学属性图 (c, ρ, α) + 初始激励源
输出: 波场时间序列 p(x, y, t)
```

### 1.2 TRUSTED数据集结构

```
TRUSTED/
├── patient_001/
│   ├── CT/                                        # CT DICOM文件（3D volume）
│   ├── US/                                        # 超声图像序列（2D B-mode）
│   └── Initial_Transforms_from_noisy_landmarks/   # 配准变换矩阵
├── patient_002/
│   └── ...
... (共48个CT + 59个US配对，来自同一批病人)
```

### 1.3 核心挑战一览

| 挑战 | 具体问题 | 对训练的影响 |
|------|---------|------------|
| 物理量不匹配 | CT测X射线衰减，超声需要声速/密度/声学衰减 | 必须做HU→声学映射，引入系统误差 |
| 多模态配准 | CT和US在不同时间/体位采集 | 坐标系对齐误差达15-25mm |
| 2D vs 3D | 超声是2D切面，CT是3D体积 | 需要提取对应切片，层位置对应不精确 |
| 分辨率差异 | CT 0.7mm vs US 0.3-1mm | 缩放后信息损失 |

---

## 2. HU声学映射详解

### 2.1 CT提供的是什么？

**CT扫描的物理基础**

CT利用X射线从不同角度穿透人体，通过滤波反投影（FBP）或迭代重建算法重建3D密度图。每个体素（voxel）的值是该位置对X射线的**线性衰减系数** μ，经过归一化后表示为Hounsfield Unit（HU）：

$$\text{HU} = \frac{\mu_{\text{tissue}} - \mu_{\text{water}}}{\mu_{\text{water}}} \times 1000$$

其中：μ_water ≈ 0.019 mm⁻¹（水在70 keV的衰减系数）

**常见组织的HU参考值**：

| 组织类型 | HU范围 | 物理含义 |
|---------|--------|---------|
| 空气 | -1000 | X射线几乎不衰减 |
| 脂肪 | -100 ~ -50 | 低电子密度，弱衰减 |
| 水 | 0 | 参考基准 |
| 肾脏髓质 | +10 ~ +25 | 中低密度 |
| 肾脏皮质 | +20 ~ +40 | 中等密度（**注意：与脂肪/肌肉重叠**） |
| 肌肉 | +20 ~ +80 | 中等密度 |
| 肝脏 | +40 ~ +70 | 中等密度 |
| 血液 | +30 ~ +50 | 中等密度 |
| 松质骨 | +100 ~ +400 | 中高密度 |
| 皮质骨 | +400 ~ +1000 | 高密度，强衰减 |
| 肾结石 | +200 ~ +800 | 高密度（钙化） |

**CT给我们的本质**：

```
CT图像 = X射线密度图（电子密度的空间分布）

✅ 能提供:
   - 组织形态信息（边界、结构）
   - 相对密度分布（用于区分组织类型）
   - 高空间分辨率（0.5-1mm isotropic）

❌ 不能直接提供:
   - 声速 c（m/s）
   - 声阻抗 Z = ρ·c
   - 声学衰减系数 α（dB/MHz/cm）
```

> **根本原因**：X射线与组织的相互作用（光电效应、康普顿散射）和声波与组织的相互作用（弹性变形、粘弹性耗散）是完全不同的物理过程，由不同的材料特性决定。

### 2.2 超声波需要什么？

超声波在介质中的传播由线性声波方程描述（k-Wave使用的一阶速度-压力方程组）：

$$\frac{\partial \mathbf{u}}{\partial t} = -\frac{1}{\rho_0}\nabla p$$

$$\frac{\partial p}{\partial t} = -\rho_0 c^2 \nabla \cdot \mathbf{u} - \frac{\eta}{\rho_0}\nabla^2 p$$

求解这组方程需要以下**三类空间分布参数**：

**① 声速 c（Sound Speed）**
- 单位：m/s
- 物理含义：声波在介质中的传播速度
- 决定：波的时延（time-of-flight）、折射（Snell定律）、相位

**② 密度 ρ（Mass Density）**
- 单位：kg/m³
- 物理含义：介质的质量密度（注意：≠CT测量的X射线密度！）
- 声阻抗 Z = ρ × c，决定：界面反射率 R = (Z₂ - Z₁)/(Z₂ + Z₁)

**③ 声学衰减系数 α（Acoustic Attenuation）**
- 单位：dB/(MHz·cm) 或 Np·m⁻¹·MHz⁻ᵇ
- 物理含义：声波传播过程中的能量耗散（热转化）
- 通常建模为幂律：α(f) = α₀ × f^b，其中b≈1.0-2.0

**常见组织的声学参数**（文献体外测量值）：

| 组织类型 | 声速 c (m/s) | 密度 ρ (kg/m³) | 声学衰减 α (dB/MHz/cm) | 数据来源 |
|---------|-------------|---------------|----------------------|---------|
| 水（37°C） | 1524 | 993 | 0.002 | Bilaniuk 1993 |
| 脂肪 | 1478 ± 15 | 911 ± 20 | 0.60 ± 0.15 | Duck 1990 |
| 肌肉（平行） | 1547 ± 21 | 1050 ± 30 | 1.09 ± 0.21 | Duck 1990 |
| 肌肉（垂直） | 1571 ± 18 | 1050 ± 30 | 0.74 ± 0.20 | Duck 1990 |
| 肝脏 | 1578 ± 14 | 1060 ± 30 | 0.45 ± 0.12 | Duck 1990 |
| 肾脏（皮质） | 1560 ± 25 | 1040 ± 40 | 1.0 ± 0.3 | Mast 2000 |
| 肾脏（髓质） | 1545 ± 20 | 1020 ± 30 | 0.5 ± 0.2 | Mast 2000 |
| 血液 | 1584 ± 10 | 1060 ± 15 | 0.14 ± 0.02 | Duck 1990 |
| 皮质骨 | 3500 ± 400 | 1900 ± 200 | 22 ± 8 | IT'IS Foundation |
| 空气 | 343 | 1.21 | ~0 | — |

*体外测量，37°C（体温），文献报告存在差异*

### 2.3 为什么必须做HU→声学映射？

**核心矛盾**：

```
我们拥有:  CT图像（HU值的空间分布）
我们需要:  声学参数图（c, ρ, α的空间分布）
两者不是同一物理量，无法直接互换
```

**解决方案**：利用HU作为**组织类型的代理指标（proxy）**，通过文献测量值建立查找表（Lookup Table）：

```
HU值 ──[分类]──► 组织类型 ──[查表]──► 声学参数 (c, ρ, α)
```

这是一种**间接估计**，不是精确测量。类比：知道某地区的年均降水量（HU），去估计该地的植被覆盖率（声学参数）——有相关性，但不是一一对应。

**典型分段线性映射实现**：

```python
def hu_to_acoustic_params(hu_value: float) -> tuple[float, float, float]:
    """
    HU → 声学参数映射（分段线性近似）
    
    Returns:
        c     (float): 声速 [m/s]
        rho   (float): 密度 [kg/m³]
        alpha (float): 声学衰减系数 [dB/MHz/cm]
    
    References:
        Duck (1990), Mast (2000), IT'IS Foundation Database
    """
    # 空气
    if hu_value <= -900:
        return 343.0, 1.21, 0.0
    
    # 肺/气体过渡区
    elif hu_value <= -100:
        t = (hu_value + 900) / 800.0  # 线性插值参数
        c     = 343   + t * (1400 - 343)
        rho   = 1.21  + t * (920  - 1.21)
        alpha = 0.0   + t * 0.6
        return c, rho, alpha
    
    # 脂肪
    elif hu_value <= -50:
        return 1478.0, 911.0, 0.60
    
    # 软组织区（水/血液/脂肪过渡 → 肌肉/器官）
    elif hu_value <= 80:
        t     = (hu_value + 50) / 130.0
        c     = np.interp(hu_value, [-50, 0, 40, 80],
                                    [1480, 1524, 1560, 1578])  # 水→肾脏→肝脏
        rho   = np.interp(hu_value, [-50, 0, 80],
                                    [950, 1000, 1060])
        alpha = np.interp(hu_value, [-50, 0, 40, 80],
                                    [0.3, 0.003, 1.0, 0.45])
        return c, rho, alpha
    
    # 松质骨过渡区
    elif hu_value <= 400:
        t     = (hu_value - 80) / 320.0
        c     = 1578 + t * (2500 - 1578)
        rho   = 1060 + t * (1600 - 1060)
        alpha = 0.45 + t * (10.0 - 0.45)
        return c, rho, alpha
    
    # 皮质骨
    else:
        return 3500.0, 1900.0, 22.0
```

### 2.4 映射的根本局限性（关键）

#### 局限1：HU分辨率不足——组织类型混叠

不同组织的HU范围高度重叠，导致分类错误：

```
组织类型对比（声速差异 vs HU重叠）:

   HU: -50    0    20   40   60   80
        │    │    │    │    │    │
脂肪:  [══════════]              声速: 1478 m/s
肾髓质:      [════════]          声速: 1545 m/s  → Δc = 67 m/s
肾皮质:          [════════]      声速: 1560 m/s  → Δc = 82 m/s
肌肉:             [═════════════] 声速: 1547 m/s
肝脏:                  [═══════]  声速: 1578 m/s

⚠️ 问题: HU=30 的像素可能是:
   - 肾脏皮质（c=1560）
   - 肌肉（c=1547）
   - 早期血栓（c≈1584）
   仅靠HU无法区分 → 声速估计误差 ±40 m/s
```

#### 局限2：个体差异（Intersubject Variability）

文献中的声学参数是**群体均值**，实际个体存在显著变异：

| 组织 | 声速均值 | 标准差 | 极端个体偏差 |
|------|---------|--------|------------|
| 肝脏 | 1578 m/s | ±14 m/s | ±50 m/s |
| 肾脏皮质 | 1560 m/s | ±25 m/s | ±80 m/s |
| 脂肪 | 1478 m/s | ±15 m/s | ±50 m/s |
| 肌肉 | 1547 m/s | ±21 m/s | ±60 m/s |

声速偏差对波场的影响估算：
- 传播路径 L = 100mm，声速偏差 Δc = 25 m/s
- 时延误差 Δt = L×Δc/c² ≈ 100mm × 25/(1560²) ≈ 1.0 μs
- 在5MHz超声中，1 μs ≈ 5个波长的相位误差

#### 局限3：病理组织的不确定性

TRUSTED数据集包含肾脏病人（可能有囊肿、结石、肿瘤）：

```
正常肾皮质:       HU ≈ 20-40,  c ≈ 1560 m/s
肾脏囊肿（浆液）:  HU ≈ 0-20,   c ≈ 1480-1510 m/s  → HU接近水，但有囊壁
肾细胞癌（增强）:  HU ≈ 30-100, c ≈ ???            → 无可靠文献值
肾结石（草酸钙）:  HU ≈ 300-800, c ≈ 4000-6000 m/s → 极端高声速！

⚠️ 结石会产生强声影（acoustic shadowing），HU值虽高，但体积小，
   在256×256的缩放图中可能只占1-2个像素 → 被平均掉，仿真中丢失
```

#### 局限4：3D各向异性被忽略

骨骼和肌肉具有**声学各向异性**（声速随方向变化）：

```
肌肉:
  - 平行纤维方向: c = 1547 m/s
  - 垂直纤维方向: c = 1571 m/s
  - 差值: 24 m/s（~1.5%）

皮质骨:
  - 纵向: c ≈ 3500 m/s
  - 横向: c ≈ 1700 m/s
  - 差值: 1800 m/s（巨大！）

CT的HU值是标量，无方向信息 → 各向异性被忽略
```

**误差量化汇总**：

| 误差来源 | 声速误差量级 | 对波场的影响 |
|---------|-----------|------------|
| 组织分类错误 | ±50-200 m/s | 波前位置误差 1-5 mm |
| 个体差异 | ±15-25 m/s | 时延误差 ±0.5-1.5% |
| 病理组织 | ±100 m/s以上 | 局部波形畸变 |
| 各向异性（骨骼） | ±1000+ m/s | 骨骼周围严重失真 |
| **总体系统误差** | **±50-300 m/s** | **波场质量受限** |

> 💡 **这正是GNN的价值所在**：如果GNN能从大量数据中学会真实的传播规律，就能在一定程度上"补偿"HU映射的系统误差，实现数据驱动的隐式物理校正。

### 2.5 CT 2D切片的含义

**CT Volume的3D结构**：

```
CT Volume结构（TRUSTED典型值）:
┌─────────────────────────────────────┐
│  矩阵大小:  512 × 512 pixels         │
│  层数:      约200-500层（axial）      │
│  像素间距:  ~0.7 mm × 0.7 mm        │  ← 面内高分辨率
│  层间距:    ~1.0-3.0 mm             │  ← 轴向分辨率低
│  视野(FOV): ~360 mm × 360 mm        │
│  数据类型:  12-bit HU（-1024~+3071） │
└─────────────────────────────────────┘

三维坐标系（DICOM标准方向）:
   x → 从右到左（Right→Left）
   y → 从前到后（Anterior→Posterior）
   z → 从下到上（Inferior→Superior）

2D轴向切片 = 固定z，取x-y平面
```

**提取2D切片并转换为声学参数图**：

```python
import numpy as np
import SimpleITK as sitk
import cv2

# 步骤1: 加载CT Volume（DICOM序列 → SimpleITK Image）
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(ct_dicom_dir)
reader.SetFileNames(dicom_names)
ct_image = reader.Execute()

# 获取CT的元数据
spacing = ct_image.GetSpacing()      # (dx, dy, dz) in mm
size    = ct_image.GetSize()         # (Nx, Ny, Nz) in pixels
origin  = ct_image.GetOrigin()       # (x0, y0, z0) in mm
print(f"CT spacing: {spacing}")      # e.g., (0.68, 0.68, 1.5) mm
print(f"CT size:    {size}")         # e.g., (512, 512, 390)

# 步骤2: 提取轴向切片（选取肾脏所在层）
ct_array = sitk.GetArrayFromImage(ct_image)  # shape: (Nz, Ny, Nx)
slice_idx = 200  # 第200层（包含肾脏中部，需手动或自动选择）
ct_slice_hu = ct_array[slice_idx, :, :]      # shape: (512, 512), HU值

# 步骤3: 缩放到仿真分辨率 256×256
# 使用双线性插值（注意HU是连续值，不能用最近邻）
ct_slice_256 = cv2.resize(ct_slice_hu.astype(np.float32),
                           (256, 256),
                           interpolation=cv2.INTER_LINEAR)
# 缩放后像素间距: 0.68mm × (512/256) = 1.36mm/pixel

# 步骤4: 逐像素HU → 声学参数映射
# 向量化实现（比逐像素循环快100x）
c_map     = np.zeros((256, 256), dtype=np.float32)
rho_map   = np.zeros((256, 256), dtype=np.float32)
alpha_map = np.zeros((256, 256), dtype=np.float32)

# 分区域赋值（示例，实际需要更精细的插值）
mask_air   = ct_slice_256 <= -900
mask_fat   = (ct_slice_256 > -100) & (ct_slice_256 <= -50)
mask_soft  = (ct_slice_256 > -50)  & (ct_slice_256 <= 80)
mask_bone  = ct_slice_256 > 400

c_map[mask_air]  = 343.0;    rho_map[mask_air]  = 1.21;   alpha_map[mask_air]  = 0.0
c_map[mask_fat]  = 1478.0;   rho_map[mask_fat]  = 911.0;  alpha_map[mask_fat]  = 0.60
c_map[mask_soft] = 1540.0;   rho_map[mask_soft] = 1040.0; alpha_map[mask_soft] = 0.75
c_map[mask_bone] = 3500.0;   rho_map[mask_bone] = 1900.0; alpha_map[mask_bone] = 22.0
# 过渡区使用线性插值...

# 步骤5: 输入k-Wave仿真
# kwave_simulation(c_map, rho_map, alpha_map, source_waveform) → p(x,y,t)
```

**分辨率约束检查**：

```
k-Wave稳定性条件: dx ≤ λ_min / 6
（Nyquist采样要求：每波长至少6个采样点）

当前设置: dx = 1.36 mm
最低满足条件的频率（水中）:
  f_max = c / (6 × dx) = 1524 / (6 × 0.00136) ≈ 186 kHz

结论: 256×256 @ 1.36mm 仅适合 <200 kHz 的低频超声！
      临床超声（1-5 MHz）需要更精细的网格，或使用pseudospectral方法

实际操作建议:
  - 选项A: 用1-2 MHz仿真（较接近当前分辨率的极限）
  - 选项B: 提高分辨率到512×512（dx=0.68mm，支持到约370 kHz）
  - 选项C: 用k-Wave的k-space校正（伪谱方法，dx可放宽到λ/2）
```

---

## 3. CT-超声配准问题

### 3.1 配准问题的物理本质

CT和超声图像在**三个维度**上都不一致，无法直接对齐：

```
  CT采集条件:                超声采集条件:
  ┌──────────────────┐       ┌──────────────────┐
  │ 设备: CT扫描仪    │       │ 设备: 超声机      │
  │ 体位: 仰卧位      │       │ 体位: 侧卧/坐位   │
  │ 呼吸: 屏气（8-15s)│       │ 呼吸: 自由呼吸    │
  │ 时间: T₁         │       │ 时间: T₂ (T₁+几天) │
  │ 范围: 全腹部      │       │ 范围: 局部肾脏    │
  │ 坐标系: DICOM    │       │ 坐标系: 探头坐标  │
  └──────────────────┘       └──────────────────┘
            │                          │
            └────────── 需要配准 ───────┘
            
数学表示: x_US = T(x_CT)
刚体变换: T(x) = R·x + t  (6个自由度: 3平移 + 3旋转)
仿射变换: T(x) = A·x + t  (12个自由度，包含缩放和剪切)
非刚体变换: T(x) = A·x + t + D(x)  (无限自由度，需要正则化)
```

### 3.2 TRUSTED数据集的配准方案

TRUSTED提供了基于**解剖标志点**（Anatomical Landmarks）的初始配准：

```
Initial_Transforms_from_noisy_landmarks/
├── CT_to_US_transform_patient001.tfm    # ITK变换文件（4×4矩阵）
├── landmarks_CT_patient001.fcsv         # CT中的标志点坐标（mm）
└── landmarks_US_patient001.fcsv         # US中的对应标志点坐标（mm）

典型标志点（肾脏相关）:
  1. 肾门（Renal Hilum）          — 最稳定的标志点
  2. 肾上极（Upper Pole）         — CT清晰，US有时被遮挡
  3. 肾下极（Lower Pole）         — CT清晰，US有时被遮挡
  4. 前缘最突点（Anterior Apex）  — 易受探头压迫影响
  5. 后缘最突点（Posterior Apex） — CT清晰，US深度可能不够
  6. 肾盂中心（Renal Pelvis）     — CT中空腔清晰，US中回声弱
```

**配准流程**：

```
Step 1: 放射科医生在CT volume上标注6-8个解剖标志点
        → 坐标以mm为单位，保存为.fcsv格式

Step 2: 超声医生在US图像上标注对应标志点
        → 坐标以mm为单位（超声的pixel→mm换算需标定）

Step 3: 最小化点对距离，求解最优刚体变换:
        min_{R,t} Σᵢ ||R·p_CT_i + t - p_US_i||²
        → 闭合解（SVD分解）

Step 4: 保存变换矩阵 T = [R|t] (4×4齐次矩阵)

Step 5: 用T将CT体积重采样到US坐标系:
        CT_resampled = Resample(CT_volume, T, US_image_grid)
```

**读取和应用变换的代码**：

```python
import SimpleITK as sitk

# 读取初始变换矩阵
transform = sitk.ReadTransform('CT_to_US_transform_patient001.tfm')
print(f"变换类型: {transform.GetName()}")  # e.g., "Euler3DTransform"

# 将CT volume重采样到超声图像的坐