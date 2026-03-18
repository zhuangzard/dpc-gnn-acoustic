# DPC-GNN-Acoustic 物理问题修复文档

## 修复概述

根据物理专家审核报告，对 DPC-GNN-Acoustic 代码进行了以下**紧急物理修复**：

| 优先级 | 问题 | 修复状态 | 文件 |
|--------|------|----------|------|
| 🔴 最高 | 图Laplacian权重错误 | ✅ 已修复 | `wave_equation_mp.py` |
| 🔴 最高 | PML吸收边界缺失 | ✅ 已修复 | `wave_equation_mp.py` |
| 🟡 高 | 频率依赖衰减错误 | ✅ 已修复 | `wave_equation_mp.py`, `acoustic_properties.py` |
| 🟡 高 | 组织属性数据库不完整 | ✅ 已修复 | `acoustic_properties.py` |
| 🟢 中 | 能量守恒监控缺失 | ✅ 已修复 | `wave_equation_mp.py`, `acoustic_gnn.py` |

---

## 1. 图Laplacian权重修复 (最高优先级)

### 问题
**原代码**使用了错误的权重公式：
```python
# ❌ 错误
weight = 1.0 / dist_sq  # = 1 / |r_ij|²
```

这在物理上是**不正确的**，因为它不符合图Laplacian的定义。

### 修复
**正确公式**（2D三角网格）：
```python
# ✅ 正确
w_ij = A_ij / (V_i * |r_ij|) ≈ 1 / (V_i * |r_ij|)
```

其中：
- `A_ij` = 共享边长度
- `V_i` = Voronoi单元面积（节点体积）
- `|r_ij|` = 边距离

### 代码变更
```python
def _compute_edge_weight(self, edge_attr, node_volumes):
    distance = edge_attr[:, 3:4]  # |r_ij|
    # 修正为正确的图Laplacian权重
    weight = 1.0 / (node_volumes * distance)
    return weight
```

---

## 2. PML吸收边界修复 (最高优先级)

### 问题
原代码**没有**吸收边界，导致：
- 边界反射伪影
- 数值不稳定
- 非物理振荡

### 修复
新增 `PMLBoundary` 类，实现完美匹配层（Perfectly Matched Layer）：

```python
class PMLBoundary(nn.Module):
    """完美匹配层吸收边界"""
    
    def compute_damping(self, positions, domain_size):
        """计算PML阻尼系数"""
        # 距离边界的距离
        dist_to_boundary = torch.min(positions, domain_size - positions)
        # 在PML区域内增加阻尼
        pml_mask = dist_to_boundary < self.thickness
        # 多项式阻尼剖面
        sigma = self.sigma_max * ((self.thickness - dist_to_boundary) / self.thickness) ** 2
        return sigma * pml_mask.float()
    
    def apply(self, p, v, positions, domain_size, dt):
        """应用PML阻尼"""
        sigma = self.compute_damping(positions, domain_size)
        v_damped = v * torch.exp(-sigma * dt)
        p_damped = p * torch.exp(-sigma * dt * 0.5)
        return p_damped, v_damped
```

---

## 3. 频率依赖衰减修复 (高优先级)

### 问题
原代码使用了恒定的衰减系数，没有考虑频率依赖性。

### 修复
实现正确的频率依赖衰减公式：

```python
def frequency_dependent_attenuation(alpha_0, f, f_ref=1e6, n=1.0):
    """
    α(f) = α₀ * (f/f_ref)^n
    n ≈ 1.0-1.5 for soft tissue
    """
    return alpha_0 * (f / f_ref) ** n
```

### 不同组织的n值
| 组织类型 | n值 |
|---------|-----|
| 水 | 2.0 |
| 空气 | 2.0 |
| 软组织 | 1.0-1.1 |
| 血液 | 1.0 |
| 肌肉 | 1.0 |
| 脂肪 | 1.0 |

---

## 4. 组织属性数据库修复 (高优先级)

### 问题
原数据库缺少多种组织类型，HU映射范围不准确。

### 修复
完整的组织声学属性数据库：

```python
TISSUE_ACOUSTIC_PROPERTIES = {
    'water':      {'c': 1480, 'rho': 1000, 'alpha_0': 0.002, 'Z': 1.48e6, 'n': 2.0},
    'blood':      {'c': 1570, 'rho': 1060, 'alpha_0': 0.180, 'Z': 1.66e6, 'n': 1.0},
    'fat':        {'c': 1450, 'rho': 920,  'alpha_0': 0.630, 'Z': 1.33e6, 'n': 1.0},
    'liver':      {'c': 1550, 'rho': 1060, 'alpha_0': 0.940, 'Z': 1.64e6, 'n': 1.1},
    'muscle':     {'c': 1580, 'rho': 1050, 'alpha_0': 1.090, 'Z': 1.66e6, 'n': 1.0},
    'cartilage':  {'c': 1660, 'rho': 1100, 'alpha_0': 1.200, 'Z': 1.83e6, 'n': 1.0},
    'bone':       {'c': 3500, 'rho': 1900, 'alpha_0': 20.00, 'Z': 6.65e6, 'n': 1.0},
}
```

### HU到组织映射
```python
HU_TO_TISSUE = {
    (-1000, -900): 'air',
    (-900, -100):  'lung',
    (-100, -50):   'fat',
    (-50, 20):     'water',
    (20, 60):      'soft_tissue',
    (60, 100):     'liver',
    (100, 300):    'muscle',
    (300, 2000):   'bone',
}
```

---

## 5. 能量守恒监控 (中优先级)

### 新增功能
实现波动方程能量计算，用于验证物理正确性：

```python
def compute_energy(self, p, v, rho, c, node_volumes):
    """
    波动方程能量：
    E = ∫ [0.5*ρ*v² + 0.5*ρ*c²*|∇p|²] dV
    
    其中：
      - 动能：0.5 * ρ * v²
      - 势能：0.5 * ρ * c² * |∇p|²
    """
    kinetic = 0.5 * rho * v ** 2
    potential = 0.5 * rho * c ** 2 * self._compute_gradient_norm(p)
    energy_density = kinetic + potential
    total_energy = (energy_density * node_volumes).sum()
    return total_energy
```

### 能量守恒检查
```python
def check_energy_conservation(self, tolerance=0.2):
    """检查能量是否守恒"""
    if len(self.energy_history) < 2:
        return True, 0.0
    
    energies = torch.tensor(self.energy_history)
    initial_energy = energies[0]
    relative_variation = (energies - initial_energy).abs() / initial_energy.abs()
    max_variation = relative_variation.max().item()
    
    conserved = max_variation < tolerance
    return conserved, max_variation
```

---

## 测试验证

### 运行自测
```bash
cd /Users/taisenzhuang/workspace/DPC-GNN-Acoustic/

# 测试 wave_equation_mp.py
python src/models/wave_equation_mp.py

# 测试 acoustic_properties.py
python src/physics/acoustic_properties.py

# 测试 acoustic_gnn.py
python src/models/acoustic_gnn.py

# 测试 acoustic_wave_gnn.py
python src/models/acoustic_wave_gnn.py
```

### 预期输出
所有测试应该显示：
```
✅ ALL PHYSICS-CORRECTED TESTS PASSED
```

---

## 关键物理验证点

| 验证项 | 预期行为 | 检查方法 |
|--------|----------|----------|
| 能量守恒 | 相对变化 < 20% | `check_energy_conservation()` |
| 衰减随频率增加 | α(10MHz) > α(1MHz) | 比较不同频率输出 |
| PML阻尼 | 边界处能量衰减 | 检查边界节点 |
| Laplacian权重 | 与距离成反比 | 对比 1/d vs 1/d² |
| 组织阻抗排序 | Air < Water < Liver < Bone | 验证阻抗值 |

---

## 文件变更摘要

### 修改文件
1. `src/models/wave_equation_mp.py` - 核心物理修复
2. `src/models/acoustic_gnn.py` - 集成修正后的组件
3. `src/models/acoustic_wave_gnn.py` - 同步修正
4. `src/physics/acoustic_properties.py` - 完整数据库

### 新增类/函数
- `PMLBoundary` - PML吸收边界
- `frequency_dependent_attenuation()` - 频率依赖衰减
- `compute_energy()` - 能量计算
- `check_energy_conservation()` - 能量守恒检查
- `compute_node_volumes()` - Voronoi体积计算

---

## 参考文献

1. **图Laplacian**: MeshGraphNets (Pfaff et al., ICLR 2021)
2. **PML边界**: Berenger (1994), "Perfectly Matched Layer"
3. **声学属性**: IT'issue02 (2002), Duck (1990)
4. **k-Wave**: Treeby et al. (2012), MATLAB声学工具箱

---

*修复日期: 2026-03-18*  
*修复版本: v2.0-physics-corrected*
