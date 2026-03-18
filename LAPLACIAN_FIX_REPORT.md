# DPC-GNN-Acoustic Laplacian权重修复报告

## 问题（BLOCKER）

**原始代码声称使用公式**：`w_ij = 1/(V_i * |r_ij|)`

**实际代码**：`w_ij = 1/|r_ij|` （**体积 V_i 完全未使用！**）

### 具体位置
1. `src/models/wave_equation_mp.py` - `_compute_edge_weight` 方法
2. `src/models/acoustic_wave_gnn.py` - `_compute_edge_weight` 方法

## 修复内容

### 核心修复
在 `_compute_edge_weight` 方法中实际使用源节点体积 V_i：

```python
# ✅ 修复后
if node_volumes is not None and edge_index is not None:
    src = edge_index[0]  # 源节点索引
    V_i = node_volumes[src]  # 获取源节点体积
    if V_i.dim() == 1:
        V_i = V_i.unsqueeze(-1)  # 确保形状正确
    
    weight = 1.0 / (V_i * distance)  # ✅ 正确公式
else:
    weight = 1.0 / distance  # 降级方案
```

### 架构调整
为避免 PyG MessagePassing 的广播问题，改为：
1. 在 `forward` 方法中预计算权重
2. 将预计算的权重通过 `propagate` 传递
3. `message` 方法直接使用预计算权重

## 关键技术细节

### 维度陷阱（已解决）
**问题**：`node_volumes` 形状为 `(N, 1)` 时，`node_volumes[src].unsqueeze(-1)` 会导致过度广播
- `node_volumes[src]` 形状：`(E, 1)`
- 再次 `unsqueeze(-1)` 后：`(E, 1, 1)`
- 与 `distance (E, 1)` 相乘后：`(E, E, 1)` ❌

**解决**：只在 `dim == 1` 时才 `unsqueeze`

### PyG MessagePassing 限制（已规避）
PyG 会对某些参数名和实例变量进行特殊处理，导致形状广播错误：
- ❌ 存储实例变量 `self._edge_index`
- ❌ 传递名为 `edge_weight` 的参数
- ✅ 在 forward 中预计算，使用非保留参数名

## 验证测试

### 测试 1：公式验证 ✅
```
Expected weight: 1/(V_i * |r_ij|) = 5.886623
Actual weight: 5.886622
Match: True ✅
```

### 测试 2：体积敏感性 ✅
不同 V_i 值产生不同权重：
```
V_i =  1.0  →  weight = 9.999999
V_i =  2.0  →  weight = 5.000000
V_i =  5.0  →  weight = 2.000000
V_i = 10.0  →  weight = 1.000000
```
每个 V_i 都产生唯一权重 ✅

### 测试 3：自测试 ✅
- AcousticWaveGNN forward pass ✅
- Gradient flow ✅
- WaveEquationMP standalone ✅
- Frequency dependence ✅

## 文件修改清单

1. **src/models/wave_equation_mp.py**
   - 修复 `_compute_edge_weight`：实际使用 V_i
   - 调整 `forward`：预计算权重
   - 更新 `message`：使用预计算权重

2. **src/models/acoustic_wave_gnn.py**
   - 同上修复

3. **新增测试文件**
   - `test_laplacian_fix.py` - 验证公式正确性
   - `test_volume_sensitivity.py` - 验证体积敏感性
   - 其他调试脚本（可删除）

## 物理正确性

修复后的 Laplacian 权重公式：
```
w_ij = (Z_j/Z_i) * exp(-α(f)*|r_ij|) / (V_i * |r_ij|)
```

符合图 Laplacian 的物理意义：
- **V_i**：Voronoi 胞体积（节点 i 的体积权重）
- **|r_ij|**：边长度（几何权重）
- **Z_j/Z_i**：声阻抗比（材料属性）
- **exp(-α(f)*|r_ij|)**：频率相关衰减

## 下一步建议

1. ✅ **立即测试**：在真实数据上验证修复效果
2. ⚠️ **数值稳定性**：当前实现可能出现极大值（1e15, 1e31），需要：
   - 添加权重裁剪
   - 检查 V_i 的数值范围
   - 考虑归一化策略
3. 📝 **文档更新**：更新代码注释，明确说明修复内容

## 结论

✅ **修复成功**：Laplacian 权重现在正确使用 `1/(V_i * |r_ij|)` 公式
✅ **测试通过**：所有验证测试通过
⚠️ **需要关注**：数值稳定性问题（可能需要后续优化）

---
修复者：GLM  
日期：2026-03-18  
状态：✅ COMPLETE
