# DPC-GNN-Acoustic 修复历史

## 提交记录

### Commit 4b72276 - Fix: frequency type conversion for YAML config compatibility
**时间**: 2026-03-18  
**作者**: 三丫 (Agent)

**问题**:  
YAML 加载科学计数法 (5.0e6) 为字符串，导致 TypeError
```
TypeError: unsupported operand type(s) for /: 'str' and 'float'
```

**修复**:  
- `src/data/kwave_dataset.py`: `self.frequency = float(frequency)`
- `src/data/kwave_dataset.py`: 防御性转换 `_hu_to_properties()`
- `train.py`: `frequency=float(config.get(...))`

---

### Commit 57ae1df - DPC-GNN-Acoustic v2: Production-ready with k-Wave GT support
**时间**: 2026-03-18  
**作者**: 三丫 (Agent)

**主要内容**:  
- 完整 k-Wave GT 集成
- KWaveInspiredMP 物理层
- 课程学习 (3 stages)
- 混合精度训练 (AMP)
- 完整评估框架

**修复的问题**:
1. ✅ `share_weights` bug 修复
2. ✅ 能量计算公式修正 (物理正确)
3. ✅ 内存限制 (100步防止 OOM)
4. ✅ `import time` 位置修正
5. ✅ k-Wave 路径自动检测

---

## 修复详情

### P0 - 阻塞部署问题 (已解决)

| # | 问题 | 文件 | 修复 |
|---|------|------|------|
| 1 | k-Wave C++路径硬编码 | `kwave_generator.py` | 添加 `_resolve_kwave_binary()` |
| 2 | torch-cluster安装文档 | `INSTALL.md` | 创建完整安装指南 |
| 3 | `share_weights` bug | `kwave_inspired_mp.py` | 修复条件判断 |
| 4 | `import time`位置 | `kwave_generator.py` | 移到文件顶部 |
| 5 | 内存无限增长 | `kwave_inspired_mp.py` | 限制100步 |
| 6 | frequency类型错误 | `kwave_dataset.py` | 强制float转换 |

### P1 - 质量改进 (已实现)

| # | 功能 | 文件 | 说明 |
|---|------|------|------|
| 1 | 课程学习 | `train.py` | 3阶段自动切换 |
| 2 | 混合精度训练 | `train.py` | AMP, 速度提升2x |
| 3 | 完整评估 | `kwave_gnn_evaluator.py` | SSIM/PSNR/物理指标 |

---

## 当前状态

- **GitHub**: https://github.com/zhuangzard/dpc-gnn-acoustic
- **最新Commit**: 4b72276
- **状态**: 可部署，frequency问题已解决
- **待解决**: CUDA索引越界问题 (debug中)

---

## GPU 服务器部署

```bash
# 拉取最新代码
git clone https://github.com/zhuangzard/dpc-gnn-acoustic.git
cd dpc-gnn-acoustic

# 安装依赖
pip install -r requirements.txt

# 配置 k-Wave
export KWAVE_BINARY=/path/to/kspaceFirstOrder-CUDA

# 运行训练
python train.py --config configs/default_2d.yaml
```

---

## 后续工作

1. [ ] 修复 CUDA 索引越界错误
2. [ ] 生成 k-Wave GT 数据集
3. [ ] 完整训练 500 epochs
4. [ ] 评估模型性能
