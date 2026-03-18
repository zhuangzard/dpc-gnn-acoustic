# DPC-GNN-Acoustic 项目计划

## 📋 详细Checklist

### Phase 1: 文献调研（Week 1）
- [x] 创建论文文件夹
- [ ] 下载5篇核心论文PDF
- [ ] 每篇论文精读笔记
- [ ] 综合深度研究报告
- [ ] 技术路线确定

### Phase 2: 核心实现（Week 2-3）

#### 2.1 WaveEquationMP层
- [ ] 实现message passing核心
- [ ] 验证波动方程约束
- [ ] 验证可微分性（梯度检查）
- [ ] 单元测试

#### 2.2 AcousticGNN模型
- [ ] Encoder（CT→初始声压）
- [ ] Physics core（WaveEquationMP×K层）
- [ ] Decoder（声压→US图像）
- [ ] 端到端前向传播

#### 2.3 数据管道
- [ ] CT数据加载器
- [ ] Mesh图构建
- [ ] US图像生成（从SimUS导入）
- [ ] 数据增强

### Phase 3: 训练（Week 4-5）

#### 3.1 训练脚本
- [ ] 损失函数（MSE + 物理一致性）
- [ ] 优化器（Adam + LR schedule）
- [ ] 训练循环
- [ ] 验证循环
- [ ] TensorBoard/WandB日志

#### 3.2 实验配置
- [ ] 超参数搜索
- [ ] GPU训练（铁蛋儿/Colab）
- [ ] Checkpoint保存/加载
- [ ] 早停机制

### Phase 4: 评估（Week 6）

#### 4.1 定量评估
- [ ] SSIM vs SimUS
- [ ] PSNR vs SimUS
- [ ] 推理时间测试
- [ ] 内存占用测试

#### 4.2 物理正确性
- [ ] 波动方程残差检查
- [ ] 与k-Wave对比
- [ ] 声影/增强效应验证

#### 4.3 消融实验
- [ ] 有/无物理约束对比
- [ ] 不同MP层数对比
- [ ] 不同时间步长对比

### Phase 5: 真实数据验证（Week 7）
- [ ] 真实US数据收集
- [ ] 医生标注验证
- [ ] 临床场景测试

### Phase 6: 论文写作（Week 8）
- [ ] 方法部分
- [ ] 实验部分
- [ ] 结果可视化
- [ ] 补充材料
- [ ] 投稿准备

---

## 📅 里程碑

| 日期 | 里程碑 |
|------|--------|
| Week 1 | 文献调研完成，技术路线确定 |
| Week 3 | 核心模型实现完成 |
| Week 5 | 训练完成，初步结果 |
| Week 7 | 真实数据验证完成 |
| Week 8 | 论文投稿 |

---

## 🎯 关键交付物

1. **代码**：GitHub repo（dpc-gnn-acoustic）
2. **数据**：训练好的模型 + 实验结果
3. **论文**：MICCAI/TMI投稿
4. **文档**：完整技术文档

---

## ⚠️ 风险与应对

| 风险 | 应对策略 |
|------|---------|
| WaveEquationMP不可微分 | 使用PyTorch Geometric的autograd |
| 训练不稳定 | 梯度裁剪 + 学习率预热 |
| 真实数据不足 | 先用SimUS数据，逐步迁移 |
| GPU资源不足 | 用铁蛋儿 + Colab MCP |

---

## 📊 资源需求

- **GPU**：铁蛋儿（主力）+ Colab（备用）
- **存储**：~50GB（数据+模型）
- **时间**：8周全职投入

---

Created: 2026-03-17
Owner: 二丫
