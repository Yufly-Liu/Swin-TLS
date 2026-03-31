# 快速开始 - 细线条恢复训练

## 问题描述

你的场景：
- ✅ 真值：黑色背景 + 白色细线条
- ❌ 输入：大量白斑噪声
- ⚠️ 问题：模型去掉白斑，但不恢复线条

## 解决方案

使用**组合损失函数**，强制模型关注线条恢复。

## 快速开始（3步）

### 步骤1：选择配置文件

我已经为你创建了专用配置：`configs/config_line_restoration.yaml`

或者修改现有配置 `configs/config_multi_dataset.yaml`：

```yaml
training:
  loss:
    type: "combined"              # 使用组合损失
    params:
      foreground_weight: 15.0     # 线条权重（15倍）
      edge_weight: 0.5            # 边缘权重
      focal_gamma: 2.0            # Focal 参数
      use_edge: true
      use_focal: true
```

### 步骤2：开始训练

```bash
# 使用专用配置
python main.py train --config configs/config_line_restoration.yaml

# 或使用修改后的配置
python main.py train --config configs/config_multi_dataset.yaml
```

### 步骤3：监控效果

```bash
# 另一个终端启动 TensorBoard
tensorboard --logdir=./experiments

# 浏览器打开 http://localhost:6006
# 查看 IMAGES 标签页，观察线条是否清晰
```

## 预期效果

使用组合损失后：

✅ 白斑噪声被有效去除  
✅ 细线条清晰可见  
✅ 线条边缘锐利  
✅ PSNR/SSIM 指标提升  

## 参数调优

如果效果不理想，调整这些参数：

### 线条太模糊？

增加前景权重：
```yaml
foreground_weight: 20.0  # 从 15 增加到 20
```

### 边缘不够锐利？

增加边缘权重：
```yaml
edge_weight: 1.0  # 从 0.5 增加到 1.0
```

### 训练不稳定？

降低学习率：
```yaml
learning_rate: 0.0005  # 从 0.001 降低到 0.0005
```

## 对比实验

建议运行对比实验，验证改进效果：

```bash
# 实验1：标准 MSE（基线）
# 修改配置，注释掉 loss 部分，使用模型默认损失
python main.py train --config configs/config_baseline.yaml

# 实验2：组合损失（优化）
python main.py train --config configs/config_line_restoration.yaml

# 对比结果
python main.py compare --experiments <exp_id_1> <exp_id_2>
```

## 技术原理

### 为什么有效？

**标准 MSE 的问题**：
- 黑色背景占 90%+ 像素
- 模型只要把白斑变黑就能获得低 loss
- 细线条（<10% 像素）被忽略

**组合损失的优势**：
1. **加权 MSE**：线条区域权重 15 倍，强制关注
2. **边缘损失**：明确优化边缘清晰度
3. **Focal 损失**：自动识别难恢复的区域

### 数学表达

```python
# 标准 MSE
Loss = mean((pred - target)²)

# 组合损失
Loss = WeightedMSE + 0.5 * EdgeLoss + 0.5 * FocalLoss

# 其中 WeightedMSE
weight_map = {15.0 if 线条, 1.0 if 背景}
WeightedMSE = mean(weight_map * (pred - target)²)
```

线条区域的误差被放大 15 倍，模型被迫学习线条恢复。

## 更多选项

### 损失函数类型

```yaml
# 1. 加权 MSE（简单有效）
loss:
  type: "weighted_mse"
  params:
    foreground_weight: 10.0

# 2. 边缘保持（锐化线条）
loss:
  type: "edge_preserving"
  params:
    edge_weight: 0.5

# 3. 组合损失（最强，推荐）⭐
loss:
  type: "combined"
  params:
    foreground_weight: 15.0
    edge_weight: 0.5
    focal_gamma: 2.0

# 4. 感知加权（自适应）
loss:
  type: "perceptual_weighted"
  params:
    percentile: 90.0
    max_weight: 20.0
```

### 模型选择

```yaml
# U-Net（推荐，细节恢复好）
model:
  name: "unet"
  params:
    base_features: 64
    depth: 4

# DnCNN（快速，参数少）
model:
  name: "dncnn"
  params:
    num_layers: 17
    num_features: 64

# Restormer（最强，但慢）
model:
  name: "restormer"
  params:
    dim: 48
    num_blocks: [4, 6, 6, 8]
```

## 常见问题

**Q: 训练变慢了？**  
A: 组合损失计算量稍大（+10-20%），但效果提升明显。

**Q: 线条还是不清晰？**  
A: 增加 `foreground_weight` 到 20-30，或增加训练轮数到 150-200。

**Q: 出现伪影？**  
A: 权重过大，降低 `foreground_weight` 或 `edge_weight`。

**Q: 如何验证改进？**  
A: 在 TensorBoard 的 IMAGES 标签页对比训练前后的效果。

## 详细文档

- 📖 [损失函数选择指南](LOSS_FUNCTIONS_GUIDE.md) - 详细原理和参数调优
- 📖 [TensorBoard 使用指南](README.md#查看训练过程) - 监控训练过程
- 📖 [完整 README](README.md) - 项目完整文档

## 测试损失函数

运行测试脚本验证损失函数是否正常工作：

```bash
python test_loss_functions.py
```

你会看到组合损失能更有效地区分"保留线条"和"丢失线条"的预测（差异提升 25 倍）。

---

**开始训练吧！** 🚀

```bash
python main.py train --config configs/config_line_restoration.yaml
```
