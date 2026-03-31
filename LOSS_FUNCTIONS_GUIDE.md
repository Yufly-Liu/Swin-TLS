# 损失函数选择指南 - 细线条恢复优化

## 问题分析

你的场景：
- **真值**：黑色背景 + 白色细线条
- **输入**：大量白斑噪声
- **问题**：模型只去除白斑（变黑），不恢复线条

**根本原因**：类别不平衡
- 黑色背景占 90%+ 像素
- 白色线条只占 <10% 像素
- 标准 MSE 损失对所有像素一视同仁
- 模型只要把白斑变黑就能获得很低的 loss
- 细线条的恢复被忽略

## 🆕 动态权重调整

**新功能**：损失函数权重可以随训练进度自动调整！

- **训练早期**：低边缘权重，专注整体去噪
- **训练后期**：高边缘权重，强化细节恢复

详细说明请查看 [DYNAMIC_LOSS_WEIGHTS.md](DYNAMIC_LOSS_WEIGHTS.md)

## 解决方案

### 1. WeightedMSELoss（加权 MSE）⭐ 推荐入门

**原理**：对真值中的亮像素（线条）赋予更高权重

```yaml
training:
  loss:
    type: "weighted_mse"
    params:
      foreground_weight: 10.0    # 线条区域权重是背景的 10 倍
      threshold: 0.5             # 亮度阈值（0-1）
```

**优点**：
- 简单直接，易于理解
- 强制模型关注线条区域
- 计算开销小

**适用**：快速验证想法，轻度不平衡

---

### 2. EdgePreservingLoss（边缘保持）

**原理**：MSE + 边缘损失，保持线条清晰

```yaml
training:
  loss:
    type: "edge_preserving"
    params:
      edge_weight: 0.5           # 边缘损失权重
      mse_weight: 1.0            # MSE 权重
```

**优点**：
- 明确优化边缘清晰度
- 使用 Sobel 算子检测边缘
- 适合细节恢复

**适用**：线条模糊、需要锐化

---

### 3. FocalMSELoss（Focal 损失）

**原理**：对误差大的像素（难样本）赋予更高权重

```yaml
training:
  loss:
    type: "focal_mse"
    params:
      gamma: 2.0                 # 聚焦参数，越大越关注难样本
```

**优点**：
- 自动识别难恢复的区域
- 借鉴目标检测的 Focal Loss
- 适应性强

**适用**：模型容易忽略的细节

---

### 4. CombinedLoss（组合损失）⭐⭐⭐ 最强推荐

**原理**：加权 MSE + 边缘保持 + Focal，三管齐下

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 15.0    # 线条权重
      edge_weight: 0.5           # 边缘权重
      focal_gamma: 2.0           # Focal 参数
      use_edge: true             # 启用边缘损失
      use_focal: true            # 启用 Focal 损失
```

**优点**：
- 结合三种策略的优势
- 最强的线条恢复能力
- 可灵活开关各个组件

**适用**：严重不平衡，需要最佳效果

---

### 5. PerceptualWeightedLoss（感知加权）⭐ 自适应

**原理**：自动识别每张图的重要区域（最亮的 10%）

```yaml
training:
  loss:
    type: "perceptual_weighted"
    params:
      percentile: 90.0           # 前 10% 最亮的像素
      min_weight: 1.0            # 背景权重
      max_weight: 20.0           # 线条权重
```

**优点**：
- 自适应，不需要手动设置阈值
- 根据每张图的亮度分布调整
- 适合亮度变化大的数据集

**适用**：数据集亮度不一致

---

## 推荐配置

### 方案 A：快速验证（加权 MSE）

```yaml
training:
  loss:
    type: "weighted_mse"
    params:
      foreground_weight: 10.0
```

**训练时间**：正常
**效果**：中等，快速看到改善

---

### 方案 B：最佳效果（组合损失）⭐ 推荐

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 15.0
      edge_weight: 0.5
      focal_gamma: 2.0
      use_edge: true
      use_focal: true
```

**训练时间**：稍慢（+10-20%）
**效果**：最佳，线条清晰

---

### 方案 C：自适应（感知加权）

```yaml
training:
  loss:
    type: "perceptual_weighted"
    params:
      percentile: 90.0
      max_weight: 20.0
```

**训练时间**：正常
**效果**：自适应，稳定

---

## 参数调优建议

### foreground_weight（前景权重）

- **默认**：10.0
- **线条很细**：15.0 - 20.0
- **线条较粗**：5.0 - 10.0
- **效果**：越大越关注线条，但过大可能过拟合

### edge_weight（边缘权重）

- **默认**：0.5
- **线条模糊**：1.0 - 2.0
- **线条清晰**：0.3 - 0.5
- **效果**：越大边缘越锐利，但过大可能产生伪影

### focal_gamma（Focal 参数）

- **默认**：2.0
- **难样本多**：3.0 - 5.0
- **难样本少**：1.0 - 2.0
- **效果**：越大越关注难样本

---

## 使用方法

### 1. 修改配置文件

编辑 `configs/config_multi_dataset.yaml`：

```yaml
training:
  loss:
    type: "combined"  # 选择损失函数类型
    params:
      foreground_weight: 15.0
      # ... 其他参数
```

### 2. 开始训练

```bash
python main.py train --config configs/config_multi_dataset.yaml
```

### 3. 监控效果

```bash
tensorboard --logdir=./experiments
```

在 TensorBoard 的 IMAGES 标签页查看去噪效果，观察线条是否清晰。

---

## 对比实验

建议运行多个实验对比效果：

```bash
# 实验 1：标准 MSE（基线）
# 修改 config，设置 loss.type: "mse"
python main.py train --config configs/config_baseline.yaml

# 实验 2：加权 MSE
# 修改 config，设置 loss.type: "weighted_mse"
python main.py train --config configs/config_weighted.yaml

# 实验 3：组合损失（最强）
# 修改 config，设置 loss.type: "combined"
python main.py train --config configs/config_combined.yaml

# 对比结果
python main.py compare --experiments <exp_id_1> <exp_id_2> <exp_id_3>
```

---

## 预期效果

使用组合损失后，你应该看到：

✅ 白斑噪声被有效去除
✅ 细线条清晰可见
✅ 线条边缘锐利
✅ PSNR/SSIM 指标提升

如果效果不理想，尝试：
1. 增加 `foreground_weight`（15 → 20）
2. 增加 `edge_weight`（0.5 → 1.0）
3. 降低学习率（0.001 → 0.0005）
4. 增加训练轮数（100 → 150）

---

## 技术原理

### 为什么加权有效？

标准 MSE：
```
Loss = mean((pred - target)²)
```
所有像素权重相同，背景占主导。

加权 MSE：
```
Loss = mean(weight_map * (pred - target)²)
weight_map = {15.0 if 线条, 1.0 if 背景}
```
线条区域的误差被放大 15 倍，模型被迫关注。

### 为什么需要边缘损失？

MSE 只关注像素值，不关注结构。
边缘损失明确优化边缘清晰度：
```
Edge_Loss = MSE(Sobel(pred), Sobel(target))
```

### 为什么需要 Focal？

Focal 自动识别难样本（误差大的像素）：
```
Focal_Weight = (1 - exp(-error))^gamma
```
误差越大，权重越高，强制模型学习难点。

---

## 常见问题

**Q: 训练变慢了？**
A: 组合损失计算量稍大（+10-20%），但效果提升明显，值得。

**Q: 线条还是不清晰？**
A: 尝试增加 `foreground_weight` 到 20-30，或增加 `edge_weight` 到 1.0-2.0。

**Q: 出现伪影？**
A: 权重过大，降低 `foreground_weight` 或 `edge_weight`。

**Q: 如何选择损失函数？**
A: 先用 `weighted_mse` 快速验证，效果好就用 `combined` 追求最佳。

**Q: 可以自己调整权重吗？**
A: 当然！在配置文件中修改参数，多次实验找到最佳值。


---

## 🆕 动态权重调整示例

### 示例 1：渐进式训练（推荐）

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 10.0
      edge_weight: 0.5                    # 最大边缘权重
      focal_gamma: 2.0
      focal_weight: 0.2
      use_edge: true
      use_focal: true
      edge_weight_schedule: "cosine"      # 🆕 余弦增长
      foreground_weight_schedule: null    # 固定前景权重
```

**效果**：
- Epoch 1-50: 边缘权重从 0 缓慢增长，模型专注整体去噪
- Epoch 50-100: 边缘权重快速增长到 0.5，强化细节恢复

### 示例 2：阶段式训练

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 12.0
      edge_weight: 0.6
      focal_gamma: 2.0
      focal_weight: 0.2
      use_edge: true
      use_focal: true
      edge_weight_schedule: "step"        # 🆕 阶梯增长
      foreground_weight_schedule: "linear" # 🆕 线性增长
```

**效果**：
- Epoch 1-50: 边缘权重 = 0，前景权重从 1.0 增长到 6.5
- Epoch 51-75: 边缘权重 = 0.3，前景权重继续增长
- Epoch 76-100: 边缘权重 = 0.6，前景权重达到 12.0

### 查看权重变化

训练时在 TensorBoard 中查看：
```bash
tensorboard --logdir=./experiments
```

找到 `Loss_Weights` 组，可以看到：
- `edge_weight`: 边缘权重随 epoch 的变化曲线
- `foreground_weight`: 前景权重随 epoch 的变化曲线

详细文档：[DYNAMIC_LOSS_WEIGHTS.md](DYNAMIC_LOSS_WEIGHTS.md)
