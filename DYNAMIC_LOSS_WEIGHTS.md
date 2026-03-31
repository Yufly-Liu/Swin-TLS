# 动态损失权重调整指南

## 概述

动态损失权重调整允许在训练过程中自动调整损失函数的权重系数，使模型在不同训练阶段关注不同的优化目标。

## 为什么需要动态权重？

### 训练早期 vs 训练后期

**训练早期（前 30-50%）：**
- 模型参数随机初始化，输出噪声很大
- 应该先学习整体结构和大致的去噪效果
- 过早关注细节（边缘）会导致训练不稳定
- 建议：低边缘权重或零边缘权重

**训练后期（后 50-70%）：**
- 模型已经学会基本的去噪
- 可以开始关注细节恢复（边缘、线条）
- 增加边缘权重强制模型保持细节清晰
- 建议：逐步增加边缘权重到最大值

### 细线条恢复的特殊需求

对于黑色背景 + 白色细线条的任务：
- 早期：模型学习去除大片白斑（主要任务）
- 后期：模型学习恢复细线条（次要但关键）
- 动态调整前景权重可以让模型逐步强化对线条的关注

## 支持的调度策略

### 1. 边缘权重调度 (`edge_weight_schedule`)

#### `null` (默认)
固定权重，不调整。

```yaml
edge_weight: 0.3
edge_weight_schedule: null
```

#### `linear`
从 0 线性增长到 `edge_weight`。

```yaml
edge_weight: 0.5
edge_weight_schedule: "linear"
```

- Epoch 1: 边缘权重 = 0.0
- Epoch 50 (50%): 边缘权重 = 0.25
- Epoch 100 (100%): 边缘权重 = 0.5

**适用场景：** 稳定的训练，适合大多数情况。

#### `cosine`
余弦增长（前期慢，后期快）。

```yaml
edge_weight: 0.5
edge_weight_schedule: "cosine"
```

- Epoch 1: 边缘权重 ≈ 0.0
- Epoch 50 (50%): 边缘权重 = 0.25
- Epoch 75 (75%): 边缘权重 ≈ 0.43
- Epoch 100 (100%): 边缘权重 = 0.5

**适用场景：** 希望在训练后期快速提升细节质量。

#### `step`
阶梯增长（分段固定）。

```yaml
edge_weight: 0.6
edge_weight_schedule: "step"
```

- Epoch 1-50 (0-50%): 边缘权重 = 0.0
- Epoch 51-75 (50-75%): 边缘权重 = 0.3
- Epoch 76-100 (75-100%): 边缘权重 = 0.6

**适用场景：** 明确的训练阶段划分，适合实验对比。

### 2. 前景权重调度 (`foreground_weight_schedule`)

#### `null` (默认)
固定权重，不调整。

```yaml
foreground_weight: 10.0
foreground_weight_schedule: null
```

#### `linear`
从 1.0 线性增长到 `foreground_weight`。

```yaml
foreground_weight: 15.0
foreground_weight_schedule: "linear"
```

- Epoch 1: 前景权重 = 1.0（等同于标准 MSE）
- Epoch 50 (50%): 前景权重 = 8.0
- Epoch 100 (100%): 前景权重 = 15.0

**适用场景：** 逐步强化对线条的关注。

#### `cosine`
余弦增长（前期慢，后期快）。

```yaml
foreground_weight: 15.0
foreground_weight_schedule: "cosine"
```

**适用场景：** 在训练后期快速提升线条恢复质量。

## 配置示例

### 示例 1：保守策略（推荐新手）

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 10.0
      edge_weight: 0.3
      focal_gamma: 2.0
      focal_weight: 0.2
      use_edge: true
      use_focal: true
      edge_weight_schedule: "linear"  # 线性增长，稳定
      foreground_weight_schedule: null  # 固定前景权重
```

### 示例 2：激进策略（适合经验用户）

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 15.0
      edge_weight: 0.6
      focal_gamma: 2.0
      focal_weight: 0.3
      use_edge: true
      use_focal: true
      edge_weight_schedule: "cosine"  # 后期快速增长
      foreground_weight_schedule: "cosine"  # 后期强化线条
```

### 示例 3：阶段式训练

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 12.0
      edge_weight: 0.5
      focal_gamma: 2.0
      focal_weight: 0.2
      use_edge: true
      use_focal: true
      edge_weight_schedule: "step"  # 分阶段增长
      foreground_weight_schedule: "linear"  # 平滑增长
```

## TensorBoard 可视化

训练时会自动记录权重变化曲线：
- `Loss_Weights/edge_weight`: 边缘权重随 epoch 的变化
- `Loss_Weights/foreground_weight`: 前景权重随 epoch 的变化

查看方法：
```bash
tensorboard --logdir=./experiments
```

在 TensorBoard 中查看 "SCALARS" 标签页，找到 "Loss_Weights" 组。

## 调优建议

### 1. 从固定权重开始

首次训练建议使用固定权重（`schedule: null`），观察训练曲线：
- 如果训练稳定，loss 平滑下降 → 可以尝试动态权重
- 如果训练不稳定，loss 震荡 → 降低权重值，保持固定

### 2. 选择合适的调度策略

- **训练稳定但细节不足** → 使用 `linear` 或 `cosine`
- **训练后期 loss 下降缓慢** → 使用 `cosine`（后期加速）
- **需要明确的训练阶段** → 使用 `step`

### 3. 权重最大值设置

边缘权重建议范围：
- 保守：0.3 - 0.5
- 中等：0.5 - 0.8
- 激进：0.8 - 1.0

前景权重建议范围：
- 保守：5.0 - 10.0
- 中等：10.0 - 15.0
- 激进：15.0 - 20.0

### 4. 观察指标

在 TensorBoard 中同时观察：
- `Loss/train`: 训练损失（应该平滑下降）
- `Metrics/PSNR`: 峰值信噪比（应该上升）
- `Metrics/SSIM`: 结构相似性（应该上升）
- `Loss_Weights/*`: 权重变化曲线

如果 PSNR/SSIM 在权重增加后反而下降，说明权重过大或增长过快。

## 实现原理

### 训练进度计算

```python
progress = (current_epoch - 1) / (total_epochs - 1)  # 范围 [0, 1]
```

### 权重计算公式

**Linear:**
```python
weight = max_weight * progress
```

**Cosine:**
```python
import math
weight = max_weight * (1 - math.cos(progress * math.pi)) / 2
```

**Step:**
```python
if progress < 0.5:
    weight = 0.0
elif progress < 0.75:
    weight = max_weight * 0.5
else:
    weight = max_weight
```

## 常见问题

### Q: 动态权重会影响训练速度吗？
A: 不会。权重计算只在每个 epoch 开始时执行一次，开销可忽略。

### Q: 可以中途改变调度策略吗？
A: 不建议。如果需要改变，建议从检查点重新开始训练。

### Q: 如何判断是否需要动态权重？
A: 如果固定权重训练后期 loss 下降缓慢，或者细节恢复不理想，可以尝试动态权重。

### Q: 动态权重适合所有任务吗？
A: 不一定。对于简单的去噪任务，固定权重可能就足够了。动态权重更适合需要平衡整体质量和细节的复杂任务。

## 总结

动态损失权重调整是一个强大的工具，可以让模型在训练过程中自动调整优化目标。建议：

1. 新手从固定权重开始
2. 有经验后尝试 `linear` 调度
3. 需要更好的细节时使用 `cosine` 调度
4. 始终通过 TensorBoard 观察训练曲线
5. 根据实际效果调整权重最大值

记住：没有万能的配置，需要根据具体数据集和任务调优。
