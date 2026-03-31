# NaN 问题修复指南

## 问题现象

```
loss: nan, val_psnr: nan, val_ssim: 1.0000
```

## 原因分析

NaN（Not a Number）通常由以下原因导致：

### 1. 数值不稳定
- 除零：`x / 0`
- 负数开方：`sqrt(-1)`
- 零的对数：`log(0)`
- 无穷大运算：`inf × 0`

### 2. 梯度爆炸
- 学习率过大
- 权重初始化不当
- 损失函数数值范围过大

### 3. 数据问题
- 输入包含 NaN 或 Inf
- 数据归一化不当
- 数据集太小导致统计不稳定

### 4. 模型配置错误
- 模型参数与模型类型不匹配
- 损失函数参数设置不当

## 已修复的问题

### 1. 配置文件错误
**问题**：模型名是 `ffdnet`，但参数是 U-Net 的
```yaml
model:
  name: "ffdnet"  # ❌ 错误
  params:
    base_features: 64  # U-Net 参数
    depth: 4
```

**修复**：
```yaml
model:
  name: "unet"  # ✅ 正确
  params:
    base_features: 64
    depth: 4
```

### 2. 数据集太小
**问题**：只有 15 张图（每个数据集 5 张）
- 训练集：10 张
- 验证集：2 张
- 测试集：3 张

统计不稳定，容易产生 NaN。

**修复**：增加样本数
```yaml
datasets:
  - num_samples: 200  # aluminum
  - num_samples: 400  # iron
  - num_samples: 600  # synthesis
```

### 3. 损失函数数值稳定性
**问题**：Focal Loss 和 Edge Loss 可能产生 NaN

**修复**：添加数值稳定性保护

```python
# FocalMSELoss
def forward(self, pred, target):
    mse = (pred - target) ** 2
    mse_max = mse.max()
    
    # 避免除零
    if mse_max < 1e-8:
        return mse.mean()
    
    mse_normalized = mse / (mse_max + 1e-8)
    mse_normalized = torch.clamp(mse_normalized, 0, 1)
    
    focal_weight = (1 - torch.exp(-mse_normalized)) ** self.gamma
    focal_mse = mse * (1 + focal_weight)
    
    # NaN 检查
    result = focal_mse.mean()
    if torch.isnan(result) or torch.isinf(result):
        return mse.mean()  # 降级到标准 MSE
    
    return result

# CombinedLoss
def forward(self, pred, target):
    loss = self.weighted_mse(pred, target)
    
    # 每个组件都检查 NaN
    if torch.isnan(loss) or torch.isinf(loss):
        loss = F.mse_loss(pred, target)
    
    if self.use_edge:
        edge_loss_val = self.edge_loss(pred, target)
        if not (torch.isnan(edge_loss_val) or torch.isinf(edge_loss_val)):
            loss = loss + edge_loss_val
    
    # 最终检查
    if torch.isnan(loss) or torch.isinf(loss):
        return F.mse_loss(pred, target)
    
    return loss
```

## 测试步骤

### 步骤1：测试损失函数
```bash
python test_loss_functions.py
```

应该看到所有损失函数都正常工作，没有 NaN。

### 步骤2：使用测试配置训练
```bash
python main.py train --config configs/config_test_loss.yaml
```

这个配置使用：
- 少量数据（50 张）
- 小模型（DnCNN, 10 层）
- 小 patch（256×256）
- 降低的权重（foreground_weight=10）

如果这个能正常训练，说明损失函数没问题。

### 步骤3：使用完整配置训练
```bash
python main.py train --config configs/config_line_restoration.yaml
```

## 调试技巧

### 1. 检查数据
```python
# 在训练前添加
for noisy, clean in train_loader:
    print(f"Noisy: min={noisy.min()}, max={noisy.max()}, has_nan={torch.isnan(noisy).any()}")
    print(f"Clean: min={clean.min()}, max={clean.max()}, has_nan={torch.isnan(clean).any()}")
    break
```

### 2. 检查损失
```python
# 在 trainer.py 的 train_epoch 中添加
loss = self.loss_fn(output, clean)
if torch.isnan(loss):
    print(f"NaN detected! output: {output.min()}-{output.max()}, clean: {clean.min()}-{clean.max()}")
    import pdb; pdb.set_trace()
```

### 3. 降低学习率
```yaml
training:
  learning_rate: 0.0001  # 从 0.0005 降低到 0.0001
```

### 4. 使用梯度裁剪
在 `trainer.py` 的 `train_epoch` 中添加：
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.optimizer.step()
```

### 5. 降低损失函数权重
```yaml
loss:
  params:
    foreground_weight: 5.0  # 从 15 降低到 5
    edge_weight: 0.2        # 从 0.5 降低到 0.2
    focal_gamma: 1.0        # 从 2.0 降低到 1.0
```

## 推荐配置

### 保守配置（稳定优先）
```yaml
training:
  learning_rate: 0.0001
  loss:
    type: "weighted_mse"  # 最简单的加权 MSE
    params:
      foreground_weight: 5.0
```

### 平衡配置（效果与稳定兼顾）
```yaml
training:
  learning_rate: 0.0005
  loss:
    type: "combined"
    params:
      foreground_weight: 10.0
      edge_weight: 0.3
      focal_gamma: 1.5
      use_edge: true
      use_focal: true
```

### 激进配置（最佳效果）
```yaml
training:
  learning_rate: 0.001
  loss:
    type: "combined"
    params:
      foreground_weight: 15.0
      edge_weight: 0.5
      focal_gamma: 2.0
      use_edge: true
      use_focal: true
```

## 常见问题

**Q: 为什么会出现 NaN？**
A: 通常是数值不稳定或梯度爆炸。已添加保护机制。

**Q: 如何确认修复成功？**
A: 运行 `python test_loss_functions.py`，所有测试应该通过。

**Q: 训练还是出现 NaN 怎么办？**
A: 
1. 降低学习率
2. 使用更简单的损失函数（weighted_mse）
3. 减小 foreground_weight
4. 添加梯度裁剪

**Q: 数据集太小会有问题吗？**
A: 会。建议至少 100+ 张图，否则统计不稳定。

## 验证清单

- [ ] 运行 `python test_loss_functions.py` 通过
- [ ] 配置文件中模型名和参数匹配
- [ ] 数据集大小 >= 100 张
- [ ] 学习率 <= 0.001
- [ ] foreground_weight <= 15
- [ ] 训练前几个 epoch 没有 NaN

## 下一步

修复完成后，重新训练：

```bash
# 测试配置（快速验证）
python main.py train --config configs/config_test_loss.yaml

# 完整配置（正式训练）
python main.py train --config configs/config_line_restoration.yaml

# 监控
tensorboard --logdir=./experiments
```

在 TensorBoard 中查看：
- Loss 曲线应该平滑下降
- PSNR/SSIM 应该逐渐上升
- 图像可视化应该显示线条恢复效果
