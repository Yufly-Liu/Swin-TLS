# 配置文件使用指南

## 配置文件列表

项目提供 3 个配置文件：

1. **config_combined_loss.yaml** - 组合损失（推荐用于细线条恢复）
2. **config_default_loss.yaml** - 默认损失（使用模型原本的损失函数）
3. **config_test_loss.yaml** - 快速测试（小数据集，10 epoch）

## 快速开始

### 1. 使用组合损失训练 DnCNN

```bash
python main.py train --config configs/config_combined_loss.yaml
```

默认使用 DnCNN 模型。

### 2. 切换到其他模型

编辑 `configs/config_combined_loss.yaml`，修改 `model.name`：

```yaml
model:
  name: "unet"  # 改成 unet, ffdnet, rednet, restormer, sunet 等
```

然后运行：

```bash
python main.py train --config configs/config_combined_loss.yaml
```

### 3. 使用默认损失对比

```bash
python main.py train --config configs/config_default_loss.yaml
```

## 支持的模型

| 模型名 | 说明 | 默认损失 | 参数量 |
|--------|------|----------|--------|
| `dncnn` | 深度卷积去噪网络 | MSE | 小 |
| `ffdnet` | 灵活快速去噪网络 | L1 | 小 |
| `rednet` | 对称编码-解码网络 | MSE | 中 |
| `unet` | U-Net 编码-解码 | MSE | 中 |
| `restormer` | Transformer 恢复网络 | Charbonnier | 大 |
| `sunet` | Swin Transformer U-Net | L1 | 大 |

## 配置说明

### 数据集配置

```yaml
dataset:
  datasets:
    - noisy_dir: "./data/aluminum/input"
      clean_dir: "./data/aluminum/target"
      num_samples: 200          # 使用多少张图
      name: "aluminum"
```

- `num_samples`: 指定使用的图像数量
  - 如果 > 实际数量：会重复采样
  - 如果 < 实际数量：会随机采样（不重复）
  - 设为 -1：使用全部数据

### 模型配置

```yaml
model:
  name: "dncnn"  # 修改这里切换模型
  params:
    in_channels: 3
    out_channels: 3
    # 其他参数会根据模型自动选择
```

不同模型使用不同的参数：
- DnCNN/FFDNet/RED-Net: `num_layers`, `num_features`
- U-Net: `base_features`, `depth`
- Restormer: `dim`, `num_blocks`, `num_heads`
- SUNet: `embed_dim`, `depths`, `window_size`, `patch_size`

### 训练配置

```yaml
training:
  num_epochs: 150
  learning_rate: 0.0002
  
  # 组合损失（仅 config_combined_loss.yaml）
  loss:
    type: "combined"
    params:
      foreground_weight: 10.0   # 线条区域权重
      edge_weight: 0.5          # 边缘损失权重
      focal_gamma: 2.0
      focal_weight: 0.2
      use_edge: true
      use_focal: true
      num_stages: 3             # 分 3 阶段调整权重
      stage_ratios: [0.4, 0.3, 0.3]
```

### 学习率调度

```yaml
scheduler:
  type: "ReduceLROnPlateau"
  patience: 8       # 8 个 epoch 没提升就降学习率
  factor: 0.3       # 每次降到 30%
  min_lr: 0.000001  # 最低学习率
```

可选调度器：
- `ReduceLROnPlateau`: 根据验证指标自动降低学习率
- `CosineAnnealingLR`: 余弦退火
- `StepLR`: 固定步长衰减

## 组合损失 vs 默认损失

### 组合损失（config_combined_loss.yaml）

**优点**：
- 针对细线条恢复优化
- 对线条区域赋予更高权重
- 保持边缘清晰
- 关注难恢复的区域
- 阶段式权重调整，训练更稳定

**适用场景**：
- 黑色背景 + 白色细线条
- 类别不平衡（线条占比小）
- 需要保持细节

**训练特点**：
- 阶段 1（前 40%）：边缘权重 = 0，专注整体去噪
- 阶段 2（中 30%）：边缘权重 = 0.25，开始关注边缘
- 阶段 3（后 30%）：边缘权重 = 0.5，全力恢复细节

### 默认损失（config_default_loss.yaml）

**优点**：
- 简单直接
- 训练稳定
- 适合一般去噪任务

**适用场景**：
- 均匀噪声
- 不需要特别关注某些区域
- 对比实验基线

## 训练流程

### 1. 准备数据

确保数据目录结构：
```
data/
├── aluminum/
│   ├── input/   # 噪声图像
│   └── target/  # 干净图像
├── iron/
│   ├── input/
│   └── target/
└── Synthesis/
    ├── input/
    └── target/
```

### 2. 选择配置文件

- 细线条恢复 → `config_combined_loss.yaml`
- 一般去噪 → `config_default_loss.yaml`
- 快速测试 → `config_test_loss.yaml`

### 3. 修改模型

编辑配置文件，修改 `model.name`。

### 4. 开始训练

```bash
python main.py train --config configs/config_combined_loss.yaml
```

### 5. 监控训练

```bash
tensorboard --logdir=./experiments
```

在浏览器打开 http://localhost:6006

### 6. 评估模型

```bash
python main.py evaluate --config configs/config_combined_loss.yaml \
    --checkpoint ./experiments/{实验ID}/checkpoints/best_model.pth
```

### 7. 对比实验

```bash
python main.py compare \
    --experiments ./experiments/{实验1} ./experiments/{实验2}
```

## 调优建议

### Loss 在 0.1 附近震荡

1. 降低学习率：`learning_rate: 0.0001`
2. 调整 scheduler：`patience: 5`, `factor: 0.3`
3. 增加 batch size（如果显存够）：`batch_size: 8`

### 训练太慢

1. 减少数据量：`num_samples: 100`
2. 减少 patch 数：`patches_per_image: 2`
3. 减小 patch 尺寸：`patch_size: 256`
4. 选择更小的模型：`dncnn` 或 `ffdnet`

### 细节恢复不好

1. 增加边缘权重：`edge_weight: 0.8`
2. 增加前景权重：`foreground_weight: 15.0`
3. 调整阶段占比：`stage_ratios: [0.3, 0.3, 0.4]`（后期更长）
4. 使用 U-Net 或 Restormer

### 显存不足

1. 减小 batch size：`batch_size: 2`
2. 减小 patch 尺寸：`patch_size: 256`
3. 选择更小的模型：避免 Restormer 和 SUNet
4. 减少 Transformer 参数：
   ```yaml
   dim: 32
   num_blocks: [2, 4, 4, 6]
   ```

## 实验命名规则

实验名称自动生成：`{model}_{experiment.name}_{timestamp}_{id}`

例如：
- `dncnn_combined_loss_20260310_143000_abc123`
- `unet_default_loss_20260310_150000_def456`

## TensorBoard 可视化

训练时自动记录：
- `Loss/train`: 训练损失
- `Loss/val`: 验证损失
- `Metrics/PSNR`: 峰值信噪比
- `Metrics/SSIM`: 结构相似性
- `Learning_Rate`: 学习率变化
- `Loss_Weights/edge_weight`: 边缘权重变化（仅组合损失）
- `Denoising/sample_*`: 去噪效果对比图

## 常见问题

### Q: 如何只训练一个数据集？

A: 删除配置文件中不需要的数据集：

```yaml
dataset:
  datasets:
    - noisy_dir: "./data/aluminum/input"
      clean_dir: "./data/aluminum/target"
      num_samples: -1  # 使用全部
      name: "aluminum"
```

### Q: 如何使用单张大图而不是 patch？

A: 修改配置：

```yaml
dataset:
  patch_size: 1376  # 改成图像实际尺寸
  use_random_crop: false
  patches_per_image: 1
```

### Q: 如何固定边缘权重不分阶段？

A: 设置 `num_stages: 1`：

```yaml
loss:
  params:
    num_stages: 1  # 全程固定权重
```

### Q: 如何从检查点继续训练？

A: 目前需要在代码中手动加载。后续版本会支持命令行参数。

## 总结

- 细线条恢复 → 用 `config_combined_loss.yaml`
- 一般去噪 → 用 `config_default_loss.yaml`
- 修改 `model.name` 切换模型
- 在 TensorBoard 中监控训练
- 根据 loss 曲线调整学习率和权重
