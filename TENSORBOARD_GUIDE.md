# TensorBoard 使用指南

## 启动方式

### 方法1：监控所有实验（推荐）
```bash
tensorboard --logdir=./experiments
```

### 方法2：监控特定实验
```bash
tensorboard --logdir=./experiments/ffdnet_multi_dataset_20260310_133517_43a0ac8d/checkpoints/tensorboard
```

然后在浏览器打开：`http://localhost:6006`

## 数据更新频率

### SCALARS（标量数据）

1. **Loss/train_step** - 