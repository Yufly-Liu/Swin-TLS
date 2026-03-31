# 图像去噪网络对比系统

模块化的深度学习图像去噪对比平台，支持训练、评估和对比多个经典去噪网络。针对细线条恢复场景（黑色背景 + 白色细线条）做了专门优化。

## 特性

- 6 种去噪模型：DnCNN, U-Net, RED-Net, FFDNet, Restormer, SUNet
- 多数据集合并训练，支持过采样/欠采样
- Patch 训练 + 数据增强（随机裁剪、翻转、旋转）
- 组合损失函数（模型原始损失 + WeightedMSE + 边缘保持 + Focal），阶段式权重调整
- TensorBoard 实时监控（损失曲线、各子损失分项、PSNR/SSIM、去噪效果图）
- 检查点保存/恢复，支持断点续训

## 支持的模型

| 模型 | 架构 | 默认损失 | 参考论文 |
|------|------|---------|---------|
| DnCNN | 17层CNN，残差学习 | MSE | Zhang et al., 2017 |
| U-Net | 编码器-解码器，跳跃连接 | MSE | Ronneberger et al., 2015 |
| RED-Net | 对称卷积-反卷积，跳跃连接 | MSE | Mao et al., 2016 |
| FFDNet | 噪声级别自适应 | L1 | Zhang et al., 2018 |
| Restormer | Transformer，多头转置自注意力 | Charbonnier | Zamir et al., 2022 |
| SUNet | Swin Transformer U-Net | L1 | 官方实现 |

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd mark2

# 创建 conda 环境
conda create -n IRNet python=3.10
conda activate IRNet

# 安装 PyTorch（CUDA 11.8，适配 Quadro RTX 5000）
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

主要依赖：PyTorch 2.7.1、TensorBoard 2.20、scikit-image 0.25、matplotlib 3.10、PyYAML 6.0、tqdm 4.67、hypothesis 6.151（测试）。完整列表见 `requirements.txt`。

## 数据集结构

```
data/
├── aluminum/          # 铝材数据集（240张，1376×2048）
│   ├── input/         # 噪声图像
│   └── target/        # 干净图像
├── iron/              # 铁材数据集（520张）
│   ├── input/
│   └── target/
└── Synthesis/         # 合成数据集（911张）
    ├── input/
    └── target/
```

支持格式：PNG、JPG、BMP、TIFF（大小写不敏感）。

## 配置文件

项目提供两个通用配置文件，通过修改 `model.name` 切换模型：

| 配置文件 | 说明 |
|---------|------|
| `configs/config_combined_loss.yaml` | 组合损失（推荐），包含模型原始损失 + WeightedMSE + 边缘 + Focal |
| `configs/config_default_loss.yaml` | 仅使用模型默认损失函数 |
| `configs/config_test_loss.yaml` | 小样本快速测试用 |

切换模型只需改一行：

```yaml
model:
  name: "dncnn"  # 可选: dncnn, ffdnet, rednet, restormer, sunet, unet
```

## 使用方法

### 训练

```bash
# 使用组合损失训练（推荐）
python main.py train --config configs/config_combined_loss.yaml

# 使用模型默认损失训练
python main.py train --config configs/config_default_loss.yaml

# 从检查点恢复训练（会从断点 epoch 继续）
python main.py train --config configs/config_combined_loss.yaml --resume experiments/<exp_id>/checkpoints/checkpoint_epoch_25.pth
```

## 数据增强

训练时自动应用以下增强（在 `data/transforms.py` 中实现）：
- 随机裁剪（`patch_size`，默认 512×512）
- 随机水平/垂直翻转
- 随机 90° 旋转
- 每张图每个 epoch 生成 `patches_per_image` 个不同的随机 patch

验证/测试时仅做 resize，不做随机增强。

## 学习率策略

默认使用余弦退火（CosineAnnealingLR），学习率从初始值平滑衰减到 `eta_min`：

```yaml
scheduler:
  type: "CosineAnnealingLR"
  T_max: 150        # 与 num_epochs 一致
  eta_min: 0.000001
```

也支持 ReduceLROnPlateau 和 StepLR。

## 项目结构

```
├── main.py                    # 主程序入口（train/evaluate/compare）
├── configs/                   # 配置文件
├── data/
│   ├── dataset.py             # 数据集加载（多数据集合并、采样）
│   ├── transforms.py          # 数据增强（随机裁剪、翻转、旋转）
│   └── __init__