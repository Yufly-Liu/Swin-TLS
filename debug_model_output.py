"""诊断模型输出问题"""

import torch
from models import get_model

# 创建模型
model_config = {
    "in_channels": 3,
    "out_channels": 3,
    "num_layers": 15,
    "num_features": 64
}

from models.rednet import REDNet
model = REDNet(model_config)
model.eval()

# 创建测试输入
batch_size = 2
x = torch.rand(batch_size, 3, 512, 512)

print("=" * 60)
print("模型输出诊断")
print("=" * 60)

print(f"\n输入统计:")
print(f"  形状: {x.shape}")
print(f"  范围: [{x.min():.4f}, {x.max():.4f}]")
print(f"  均值: {x.mean():.4f}")
print(f"  标准差: {x.std():.4f}")

# 前向传播
with torch.no_grad():
    output = model(x)

print(f"\n输出统计:")
print(f"  形状: {output.shape}")
print(f"  范围: [{output.min():.4f}, {output.max():.4f}]")
print(f"  均值: {output.mean():.4f}")
print(f"  标准差: {output.std():.4f}")

# 检查是否有异常
if output.std() < 0.01:
    print("\n⚠️  警告: 输出标准差太小，模型可能输出常数！")
    
if torch.isnan(output).any():
    print("\n⚠️  警告: 输出包含 NaN！")
    
if torch.isinf(output).any():
    print("\n⚠️  警告: 输出包含 Inf！")

if (output < 0).any():
    print(f"\n⚠️  警告: 输出包含负值（{(output < 0).sum()} 个像素）")
    
if (output > 1).any():
    print(f"\n⚠️  警告: 输出超过 1.0（{(output > 1).sum()} 个像素）")

# 计算 PSNR
mse = ((output - x) ** 2).mean()
psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
print(f"\n模拟 PSNR: {psnr.item():.4f} dB")

if psnr < 15:
    print("⚠️  PSNR 太低，模型输出与输入差异很大！")

print("\n" + "=" * 60)
