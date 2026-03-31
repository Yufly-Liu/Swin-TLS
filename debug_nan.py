"""调试 NaN 问题 - 用真实数据测试损失函数"""
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from models.losses import WeightedMSELoss, EdgePreservingLoss, FocalMSELoss, CombinedLoss

# 1. 加载一张真实图像
noisy_path = "data/aluminum/input/000000.BMP"
clean_path = "data/aluminum/target/000000.BMP"

noisy = TF.to_tensor(Image.open(noisy_path).convert('RGB')).unsqueeze(0)  # (1,3,H,W)
clean = TF.to_tensor(Image.open(clean_path).convert('RGB')).unsqueeze(0)

print(f"Noisy: shape={noisy.shape}, min={noisy.min():.4f}, max={noisy.max():.4f}, has_nan={torch.isnan(noisy).any()}")
print(f"Clean: shape={clean.shape}, min={clean.min():.4f}, max={clean.max():.4f}, has_nan={torch.isnan(clean).any()}")

# 裁剪到 256x256
noisy = noisy[:, :, :256, :256]
clean = clean[:, :, :256, :256]

# 2. 模拟模型输出（随机初始化的模型输出）
pred = torch.randn_like(clean) * 0.5 + 0.5
pred = pred.clamp(0, 1)
print(f"Pred:  shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}")

# 3. 逐个测试损失函数
print("\n--- 测试各损失函数 ---")

# MSE
mse = torch.nn.MSELoss()(pred, clean)
print(f"MSE:          {mse.item():.6f}, nan={torch.isnan(mse).any()}")

# WeightedMSE
wmse = WeightedMSELoss(foreground_weight=10.0, threshold=0.5)(pred, clean)
print(f"WeightedMSE:  {wmse.item():.6f}, nan={torch.isnan(wmse).any()}")

# EdgePreserving
edge = EdgePreservingLoss(edge_weight=0.3)(pred, clean)
print(f"EdgeLoss:     {edge.item():.6f}, nan={torch.isnan(edge).any()}")

# FocalMSE
focal = FocalMSELoss(gamma=1.5)(pred, clean)
print(f"FocalMSE:     {focal.item():.6f}, nan={torch.isnan(focal).any()}")

# Combined
combined = CombinedLoss(foreground_weight=10.0, edge_weight=0.3, focal_gamma=1.5)(pred, clean)
print(f"Combined:     {combined.item():.6f}, nan={torch.isnan(combined).any()}")

# 4. 测试梯度
print("\n--- 测试梯度 ---")
pred_grad = pred.clone().requires_grad_(True)
loss = CombinedLoss(foreground_weight=10.0, edge_weight=0.3, focal_gamma=1.5)(pred_grad, clean)
print(f"Loss: {loss.item():.6f}")
loss.backward()
print(f"Grad: min={pred_grad.grad.min():.6f}, max={pred_grad.grad.max():.6f}, has_nan={torch.isnan(pred_grad.grad).any()}")

# 5. 测试 GPU
if torch.cuda.is_available():
    print("\n--- GPU 测试 ---")
    pred_gpu = pred.cuda().requires_grad_(True)
    clean_gpu = clean.cuda()
    loss_fn = CombinedLoss(foreground_weight=10.0, edge_weight=0.3, focal_gamma=1.5).cuda()
    loss_gpu = loss_fn(pred_gpu, clean_gpu)
    print(f"GPU Loss: {loss_gpu.item():.6f}, nan={torch.isnan(loss_gpu).any()}")
    loss_gpu.backward()
    print(f"GPU Grad: has_nan={torch.isnan(pred_gpu.grad).any()}")

# 6. 模拟真实训练：DnCNN + CombinedLoss
print("\n--- 模拟真实训练 ---")
from models import get_model
model = get_model("dncnn", {"in_channels": 3, "out_channels": 3, "num_layers": 10, "num_features": 32})
if torch.cuda.is_available():
    model = model.cuda()
    noisy_gpu = noisy.cuda()
    clean_gpu = clean.cuda()
    loss_fn = CombinedLoss(foreground_weight=10.0, edge_weight=0.3, focal_gamma=1.5).cuda()
else:
    noisy_gpu = noisy
    clean_gpu = clean
    loss_fn = CombinedLoss(foreground_weight=10.0, edge_weight=0.3, focal_gamma=1.5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(5):
    optimizer.zero_grad()
    output = model(noisy_gpu)
    print(f"Step {step}: output min={output.min():.4f}, max={output.max():.4f}, has_nan={torch.isnan(output).any()}")
    loss = loss_fn(output, clean_gpu)
    print(f"Step {step}: loss={loss.item():.6f}, nan={torch.isnan(loss).any()}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Step {step}: grad_norm={grad_norm:.6f}")
    optimizer.step()

print("\n完成！")
