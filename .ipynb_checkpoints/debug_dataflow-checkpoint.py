"""调试数据流 - 检查 MultiPatchDataset 输出"""
import torch
from torch.utils.data import DataLoader
from utils.config import ConfigManager
from data.dataset import DenoisingDataset, create_multi_dataset, split_dataset, MultiPatchDataset
from data.transforms import DataTransforms

# 加载配置
config = ConfigManager.load_config("configs/config_test_loss.yaml")
dataset_cfg = config["dataset"]

# 创建数据集
dataset = create_multi_dataset(dataset_cfg["datasets"])
print(f"总数据集大小: {len(dataset)}")

# 划分
train_set, val_set, test_set = split_dataset(dataset, 0.7, 0.15, 0.15)
print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

# 创建 transform
train_transform = DataTransforms.get_train_transforms(
    target_size=256, use_random_crop=True, patches_per_image=2
)

# 创建 MultiPatchDataset
train_ds = MultiPatchDataset(train_set, 2, train_transform)
print(f"MultiPatchDataset 大小: {len(train_ds)}")

# 测试取数据
print("\n--- 测试取数据 ---")
for i in range(min(5, len(train_ds))):
    try:
        noisy, clean = train_ds[i]
        print(f"[{i}] noisy: {noisy.shape}, min={noisy.min():.4f}, max={noisy.max():.4f}, nan={torch.isnan(noisy).any()}")
        print(f"[{i}] clean: {clean.shape}, min={clean.min():.4f}, max={clean.max():.4f}, nan={torch.isnan(clean).any()}")
    except Exception as e:
        print(f"[{i}] ERROR: {e}")

# 测试 DataLoader
print("\n--- 测试 DataLoader ---")
loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
for batch_idx, (noisy, clean) in enumerate(loader):
    print(f"Batch {batch_idx}: noisy={noisy.shape}, clean={clean.shape}")
    print(f"  noisy: min={noisy.min():.4f}, max={noisy.max():.4f}, nan={torch.isnan(noisy).any()}")
    print(f"  clean: min={clean.min():.4f}, max={clean.max():.4f}, nan={torch.isnan(clean).any()}")
    if batch_idx >= 2:
        break

# 测试模型 + 损失
print("\n--- 测试模型 + 损失 ---")
from models import get_model
from models.losses import CombinedLoss

model = get_model("dncnn", {"in_channels": 3, "out_channels": 3, "num_layers": 10, "num_features": 32}).cuda()
loss_fn = CombinedLoss(foreground_weight=10.0, edge_weight=0.3, focal_gamma=1.5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for batch_idx, (noisy, clean) in enumerate(loader):
    noisy, clean = noisy.cuda(), clean.cuda()
    optimizer.zero_grad()
    output = model(noisy)
    loss = loss_fn(output, clean)
    print(f"Batch {batch_idx}: loss={loss.item():.6f}, nan={torch.isnan(loss).any()}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  grad_norm={grad_norm:.4f}")
    optimizer.step()
    if batch_idx >= 3:
        break

print("\n完成！")
