"""调试验证集数据"""
import torch
from torch.utils.data import DataLoader
from utils.config import ConfigManager
from data.dataset import create_multi_dataset, split_dataset
from data.transforms import DataTransforms
import sys
sys.path.insert(0, '.')

# 重现 main.py 的逻辑
config = ConfigManager.load_config("configs/config_test_loss.yaml")
dataset_cfg = config["dataset"]

dataset = create_multi_dataset(dataset_cfg["datasets"])
train_set, val_set, test_set = split_dataset(dataset, 0.7, 0.15, 0.15)

val_transform = DataTransforms.get_val_transforms(target_size=256)

# 模拟 _TransformSubset
class _TransformSubset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        noisy, clean = self.subset[idx]
        if self.transform is not None:
            from PIL import Image as PILImage
            if isinstance(noisy, torch.Tensor):
                return noisy, clean  # 已经是 tensor，跳过 transform！
            return self.transform(noisy, clean)
        return noisy, clean
    def __len__(self):
        return len(self.subset)

val_ds = _TransformSubset(val_set, val_transform)

print(f"验证集大小: {len(val_ds)}")
for i in range(len(val_ds)):
    noisy, clean = val_ds[i]
    print(f"[{i}] noisy: {noisy.shape}, clean: {clean.shape}")
    print(f"     noisy type: {type(noisy)}, is_tensor: {isinstance(noisy, torch.Tensor)}")
