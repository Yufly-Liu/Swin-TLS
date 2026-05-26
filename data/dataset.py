"""数据集加载和管理模块"""

import os
from typing import Tuple, Optional, List, Callable, Dict

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, ConcatDataset


# 支持的图像格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


class DatasetError(Exception):
    """数据集相关错误"""
    pass


class DenoisingDataset(Dataset):
    """去噪数据集，加载噪声图像和干净图像对。

    Args:
        noisy_dir: 噪声图像目录路径
        clean_dir: 干净图像目录路径
        transform: 可选的数据变换函数，接收 (noisy, clean) PIL Image 对，
                   返回 (noisy_tensor, clean_tensor) 对
    """

    def __init__(self, noisy_dir: str, clean_dir: str, transform: Optional[Callable] = None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform

        # 验证目录存在
        if not os.path.isdir(noisy_dir):
            raise DatasetError(f"噪声图像目录不存在: {noisy_dir}")
        if not os.path.isdir(clean_dir):
            raise DatasetError(f"干净图像目录不存在: {clean_dir}")

        # 加载并验证图像对
        self.image_pairs = self._load_image_pairs()

        if len(self.image_pairs) == 0:
            raise DatasetError("数据集为空，未找到有效的图像对")

    def _load_image_pairs(self) -> List[Tuple[str, str]]:
        """扫描目录，匹配噪声和干净图像对。

        匹配规则：文件名相同（不区分大小写，含扩展名）的图像视为一对。
        只使用能配对上的图像，忽略无法配对的文件。

        Returns:
            匹配的 (noisy_path, clean_path) 列表

        Raises:
            DatasetError: 没有找到任何配对的图像
        """
        # 获取所有支持格式的文件
        noisy_files = {
            f for f in os.listdir(self.noisy_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        }
        clean_files = {
            f for f in os.listdir(self.clean_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        }

        # 创建小写文件名到实际文件名的映射（处理大小写不敏感）
        noisy_map = {f.lower(): f for f in noisy_files}
        clean_map = {f.lower(): f for f in clean_files}

        # 找到匹配的文件对
        matched_pairs = []
        
        for noisy_lower, noisy_actual in noisy_map.items():
            if noisy_lower in clean_map:
                noisy_path = os.path.join(self.noisy_dir, noisy_actual)
                clean_path = os.path.join(self.clean_dir, clean_map[noisy_lower])
                matched_pairs.append((noisy_path, clean_path))

        # 检查是否有配对的图像
        if len(matched_pairs) == 0:
            raise DatasetError(
                f"未找到任何配对的图像。\n"
                f"噪声目录: {self.noisy_dir} ({len(noisy_files)} 个文件)\n"
                f"干净目录: {self.clean_dir} ({len(clean_files)} 个文件)"
            )

        # 统计未配对的文件
        unmatched_noisy = len(noisy_files) - len(matched_pairs)
        unmatched_clean = len(clean_files) - len(matched_pairs)
        
        if unmatched_noisy > 0 or unmatched_clean > 0:
            print(f"警告: {self.noisy_dir} - 找到 {len(matched_pairs)} 对图像，"
                  f"忽略 {unmatched_noisy} 个未配对的噪声图像和 {unmatched_clean} 个未配对的干净图像")

        # 按文件名排序保证顺序一致
        matched_pairs.sort(key=lambda x: os.path.basename(x[0]).lower())
        return matched_pairs

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """获取第 idx 个图像对。

        Returns:
            (noisy_image, clean_image) 张量对，值范围 [0, 1]
        """
        noisy_path, clean_path = self.image_pairs[idx]

        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        if self.transform is not None:
            noisy_img, clean_img = self.transform(noisy_img, clean_img)
        else:
            # 默认转换为张量，归一化到 [0, 1]
            import torchvision.transforms.functional as TF
            noisy_img = TF.to_tensor(noisy_img)
            clean_img = TF.to_tensor(clean_img)

        return noisy_img, clean_img

    def __len__(self) -> int:
        return len(self.image_pairs)


def split_dataset(
    dataset: DenoisingDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """将数据集划分为训练集、验证集和测试集。

    Args:
        dataset: 要划分的数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子，保证可重现

    Returns:
        (train_subset, val_subset, test_subset)

    Raises:
        ValueError: 比例之和不为1或比例为负
    """
    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("划分比例不能为负数")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为1，当前为 {total}")

    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    # test gets the remainder to avoid rounding issues
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )



def create_multi_dataset(
    dataset_configs: List[Dict[str, any]],
    seed: int = 42,
) -> DenoisingDataset:
    """创建多个数据集的合并数据集，支持指定每个数据集的样本数。

    Args:
        dataset_configs: 数据集配置列表，每个配置包含:
            - noisy_dir: 噪声图像目录
            - clean_dir: 干净图像目录
            - num_samples: 要使用的样本数
                - 如果 > 数据集大小：重复采样（允许重复）
                - 如果 < 数据集大小：随机采样（不重复）
                - 如果未指定或为 -1：使用全部数据
            - name: 数据集名称（可选，用于日志）
        seed: 随机种子

    Returns:
        合并后的数据集

    Example:
        configs = [
            {"noisy_dir": "./data/aluminum/noisy", "clean_dir": "./data/aluminum/clean", "num_samples": 1000},
            {"noisy_dir": "./data/iron/noisy", "clean_dir": "./data/iron/clean", "num_samples": 500},
            {"noisy_dir": "./data/synthesis/noisy", "clean_dir": "./data/synthesis/clean", "num_samples": 2000},
        ]
        dataset = create_multi_dataset(configs)
    """
    all_datasets = []
    
    for cfg in dataset_configs:
        # 加载单个数据集
        ds = DenoisingDataset(
            noisy_dir=cfg["noisy_dir"],
            clean_dir=cfg["clean_dir"],
            transform=None  # transform 在外部统一应用
        )
        
        n_total = len(ds)
        num_samples = cfg.get("num_samples", -1)
        name = cfg.get("name", cfg["noisy_dir"])
        
        if num_samples == -1:
            # 使用全部数据
            print(f"数据集 {name}: 使用全部 {n_total} 张")
            all_datasets.append(ds)
        elif num_samples <= n_total:
            # 欠采样：随机选择不重复
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(n_total, generator=generator)[:num_samples].tolist()
            ds_sampled = Subset(ds, indices)
            print(f"数据集 {name}: 随机采样 {num_samples}/{n_total} 张（不重复）")
            all_datasets.append(ds_sampled)
        else:
            # 过采样：重复采样
            generator = torch.Generator().manual_seed(seed)
            # 先打乱所有索引
            base_indices = torch.randperm(n_total, generator=generator).tolist()
            # 计算需要重复多少轮
            n_repeats = num_samples // n_total
            n_remainder = num_samples % n_total
            # 完整重复 + 剩余部分
            indices = base_indices * n_repeats + base_indices[:n_remainder]
            ds_sampled = Subset(ds, indices)
            print(f"数据集 {name}: 重复采样 {num_samples}/{n_total} 张（重复 {n_repeats}+ 轮）")
            all_datasets.append(ds_sampled)
    
    # 合并所有数据集
    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return ConcatDataset(all_datasets)



class MultiPatchDataset(Dataset):
    """包装数据集，让每张图像返回多个随机裁剪的patch。
    
    这样可以增加每个epoch的有效训练样本数。
    
    Args:
        base_dataset: 基础数据集（Subset）
        patches_per_image: 每张图像返回多少个patch
        transform: 数据变换（应该包含随机裁剪）
    """
    
    def __init__(self, base_dataset: Dataset, patches_per_image: int, transform: Callable):
        self.base_dataset = base_dataset
        self.patches_per_image = patches_per_image
        self.transform = transform
    
    def __len__(self) -> int:
        # 总样本数 = 原始图像数 × 每张图的patch数
        return len(self.base_dataset) * self.patches_per_image
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # 计算对应的原始图像索引
        image_idx = idx // self.patches_per_image
        
        # 递归解包，追溯到原始的 DenoisingDataset 和真实索引
        dataset, real_idx = self._unwrap(self.base_dataset, image_idx)
        
        # 直接从 DenoisingDataset 读取文件路径
        noisy_path, clean_path = dataset.image_pairs[real_idx]
        from PIL import Image
        noisy = Image.open(noisy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        
        # 应用 transform（包含随机裁剪）
        if self.transform is not None:
            noisy, clean = self.transform(noisy, clean)
        
        return noisy, clean
    
    @staticmethod
    def _unwrap(dataset, idx):
        """递归解包 Subset / ConcatDataset，返回 (DenoisingDataset, real_index)"""
        while True:
            if isinstance(dataset, DenoisingDataset):
                return dataset, idx
            elif isinstance(dataset, Subset):
                idx = dataset.indices[idx]
                dataset = dataset.dataset
            elif isinstance(dataset, ConcatDataset):
                for sub_ds in dataset.datasets:
                    if idx < len(sub_ds):
                        dataset = sub_ds
                        break
                    idx -= len(sub_ds)
            else:
                raise TypeError(f"无法解包数据集类型: {type(dataset)}")
