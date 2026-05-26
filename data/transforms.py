"""数据增强和预处理模块 - 为去噪图像对提供一致的变换。"""

import random
from typing import Tuple

import torch
from PIL import Image, ImageEnhance
from torch import Tensor
import torchvision.transforms.functional as TF


class DataTransforms:
    """为去噪数据集提供训练和验证/测试时的数据变换。

    所有空间变换（翻转、旋转）对噪声图像和干净图像同步应用，
    以保持配对关系。
    """

    def __init__(self, target_size: int = 256):
        """
        Args:
            target_size: 输出图像的目标尺寸（正方形裁剪/缩放边长）。
        """
        self.target_size = target_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_train_transforms(target_size: int = 256, use_random_crop: bool = False,
                           patches_per_image: int = 1) -> "TrainTransform":
        """返回训练时的数据增强变换。

        包含：随机裁剪/缩放、随机水平翻转、随机垂直翻转、随机90°旋转、
        随机亮度变化、归一化到 [0, 1]。
        
        Args:
            target_size: 输出patch尺寸
            use_random_crop: True=从大图随机裁剪patch, False=resize到目标尺寸
            patches_per_image: 每张图像裁剪多少个patch
        """
        return TrainTransform(target_size, use_random_crop, patches_per_image)

    @staticmethod
    def get_val_transforms(target_size: int = 256) -> "ValTransform":
        """返回验证/测试时的预处理变换。

        包含：中心裁剪/缩放到目标尺寸、归一化到 [0, 1]。
        """
        return ValTransform(target_size)


class TrainTransform:
    """训练时数据增强：随机裁剪patch、随机翻转、旋转、亮度变化。

    对噪声和干净图像同步应用空间变换，保持配对关系。
    亮度变化仅应用于噪声图像（模拟不同光照条件下的噪声）。
    """

    def __init__(self, target_size: int = 256, use_random_crop: bool = False, 
                 patches_per_image: int = 1):
        """
        Args:
            target_size: 输出patch尺寸
            use_random_crop: True=随机裁剪patch, False=resize到目标尺寸
            patches_per_image: 每张图像裁剪多少个patch（仅在use_random_crop=True时有效）
        """
        self.target_size = target_size
        self.use_random_crop = use_random_crop
        self.patches_per_image = patches_per_image

    def __call__(self, noisy: Image.Image, clean: Image.Image) -> Tuple[Tensor, Tensor]:
        """返回单个patch（会被外部循环调用多次）"""
        # 1. 裁剪或缩放
        if self.use_random_crop:
            # 随机裁剪 patch（保留原始分辨率）
            w, h = noisy.size
            if w < self.target_size or h < self.target_size:
                # 图像太小，先 resize 再裁剪
                scale = max(self.target_size / w, self.target_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                noisy = TF.resize(noisy, [new_h, new_w])
                clean = TF.resize(clean, [new_h, new_w])
                w, h = new_w, new_h
            
            # 随机裁剪位置
            top = random.randint(0, h - self.target_size)
            left = random.randint(0, w - self.target_size)
            noisy = TF.crop(noisy, top, left, self.target_size, self.target_size)
            clean = TF.crop(clean, top, left, self.target_size, self.target_size)
        else:
            # 直接缩放到目标尺寸
            noisy = TF.resize(noisy, [self.target_size, self.target_size])
            clean = TF.resize(clean, [self.target_size, self.target_size])

        # 2. 随机水平翻转
        if random.random() > 0.5:
            noisy = TF.hflip(noisy)
            clean = TF.hflip(clean)

        # 3. 随机垂直翻转
        if random.random() > 0.5:
            noisy = TF.vflip(noisy)
            clean = TF.vflip(clean)

        # 4. 随机90°旋转 (0, 90, 180, 270)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            noisy = TF.rotate(noisy, angle)
            clean = TF.rotate(clean, angle)

        # 5. 随机亮度变化（同步应用于两张图像以保持对应关系）
        brightness_factor = random.uniform(0.8, 1.2)
        noisy = ImageEnhance.Brightness(noisy).enhance(brightness_factor)
        clean = ImageEnhance.Brightness(clean).enhance(brightness_factor)

        # 6. 转换为张量 [0, 1]
        noisy_t = TF.to_tensor(noisy)
        clean_t = TF.to_tensor(clean)

        return noisy_t, clean_t


class ValTransform:
    """验证/测试时预处理：缩放到目标尺寸、归一化。"""

    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def __call__(self, noisy: Image.Image, clean: Image.Image) -> Tuple[Tensor, Tensor]:
        # 1. 缩放到目标尺寸
        noisy = TF.resize(noisy, [self.target_size, self.target_size])
        clean = TF.resize(clean, [self.target_size, self.target_size])

        # 2. 转换为张量 [0, 1]
        noisy_t = TF.to_tensor(noisy)
        clean_t = TF.to_tensor(clean)

        return noisy_t, clean_t
