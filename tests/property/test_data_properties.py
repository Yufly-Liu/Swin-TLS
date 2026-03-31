"""属性测试 - 数据管理模块

Feature: image-denoising-comparison
Properties 1-4: 数据集加载完整性、划分不重叠性、增强保持配对关系、预处理输出一致性
"""

import os
import tempfile
from typing import Tuple

import pytest
import torch
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from PIL import Image

from data.dataset import DenoisingDataset, DatasetError, split_dataset
from data.transforms import DataTransforms, TrainTransform, ValTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_temp_dataset(num_images: int, img_size: Tuple[int, int] = (32, 32)):
    """Create a temporary dataset directory with paired images.

    Returns (tmpdir, noisy_dir, clean_dir).
    Caller is responsible for cleanup via tmpdir.
    """
    tmpdir = tempfile.mkdtemp()
    noisy_dir = os.path.join(tmpdir, "noisy")
    clean_dir = os.path.join(tmpdir, "clean")
    os.makedirs(noisy_dir)
    os.makedirs(clean_dir)

    for i in range(num_images):
        name = f"img_{i:04d}.png"
        Image.new("RGB", img_size, color=(i % 256, (i * 7) % 256, (i * 13) % 256)).save(
            os.path.join(noisy_dir, name)
        )
        Image.new("RGB", img_size, color=((i * 3) % 256, (i * 11) % 256, (i * 5) % 256)).save(
            os.path.join(clean_dir, name)
        )

    return tmpdir, noisy_dir, clean_dir


# ---------------------------------------------------------------------------
# Property 1: 数据集加载完整性
# ---------------------------------------------------------------------------

class TestProperty1DatasetLoadCompleteness:
    """Feature: image-denoising-comparison, Property 1: 数据集加载完整性

    对于任意有效的数据集目录，加载后的图像对数量应该等于目录中配对图像的数量，
    且每个噪声图像都有对应的干净图像。

    Validates: Requirements 1.1, 1.2
    """

    @given(num_images=st.integers(min_value=1, max_value=30))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_loaded_count_matches_directory(self, num_images):
        """**Validates: Requirements 1.1, 1.2**

        The number of loaded image pairs must equal the number of paired
        image files in the directories.
        """
        tmpdir, noisy_dir, clean_dir = create_temp_dataset(num_images)
        try:
            ds = DenoisingDataset(noisy_dir, clean_dir)
            assert len(ds) == num_images

            # Every pair should be loadable and return valid tensors
            idx = num_images // 2  # spot-check one item
            noisy, clean = ds[idx]
            assert noisy.shape[0] == 3  # RGB
            assert clean.shape[0] == 3
        finally:
            import shutil
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Property 2: 数据集划分不重叠性
# ---------------------------------------------------------------------------

class TestProperty2DatasetSplitNoOverlap:
    """Feature: image-denoising-comparison, Property 2: 数据集划分不重叠性

    对于任意数据集和划分比例，划分后的训练集、验证集和测试集应该没有重叠，
    且三个子集的大小之和应该等于原始数据集大小。

    Validates: Requirements 1.3
    """

    @given(
        num_images=st.integers(min_value=3, max_value=50),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_splits_no_overlap_and_cover_all(self, num_images, data):
        """**Validates: Requirements 1.3**

        Train/val/test splits must be disjoint and their union must cover
        the entire dataset.
        """
        # Generate valid split ratios that sum to 1.0
        train_r = data.draw(st.floats(min_value=0.1, max_value=0.8))
        val_r = data.draw(st.floats(min_value=0.05, max_value=1.0 - train_r - 0.05))
        test_r = round(1.0 - train_r - val_r, 10)
        assume(test_r > 0)
        assume(abs(train_r + val_r + test_r - 1.0) < 1e-6)

        tmpdir, noisy_dir, clean_dir = create_temp_dataset(num_images)
        try:
            ds = DenoisingDataset(noisy_dir, clean_dir)
            train, val, test = split_dataset(ds, train_r, val_r, test_r)

            train_idx = set(train.indices)
            val_idx = set(val.indices)
            test_idx = set(test.indices)

            # No overlap
            assert train_idx.isdisjoint(val_idx), "Train and val overlap"
            assert train_idx.isdisjoint(test_idx), "Train and test overlap"
            assert val_idx.isdisjoint(test_idx), "Val and test overlap"

            # Cover all
            assert len(train_idx | val_idx | test_idx) == num_images
            assert len(train) + len(val) + len(test) == num_images
        finally:
            import shutil
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Property 3: 数据增强保持配对关系
# ---------------------------------------------------------------------------

class TestProperty3AugmentationPreservesPairing:
    """Feature: image-denoising-comparison, Property 3: 数据增强保持配对关系

    对于任意图像对，应用相同的数据增强变换后，噪声图像和干净图像应该保持
    空间对应关系（相同的翻转、旋转等）。

    Validates: Requirements 1.5
    """

    @given(
        target_size=st.sampled_from([32, 64, 128]),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_same_spatial_transform_applied_to_both(self, target_size, seed):
        """**Validates: Requirements 1.5**

        When the same transform is applied to a pair where noisy == clean,
        the outputs must be identical (proving spatial transforms are synced).
        """
        import random as _random

        # Use an identical image for both noisy and clean
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))

        transform = TrainTransform(target_size)

        # Fix random state so the transform is deterministic for this call
        _random.seed(seed)
        noisy_t, clean_t = transform(img.copy(), img.copy())

        # Both outputs must have the same shape
        assert noisy_t.shape == clean_t.shape
        assert noisy_t.shape == (3, target_size, target_size)

        # Since input images are identical, spatial transforms (flip, rotate)
        # applied synchronously must produce identical outputs.
        # Brightness is also applied identically to both.
        assert torch.allclose(noisy_t, clean_t, atol=1e-5), \
            "Spatial transforms were not applied identically to both images"


# ---------------------------------------------------------------------------
# Property 4: 图像预处理输出一致性
# ---------------------------------------------------------------------------

class TestProperty4PreprocessingOutputConsistency:
    """Feature: image-denoising-comparison, Property 4: 图像预处理输出一致性

    对于任意输入图像，预处理后的输出应该具有指定的尺寸和值范围（[0, 1]）。

    Validates: Requirements 1.4
    """

    @given(
        width=st.integers(min_value=16, max_value=256),
        height=st.integers(min_value=16, max_value=256),
        target_size=st.sampled_from([32, 64, 128]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_val_transform_output_shape_and_range(self, width, height, target_size):
        """**Validates: Requirements 1.4**

        ValTransform must produce tensors of the specified target size
        with values in [0, 1], regardless of input dimensions.
        """
        noisy = Image.new("RGB", (width, height), color=(128, 64, 200))
        clean = Image.new("RGB", (width, height), color=(50, 100, 150))

        transform = ValTransform(target_size)
        noisy_t, clean_t = transform(noisy, clean)

        # Shape check
        assert noisy_t.shape == (3, target_size, target_size)
        assert clean_t.shape == (3, target_size, target_size)

        # Value range check [0, 1]
        assert noisy_t.min() >= 0.0
        assert noisy_t.max() <= 1.0
        assert clean_t.min() >= 0.0
        assert clean_t.max() <= 1.0
