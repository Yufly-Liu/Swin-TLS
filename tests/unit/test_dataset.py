"""DenoisingDataset 和 split_dataset 单元测试。"""

import os
import tempfile

import pytest
from PIL import Image

from data.dataset import DenoisingDataset, DatasetError, split_dataset


@pytest.fixture
def tmp_dataset():
    """创建包含噪声和干净图像对的临时数据集目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        noisy_dir = os.path.join(tmpdir, "noisy")
        clean_dir = os.path.join(tmpdir, "clean")
        os.makedirs(noisy_dir)
        os.makedirs(clean_dir)

        # 创建10个图像对
        for i in range(10):
            name = f"img_{i:03d}.png"
            Image.new("RGB", (32, 32), color=(i * 25, 0, 0)).save(
                os.path.join(noisy_dir, name)
            )
            Image.new("RGB", (32, 32), color=(0, i * 25, 0)).save(
                os.path.join(clean_dir, name)
            )

        yield noisy_dir, clean_dir


class TestDenoisingDataset:
    def test_loads_all_pairs(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        assert len(ds) == 10

    def test_getitem_returns_tensor_pair(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        noisy, clean = ds[0]
        assert noisy.shape == (3, 32, 32)
        assert clean.shape == (3, 32, 32)
        assert noisy.min() >= 0.0 and noisy.max() <= 1.0

    def test_missing_clean_raises(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        # Add an extra noisy image with no clean counterpart
        Image.new("RGB", (32, 32)).save(os.path.join(noisy_dir, "extra.png"))
        with pytest.raises(DatasetError, match="缺少对应的干净图像"):
            DenoisingDataset(noisy_dir, clean_dir)

    def test_nonexistent_dir_raises(self):
        with pytest.raises(DatasetError, match="目录不存在"):
            DenoisingDataset("/nonexistent/noisy", "/nonexistent/clean")

    def test_empty_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            noisy = os.path.join(tmpdir, "noisy")
            clean = os.path.join(tmpdir, "clean")
            os.makedirs(noisy)
            os.makedirs(clean)
            with pytest.raises(DatasetError, match="数据集为空"):
                DenoisingDataset(noisy, clean)


class TestSplitDataset:
    def test_default_split_sizes(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        train, val, test = split_dataset(ds)
        assert len(train) + len(val) + len(test) == len(ds)

    def test_no_overlap(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        train, val, test = split_dataset(ds)
        train_idx = set(train.indices)
        val_idx = set(val.indices)
        test_idx = set(test.indices)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_invalid_ratio_sum(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        with pytest.raises(ValueError, match="比例之和必须为1"):
            split_dataset(ds, 0.5, 0.5, 0.5)

    def test_negative_ratio(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        with pytest.raises(ValueError, match="不能为负数"):
            split_dataset(ds, -0.1, 0.6, 0.5)

    def test_reproducible_with_seed(self, tmp_dataset):
        noisy_dir, clean_dir = tmp_dataset
        ds = DenoisingDataset(noisy_dir, clean_dir)
        t1, v1, te1 = split_dataset(ds, seed=123)
        t2, v2, te2 = split_dataset(ds, seed=123)
        assert t1.indices == t2.indices
        assert v1.indices == v2.indices
        assert te1.indices == te2.indices
