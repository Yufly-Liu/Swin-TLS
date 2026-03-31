"""属性测试 - 评估指标模块

Feature: image-denoising-comparison
Properties 12-16: PSNR计算正确性、SSIM计算边界性、MSE计算正确性、
                  评估结果统计完整性、样本图像保存数量一致性
"""

import os
import tempfile

import pytest
import torch
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import Metrics, MetricError
from evaluation.evaluator import Evaluator
from models.dncnn import DnCNN


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _image_tensor(batch=False, min_size=11, max_size=32):
    """Strategy that generates image tensors with values in [0, 1]."""
    if batch:
        return st.builds(
            lambda b, h, w: torch.rand(b, 3, h, w),
            b=st.integers(min_value=1, max_value=2),
            h=st.integers(min_value=min_size, max_value=max_size),
            w=st.integers(min_value=min_size, max_value=max_size),
        )
    return st.builds(
        lambda h, w: torch.rand(3, h, w),
        h=st.integers(min_value=min_size, max_value=max_size),
        w=st.integers(min_value=min_size, max_value=max_size),
    )


def _image_pair(min_size=11, max_size=32):
    """Strategy that generates a pair of image tensors with the same shape."""
    return st.builds(
        lambda h, w: (torch.rand(3, h, w), torch.rand(3, h, w)),
        h=st.integers(min_value=min_size, max_value=max_size),
        w=st.integers(min_value=min_size, max_value=max_size),
    )


# ---------------------------------------------------------------------------
# Property 12: PSNR计算正确性
# ---------------------------------------------------------------------------

class TestProperty12PSNRCorrectness:
    """Feature: image-denoising-comparison, Property 12: PSNR计算正确性

    对于任意两张相同的图像，PSNR值应该趋向无穷大；对于完全不同的图像，
    PSNR值应该较低。PSNR计算应该满足：PSNR = 10 * log10(MAX^2 / MSE)。

    Validates: Requirements 4.1
    """

    @given(img=_image_tensor())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_identical_images_high_psnr(self, img):
        """**Validates: Requirements 4.1**

        PSNR of identical images should be very high (100.0 as proxy for infinity).
        """
        psnr = Metrics.calculate_psnr(img, img)
        assert psnr == 100.0, f"PSNR of identical images should be 100.0, got {psnr}"

    @given(data=_image_pair())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_psnr_formula_consistency(self, data):
        """**Validates: Requirements 4.1**

        PSNR should equal 10 * log10(MAX^2 / MSE) for non-identical images.
        """
        img1, img2 = data
        mse_val = torch.mean((img1.float() - img2.float()) ** 2).item()
        assume(mse_val >= 1e-10)  # skip near-identical

        psnr = Metrics.calculate_psnr(img1, img2)
        expected = 10.0 * torch.log10(torch.tensor(1.0 / mse_val)).item()
        assert abs(psnr - expected) < 1e-4, (
            f"PSNR mismatch: got {psnr}, expected {expected}"
        )

    @given(img=_image_tensor())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_psnr_non_negative(self, img):
        """**Validates: Requirements 4.1**

        PSNR should always be non-negative for images in [0, 1].
        """
        # Compare with a different image (zeros)
        other = torch.zeros_like(img)
        psnr = Metrics.calculate_psnr(img, other)
        assert psnr >= 0, f"PSNR should be non-negative, got {psnr}"


# ---------------------------------------------------------------------------
# Property 13: SSIM计算边界性
# ---------------------------------------------------------------------------

class TestProperty13SSIMBoundedness:
    """Feature: image-denoising-comparison, Property 13: SSIM计算边界性

    对于任意两张图像，SSIM值应该在[0, 1]范围内，
    且两张相同图像的SSIM值应该等于1。

    Validates: Requirements 4.2
    """

    @given(img=_image_tensor(min_size=11, max_size=32))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_identical_images_ssim_one(self, img):
        """**Validates: Requirements 4.2**

        SSIM of identical images should be 1.0.
        """
        ssim = Metrics.calculate_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-4, (
            f"SSIM of identical images should be 1.0, got {ssim}"
        )

    @given(data=_image_pair(min_size=11, max_size=32))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_ssim_in_valid_range(self, data):
        """**Validates: Requirements 4.2**

        SSIM should always be in [0, 1].
        """
        img1, img2 = data
        ssim = Metrics.calculate_ssim(img1, img2)
        assert 0.0 <= ssim <= 1.0, f"SSIM out of range: {ssim}"


# ---------------------------------------------------------------------------
# Property 14: MSE计算正确性
# ---------------------------------------------------------------------------

class TestProperty14MSECorrectness:
    """Feature: image-denoising-comparison, Property 14: MSE计算正确性

    对于任意两张相同的图像，MSE值应该等于0；MSE值应该始终非负。

    Validates: Requirements 4.3
    """

    @given(img=_image_tensor())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_identical_images_zero_mse(self, img):
        """**Validates: Requirements 4.3**

        MSE of identical images should be 0.0.
        """
        mse = Metrics.calculate_mse(img, img)
        assert abs(mse) < 1e-10, f"MSE of identical images should be 0, got {mse}"

    @given(data=_image_pair())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_mse_non_negative(self, data):
        """**Validates: Requirements 4.3**

        MSE should always be non-negative.
        """
        img1, img2 = data
        mse = Metrics.calculate_mse(img1, img2)
        assert mse >= 0.0, f"MSE should be non-negative, got {mse}"

    @given(data=_image_pair())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_mse_symmetric(self, data):
        """**Validates: Requirements 4.3**

        MSE(a, b) should equal MSE(b, a).
        """
        img1, img2 = data
        mse_ab = Metrics.calculate_mse(img1, img2)
        mse_ba = Metrics.calculate_mse(img2, img1)
        assert abs(mse_ab - mse_ba) < 1e-6, (
            f"MSE not symmetric: {mse_ab} vs {mse_ba}"
        )


# ---------------------------------------------------------------------------
# Helpers for Properties 15 & 16
# ---------------------------------------------------------------------------

def _make_test_loader(n_samples=8, batch_size=4, img_size=32):
    """创建合成测试数据 DataLoader。"""
    noisy = torch.rand(n_samples, 3, img_size, img_size)
    clean = torch.rand(n_samples, 3, img_size, img_size)
    ds = TensorDataset(noisy, clean)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_evaluator(n_samples=8, batch_size=4):
    """创建轻量级 Evaluator 用于测试。"""
    model = DnCNN({"num_layers": 3, "num_features": 8})
    loader = _make_test_loader(n_samples=n_samples, batch_size=batch_size)
    return Evaluator(model, loader)


# ---------------------------------------------------------------------------
# Property 15: 评估结果统计完整性
# ---------------------------------------------------------------------------

class TestProperty15EvaluationStatsCompleteness:
    """Feature: image-denoising-comparison, Property 15: 评估结果统计完整性

    对于任意测试集评估结果，返回的指标字典应该包含每个指标的均值和标准差。

    Validates: Requirements 4.6
    """

    @given(n_samples=st.integers(min_value=2, max_value=8))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_results_contain_mean_and_std(self, n_samples):
        """**Validates: Requirements 4.6**

        Evaluation results must contain mean and std for psnr, ssim, and mse.
        """
        evaluator = _make_evaluator(n_samples=n_samples, batch_size=2)
        results = evaluator.evaluate()

        required_metrics = ["psnr", "ssim", "mse"]
        for metric in required_metrics:
            assert metric in results, f"Missing metric: {metric}"
            assert "mean" in results[metric], f"Missing mean for {metric}"
            assert "std" in results[metric], f"Missing std for {metric}"
            assert isinstance(results[metric]["mean"], float)
            assert isinstance(results[metric]["std"], float)
            assert results[metric]["std"] >= 0, f"Std should be non-negative for {metric}"


# ---------------------------------------------------------------------------
# Property 16: 样本图像保存数量一致性
# ---------------------------------------------------------------------------

class TestProperty16SampleSaveCountConsistency:
    """Feature: image-denoising-comparison, Property 16: 样本图像保存数量一致性

    对于任意评估过程，如果指定保存N个样本，则应该生成恰好N个去噪结果图像文件。

    Validates: Requirements 4.5
    """

    @given(num_samples=st.integers(min_value=1, max_value=6))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_saved_sample_count_matches_request(self, num_samples):
        """**Validates: Requirements 4.5**

        save_sample_results should produce exactly num_samples files
        (or all available if fewer than num_samples in the dataset).
        """
        dataset_size = 8
        evaluator = _make_evaluator(n_samples=dataset_size, batch_size=2)
        expected_count = min(num_samples, dataset_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = evaluator.save_sample_results(tmp_dir, num_samples=num_samples)

            assert len(paths) == expected_count, (
                f"Expected {expected_count} samples, got {len(paths)}"
            )

            # Verify all files actually exist
            for p in paths:
                assert os.path.exists(p), f"Sample file not found: {p}"
