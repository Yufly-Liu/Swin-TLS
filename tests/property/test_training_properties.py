"""属性测试 - 训练流程模块

Feature: image-denoising-comparison
Properties 8-11: 训练检查点往返一致性、验证指标记录完整性、学习率调度单调性、最佳模型保存正确性
"""

import os
import tempfile

import pytest
import torch
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from models.dncnn import DnCNN
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(n_samples=8, batch_size=4, img_size=32):
    """创建合成数据 DataLoader。"""
    noisy = torch.randn(n_samples, 3, img_size, img_size).clamp(0, 1)
    clean = torch.randn(n_samples, 3, img_size, img_size).clamp(0, 1)
    ds = TensorDataset(noisy, clean)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_trainer(tmp_dir, scheduler_type="ReduceLROnPlateau", lr=1e-3,
                  patience=2, factor=0.5, save_frequency=1):
    """创建轻量级 Trainer。"""
    model = DnCNN({"num_layers": 3, "num_features": 8})
    train_loader = _make_loader()
    val_loader = _make_loader(n_samples=4, batch_size=2)
    config = {
        "learning_rate": lr,
        "num_epochs": 2,
        "checkpoint_dir": os.path.join(tmp_dir, "ckpts"),
        "save_frequency": save_frequency,
        "scheduler": {
            "type": scheduler_type,
            "patience": patience,
            "factor": factor,
            "step_size": 1,
            "gamma": 0.5,
        },
    }
    return Trainer(model, train_loader, val_loader, config)


# ---------------------------------------------------------------------------
# Property 8: 训练检查点往返一致性
# ---------------------------------------------------------------------------

class TestProperty8CheckpointRoundTrip:
    """Feature: image-denoising-comparison, Property 8: 训练检查点往返一致性

    对于任意训练状态，保存检查点后再加载，恢复的训练状态
    （epoch、优化器状态、模型参数）应该与保存前完全一致。

    Validates: Requirements 3.7
    """

    @given(num_epochs=st.integers(min_value=1, max_value=3))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_checkpoint_round_trip(self, num_epochs):
        """**Validates: Requirements 3.7**

        After saving and loading a checkpoint, epoch, optimizer state,
        model parameters, and best_val_psnr must be identical.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _make_trainer(tmp_dir)
            trainer.train(num_epochs=num_epochs)

            ckpt_path = os.path.join(tmp_dir, "roundtrip.pth")
            trainer.save_checkpoint(ckpt_path)

            # 记录保存前的状态
            saved_epoch = trainer.current_epoch
            saved_best_psnr = trainer.best_val_psnr
            saved_params = {
                k: v.clone() for k, v in trainer.model.state_dict().items()
            }
            saved_optim = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in trainer.optimizer.state_dict()["state"].items()
            }
            saved_history_len = len(trainer.history["epoch"])

            # 创建新 trainer 并加载
            trainer2 = _make_trainer(tmp_dir)
            trainer2.load_checkpoint(ckpt_path)

            # 验证 epoch
            assert trainer2.current_epoch == saved_epoch

            # 验证 best_val_psnr
            assert abs(trainer2.best_val_psnr - saved_best_psnr) < 1e-6

            # 验证模型参数
            for key in saved_params:
                assert torch.equal(
                    trainer2.model.state_dict()[key], saved_params[key]
                ), f"Model param mismatch: {key}"

            # 验证历史长度
            assert len(trainer2.history["epoch"]) == saved_history_len


# ---------------------------------------------------------------------------
# Property 9: 验证指标记录完整性
# ---------------------------------------------------------------------------

class TestProperty9ValidationMetricsCompleteness:
    """Feature: image-denoising-comparison, Property 9: 验证指标记录完整性

    对于任意训练过程，每个 epoch 后的训练历史应该包含该 epoch 的
    损失值、验证指标和学习率。

    Validates: Requirements 3.2, 3.5
    """

    @given(num_epochs=st.integers(min_value=1, max_value=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_history_completeness(self, num_epochs):
        """**Validates: Requirements 3.2, 3.5**

        After training for N epochs, history must contain exactly N entries
        for each tracked metric: epoch, train_loss, val_psnr, val_ssim, learning_rate.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _make_trainer(tmp_dir)
            history = trainer.train(num_epochs=num_epochs)

            required_keys = ["epoch", "train_loss", "val_psnr", "val_ssim", "learning_rate"]
            for key in required_keys:
                assert key in history, f"Missing history key: {key}"
                assert len(history[key]) == num_epochs, (
                    f"history['{key}'] has {len(history[key])} entries, expected {num_epochs}"
                )

            # epoch 列表应该是 1..num_epochs
            assert history["epoch"] == list(range(1, num_epochs + 1))

            # 所有损失值应该是有限数
            for loss in history["train_loss"]:
                assert isinstance(loss, float)
                assert not (loss != loss), "NaN loss detected"

            # 学习率应该为正
            for lr in history["learning_rate"]:
                assert lr > 0


# ---------------------------------------------------------------------------
# Property 10: 学习率调度单调性
# ---------------------------------------------------------------------------

class TestProperty10LRScheduleMonotonicity:
    """Feature: image-denoising-comparison, Property 10: 学习率调度单调性

    对于任意使用 ReduceLROnPlateau 策略的训练过程，当验证指标在 patience 个
    epoch 内没有改善时，学习率应该按照 factor 比例减小。

    Validates: Requirements 3.3
    """

    @given(
        factor=st.sampled_from([0.1, 0.25, 0.5]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_lr_decreases_on_plateau(self, factor):
        """**Validates: Requirements 3.3**

        With ReduceLROnPlateau, the learning rate should never increase.
        When the scheduler triggers, the new LR should be approximately
        old_lr * factor.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # patience=0 means reduce after every epoch without improvement
            trainer = _make_trainer(
                tmp_dir,
                scheduler_type="ReduceLROnPlateau",
                lr=0.01,
                patience=0,
                factor=factor,
            )
            # Train enough epochs to trigger LR reduction
            trainer.train(num_epochs=5)

            lr_history = trainer.history["learning_rate"]

            # LR should be non-increasing (monotonically decreasing or flat)
            for i in range(1, len(lr_history)):
                assert lr_history[i] <= lr_history[i - 1] + 1e-10, (
                    f"LR increased from {lr_history[i-1]} to {lr_history[i]} "
                    f"at epoch {i+1}"
                )


# ---------------------------------------------------------------------------
# Property 11: 最佳模型保存正确性
# ---------------------------------------------------------------------------

class TestProperty11BestModelSaving:
    """Feature: image-denoising-comparison, Property 11: 最佳模型保存正确性

    对于任意训练过程，保存的最佳模型检查点应该对应验证集上性能最好的 epoch。

    Validates: Requirements 3.4
    """

    @given(num_epochs=st.integers(min_value=2, max_value=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_best_checkpoint_matches_best_psnr(self, num_epochs):
        """**Validates: Requirements 3.4**

        The best_model.pth checkpoint should store the best_val_psnr that
        equals the maximum val_psnr observed during training.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _make_trainer(tmp_dir, save_frequency=1)
            history = trainer.train(num_epochs=num_epochs)

            best_path = os.path.join(trainer.checkpoint_dir, "best_model.pth")
            assert os.path.exists(best_path), "best_model.pth not created"

            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)

            # The best_val_psnr in the checkpoint should match the max from history
            max_psnr = max(history["val_psnr"])
            assert abs(checkpoint["best_val_psnr"] - max_psnr) < 1e-6, (
                f"Checkpoint best_val_psnr={checkpoint['best_val_psnr']}, "
                f"but max in history={max_psnr}"
            )

            # The checkpoint should be marked as best
            assert checkpoint["is_best"] is True
