"""训练器单元测试 - 验证 Trainer 类的核心功能。"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.dncnn import DnCNN
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(n_samples=8, batch_size=4, img_size=32):
    """创建一个简单的合成数据 DataLoader。"""
    noisy = torch.randn(n_samples, 3, img_size, img_size).clamp(0, 1)
    clean = torch.randn(n_samples, 3, img_size, img_size).clamp(0, 1)
    ds = TensorDataset(noisy, clean)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_trainer(tmp_path, scheduler_type="ReduceLROnPlateau"):
    """创建一个轻量级 Trainer 用于测试。"""
    model = DnCNN({"num_layers": 3, "num_features": 8})
    train_loader = _make_loader()
    val_loader = _make_loader(n_samples=4, batch_size=2)
    config = {
        "learning_rate": 1e-3,
        "num_epochs": 2,
        "checkpoint_dir": str(tmp_path / "ckpts"),
        "save_frequency": 1,
        "scheduler": {"type": scheduler_type, "patience": 2, "factor": 0.5,
                       "step_size": 1, "gamma": 0.5},
    }
    return Trainer(model, train_loader, val_loader, config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainerBasic:
    def test_train_epoch_returns_float(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate_returns_metrics(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        metrics = trainer.validate()
        assert "val_psnr" in metrics
        assert "val_ssim" in metrics
        assert "val_loss" in metrics

    def test_train_records_history(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        history = trainer.train(num_epochs=2)
        assert len(history["epoch"]) == 2
        assert len(history["train_loss"]) == 2
        assert len(history["val_psnr"]) == 2
        assert len(history["learning_rate"]) == 2


class TestCheckpoint:
    def test_save_and_load_checkpoint(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.train(num_epochs=2)

        ckpt_path = str(tmp_path / "test_ckpt.pth")
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

        # 创建新 trainer 并加载
        trainer2 = _make_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.current_epoch == trainer.current_epoch
        assert abs(trainer2.best_val_psnr - trainer.best_val_psnr) < 1e-6

    def test_load_nonexistent_raises(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(str(tmp_path / "nope.pth"))


class TestScheduler:
    def test_step_lr_scheduler(self, tmp_path):
        trainer = _make_trainer(tmp_path, scheduler_type="StepLR")
        assert isinstance(
            trainer.scheduler, torch.optim.lr_scheduler.StepLR
        )

    def test_reduce_on_plateau_scheduler(self, tmp_path):
        trainer = _make_trainer(tmp_path, scheduler_type="ReduceLROnPlateau")
        assert isinstance(
            trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    def test_best_model_saved(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.train(num_epochs=2)
        best_path = os.path.join(trainer.checkpoint_dir, "best_model.pth")
        assert os.path.exists(best_path)
