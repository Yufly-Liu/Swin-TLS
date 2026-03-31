"""端到端集成测试 - 验证所有模块协同工作。

使用小型合成数据集和少量epoch运行完整的训练、评估和对比流程。
"""

import os
import tempfile

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import DenoisingDataset, split_dataset
from data.transforms import DataTransforms
from models import get_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics
from experiments.experiment_manager import ExperimentManager
from utils.config import ConfigManager
from utils.visualization import Visualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset_dir():
    """创建包含合成噪声/干净图像对的临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        noisy_dir = os.path.join(tmpdir, "noisy")
        clean_dir = os.path.join(tmpdir, "clean")
        os.makedirs(noisy_dir)
        os.makedirs(clean_dir)

        # 创建20个64x64 RGB图像对
        for i in range(20):
            name = f"img_{i:03d}.png"
            # 干净图像: 渐变色
            clean_img = Image.new("RGB", (64, 64), color=(i * 12, 100, 200 - i * 8))
            clean_img.save(os.path.join(clean_dir, name))
            # 噪声图像: 在干净图像基础上加一些变化
            noisy_img = Image.new("RGB", (64, 64), color=(i * 12 + 10, 110, 210 - i * 8))
            noisy_img.save(os.path.join(noisy_dir, name))

        yield tmpdir, noisy_dir, clean_dir


@pytest.fixture
def work_dir():
    """创建临时工作目录用于保存检查点、实验结果等。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Lightweight model config for fast testing
LIGHT_CONFIG = {
    "dncnn": {"in_channels": 3, "out_channels": 3, "num_layers": 3, "num_features": 8},
}


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """端到端集成测试：数据加载 → 训练 → 评估 → 实验管理 → 可视化。"""

    def test_full_pipeline_dncnn(self, synthetic_dataset_dir, work_dir):
        """使用DnCNN运行完整的训练-评估-对比流程。"""
        tmpdir, noisy_dir, clean_dir = synthetic_dataset_dir

        # ---- 1. 数据加载和划分 ----
        val_transform = DataTransforms.get_val_transforms(target_size=32)
        dataset = DenoisingDataset(noisy_dir, clean_dir, transform=val_transform)
        assert len(dataset) == 20

        train_set, val_set, test_set = split_dataset(
            dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        assert len(train_set) + len(val_set) + len(test_set) == 20

        train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

        # ---- 2. 模型创建 ----
        model = get_model("dncnn", LIGHT_CONFIG["dncnn"])
        assert model.count_parameters() > 0

        # ---- 3. 训练 ----
        checkpoint_dir = os.path.join(work_dir, "checkpoints")
        train_config = {
            "learning_rate": 1e-3,
            "num_epochs": 3,
            "checkpoint_dir": checkpoint_dir,
            "save_frequency": 1,
            "scheduler": {"type": "ReduceLROnPlateau", "patience": 2, "factor": 0.5},
        }
        trainer = Trainer(model, train_loader, val_loader, train_config)
        history = trainer.train(num_epochs=3)

        # 验证训练历史
        assert len(history["epoch"]) == 3
        assert len(history["train_loss"]) == 3
        assert len(history["val_psnr"]) == 3
        assert len(history["val_ssim"]) == 3
        assert len(history["learning_rate"]) == 3
        assert all(isinstance(v, float) for v in history["train_loss"])

        # 验证检查点已保存
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        assert os.path.exists(best_model_path)

        # ---- 4. 评估 ----
        evaluator = Evaluator(model, test_loader)
        results = evaluator.evaluate()

        assert "psnr" in results
        assert "ssim" in results
        assert "mse" in results
        assert "mean" in results["psnr"] and "std" in results["psnr"]
        assert results["ssim"]["mean"] >= 0.0
        assert results["mse"]["mean"] >= 0.0

        # 保存去噪样本
        samples_dir = os.path.join(work_dir, "samples")
        saved_paths = evaluator.save_sample_results(samples_dir, num_samples=3)
        assert len(saved_paths) == min(3, len(test_set))
        for p in saved_paths:
            assert os.path.exists(p)

        # ---- 5. 实验管理 ----
        exp_dir = os.path.join(work_dir, "experiments")
        manager = ExperimentManager(base_dir=exp_dir)

        exp_config = {
            "model": {"name": "dncnn"},
            "training": train_config,
        }
        exp_id = manager.create_experiment("integration_test_dncnn", exp_config)
        assert exp_id is not None

        exp_results = {
            "model_name": "dncnn",
            "test_metrics": results,
            "training_time": 1.0,
            "inference_time": 0.01,
            "model_params": model.count_parameters(),
        }
        manager.save_results(exp_id, exp_results)

        loaded = manager.load_results(exp_id)
        assert loaded["model_name"] == "dncnn"
        assert "test_metrics" in loaded

        experiments = manager.list_experiments()
        assert len(experiments) >= 1

        # ---- 6. 可视化 ----
        vis_dir = os.path.join(work_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 训练曲线
        curves_path = os.path.join(vis_dir, "training_curves.png")
        Visualizer.plot_training_curves(history, curves_path)
        assert os.path.exists(curves_path)

        # 去噪对比图
        noisy_sample, clean_sample = test_set[0]
        model.eval()
        with torch.no_grad():
            denoised_sample = model(noisy_sample.unsqueeze(0)).squeeze(0).clamp(0, 1)
        comparison_path = os.path.join(vis_dir, "comparison.png")
        Visualizer.save_image_comparison(noisy_sample, denoised_sample, clean_sample, comparison_path)
        assert os.path.exists(comparison_path)

    def test_checkpoint_resume_training(self, synthetic_dataset_dir, work_dir):
        """验证检查点保存和恢复训练的完整流程。"""
        tmpdir, noisy_dir, clean_dir = synthetic_dataset_dir

        val_transform = DataTransforms.get_val_transforms(target_size=32)
        dataset = DenoisingDataset(noisy_dir, clean_dir, transform=val_transform)
        train_set, val_set, _ = split_dataset(dataset, 0.7, 0.15, 0.15, seed=42)

        train_loader = DataLoader(train_set, batch_size=4, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

        checkpoint_dir = os.path.join(work_dir, "ckpts")
        config = {
            "learning_rate": 1e-3,
            "checkpoint_dir": checkpoint_dir,
            "save_frequency": 1,
            "scheduler": {"type": "StepLR", "step_size": 2, "gamma": 0.5},
        }

        # 训练2个epoch并保存
        model1 = get_model("dncnn", LIGHT_CONFIG["dncnn"])
        trainer1 = Trainer(model1, train_loader, val_loader, config)
        trainer1.train(num_epochs=2)

        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_epoch_2.pth")
        assert os.path.exists(ckpt_path)

        # 加载检查点到新trainer
        model2 = get_model("dncnn", LIGHT_CONFIG["dncnn"])
        trainer2 = Trainer(model2, train_loader, val_loader, config)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.current_epoch == 2
        assert len(trainer2.history["epoch"]) == 2

    def test_config_driven_workflow(self, synthetic_dataset_dir, work_dir):
        """验证配置驱动的工作流：加载配置 → 创建模型 → 训练。"""
        tmpdir, noisy_dir, clean_dir = synthetic_dataset_dir

        # 创建配置文件
        config = ConfigManager.get_default_config()
        config["dataset"]["noisy_dir"] = noisy_dir
        config["dataset"]["clean_dir"] = clean_dir
        config["dataset"]["batch_size"] = 4
        config["model"]["name"] = "dncnn"
        config["model"]["params"] = LIGHT_CONFIG["dncnn"]
        config["training"]["num_epochs"] = 2
        config["training"]["checkpoint_dir"] = os.path.join(work_dir, "ckpts")
        config["experiment"]["output_dir"] = os.path.join(work_dir, "experiments")

        config_path = os.path.join(work_dir, "test_config.yaml")
        ConfigManager.save_config(config, config_path)

        # 加载并验证配置
        loaded_config = ConfigManager.load_config(config_path)
        assert ConfigManager.validate_config(loaded_config)

        # 使用配置创建数据集和模型
        ds_cfg = loaded_config["dataset"]
        val_transform = DataTransforms.get_val_transforms(target_size=32)
        dataset = DenoisingDataset(ds_cfg["noisy_dir"], ds_cfg["clean_dir"], transform=val_transform)

        train_set, val_set, test_set = split_dataset(
            dataset,
            ds_cfg["train_split"], ds_cfg["val_split"], ds_cfg["test_split"]
        )

        model = get_model(
            loaded_config["model"]["name"],
            loaded_config["model"]["params"]
        )

        train_loader = DataLoader(train_set, batch_size=ds_cfg["batch_size"])
        val_loader = DataLoader(val_set, batch_size=ds_cfg["batch_size"])

        trainer = Trainer(model, train_loader, val_loader, loaded_config["training"])
        history = trainer.train(num_epochs=2)

        assert len(history["epoch"]) == 2

    def test_metrics_consistency(self, synthetic_dataset_dir):
        """验证评估指标模块与评估器的一致性。"""
        tmpdir, noisy_dir, clean_dir = synthetic_dataset_dir

        val_transform = DataTransforms.get_val_transforms(target_size=32)
        dataset = DenoisingDataset(noisy_dir, clean_dir, transform=val_transform)

        # 直接使用Metrics计算
        noisy, clean = dataset[0]
        psnr = Metrics.calculate_psnr(noisy, clean)
        ssim = Metrics.calculate_ssim(noisy, clean)
        mse = Metrics.calculate_mse(noisy, clean)

        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert isinstance(mse, float)
        assert mse >= 0.0
        assert 0.0 <= ssim <= 1.0

        # 相同图像的指标
        psnr_same = Metrics.calculate_psnr(clean, clean)
        ssim_same = Metrics.calculate_ssim(clean, clean)
        mse_same = Metrics.calculate_mse(clean, clean)

        assert psnr_same == 100.0  # 近似无穷大
        assert ssim_same >= 0.99
        assert mse_same < 1e-10
