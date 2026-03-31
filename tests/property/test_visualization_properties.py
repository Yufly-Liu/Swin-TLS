"""属性测试 - 可视化和对比模块

Feature: image-denoising-comparison
Properties 17, 18, 19: 对比表格完整性、对比数据序列化往返一致性、可视化输出文件存在性
"""

import json
import os
import tempfile

import pandas as pd
import pytest
import torch
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from experiments.experiment_manager import ExperimentManager
from utils.visualization import Visualizer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _model_name():
    return st.sampled_from(["dncnn", "unet", "rednet", "ffdnet", "restormer", "sunet"])


def _positive_float(min_val=0.0, max_val=1000.0):
    return st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False)


def _experiment_results():
    """Generate plausible experiment results with all required fields."""
    return st.fixed_dictionaries({
        "test_metrics": st.fixed_dictionaries({
            "psnr": st.fixed_dictionaries({
                "mean": st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
                "std": st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            }),
            "ssim": st.fixed_dictionaries({
                "mean": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                "std": st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            }),
            "mse": st.fixed_dictionaries({
                "mean": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                "std": st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
            }),
        }),
        "training_time": st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False),
        "inference_time": st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        "model_params": st.integers(min_value=1000, max_value=100_000_000),
    })


def _experiment_config(model_name_st=None):
    """Generate experiment config with a specific model name strategy."""
    if model_name_st is None:
        model_name_st = _model_name()
    return st.builds(
        lambda name: {
            "model": {"name": name},
            "training": {"learning_rate": 0.001, "num_epochs": 10},
        },
        name=model_name_st,
    )


def _training_history():
    """Generate plausible training history dicts."""
    return st.integers(min_value=2, max_value=10).flatmap(
        lambda n: st.fixed_dictionaries({
            "epoch": st.just(list(range(1, n + 1))),
            "train_loss": st.lists(
                st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            ),
            "val_psnr": st.lists(
                st.floats(min_value=10.0, max_value=50.0, allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            ),
            "val_ssim": st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            ),
            "learning_rate": st.lists(
                st.floats(min_value=1e-6, max_value=0.01, allow_nan=False, allow_infinity=False),
                min_size=n, max_size=n,
            ),
        })
    )


# ---------------------------------------------------------------------------
# Property 17: 对比表格完整性
# ---------------------------------------------------------------------------

class TestProperty17ComparisonTableCompleteness:
    """Feature: image-denoising-comparison, Property 17: 对比表格完整性

    对于任意一组模型评估结果，生成的对比表格应该包含所有模型的所有指标
    （PSNR、SSIM、MSE、参数量、训练时间、推理时间）。

    Validates: Requirements 5.1, 5.3
    """

    @given(
        model_names=st.lists(_model_name(), min_size=2, max_size=4, unique=True),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_comparison_table_has_all_models_and_metrics(self, model_names, results):
        """**Validates: Requirements 5.1, 5.3**

        The comparison DataFrame should have one row per model and columns
        for all required metrics.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_ids = []
            for name in model_names:
                config = {"model": {"name": name}, "training": {"learning_rate": 0.001, "num_epochs": 10}}
                exp_id = mgr.create_experiment(name, config)
                mgr.save_results(exp_id, results)
                exp_ids.append(exp_id)

            df = mgr.compare_experiments(exp_ids)

            # Should have one row per experiment
            assert len(df) == len(model_names), (
                f"Expected {len(model_names)} rows, got {len(df)}"
            )

            # Required metric columns
            required_cols = [
                "psnr_mean", "psnr_std",
                "ssim_mean", "ssim_std",
                "mse_mean", "mse_std",
                "training_time", "inference_time", "model_params",
            ]
            for col in required_cols:
                assert col in df.columns, f"Missing column: {col}"

            # All values should be non-null
            for col in required_cols:
                assert df[col].notna().all(), f"Column {col} has null values"


# ---------------------------------------------------------------------------
# Property 18: 对比数据序列化往返一致性
# ---------------------------------------------------------------------------

class TestProperty18ComparisonSerializationRoundtrip:
    """Feature: image-denoising-comparison, Property 18: 对比数据序列化往返一致性

    对于任意对比报告数据，导出为JSON/CSV后再导入，数据内容应该保持一致
    （数值精度在可接受范围内）。

    Validates: Requirements 5.5
    """

    @given(
        model_names=st.lists(_model_name(), min_size=2, max_size=4, unique=True),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_json_roundtrip(self, model_names, results):
        """**Validates: Requirements 5.5**

        Exporting comparison data to JSON and re-importing should preserve values.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_ids = []
            for name in model_names:
                config = {"model": {"name": name}, "training": {"learning_rate": 0.001, "num_epochs": 10}}
                exp_id = mgr.create_experiment(name, config)
                mgr.save_results(exp_id, results)
                exp_ids.append(exp_id)

            df = mgr.compare_experiments(exp_ids)

            # Export to JSON
            json_path = os.path.join(tmpdir, "comparison.json")
            df.to_json(json_path, orient="records", indent=2)

            # Re-import
            loaded_df = pd.read_json(json_path, orient="records")

            # Check shape
            assert loaded_df.shape == df.shape, (
                f"Shape mismatch: {loaded_df.shape} vs {df.shape}"
            )

            # Check numeric columns are close
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                orig = df[col].values
                loaded = loaded_df[col].values
                assert all(
                    abs(a - b) < 1e-6 if a is not None and b is not None else a == b
                    for a, b in zip(orig, loaded)
                ), f"Values differ in column {col}"

    @given(
        model_names=st.lists(_model_name(), min_size=2, max_size=4, unique=True),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_csv_roundtrip(self, model_names, results):
        """**Validates: Requirements 5.5**

        Exporting comparison data to CSV and re-importing should preserve values.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_ids = []
            for name in model_names:
                config = {"model": {"name": name}, "training": {"learning_rate": 0.001, "num_epochs": 10}}
                exp_id = mgr.create_experiment(name, config)
                mgr.save_results(exp_id, results)
                exp_ids.append(exp_id)

            df = mgr.compare_experiments(exp_ids)

            # Export to CSV
            csv_path = os.path.join(tmpdir, "comparison.csv")
            df.to_csv(csv_path, index=False)

            # Re-import
            loaded_df = pd.read_csv(csv_path)

            # Check shape
            assert loaded_df.shape == df.shape, (
                f"Shape mismatch: {loaded_df.shape} vs {df.shape}"
            )

            # Check numeric columns are close
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                orig = df[col].values
                loaded = loaded_df[col].values
                for a, b in zip(orig, loaded):
                    if pd.notna(a) and pd.notna(b):
                        assert abs(a - b) < 1e-4, (
                            f"Values differ in column {col}: {a} vs {b}"
                        )


# ---------------------------------------------------------------------------
# Property 19: 可视化输出文件存在性
# ---------------------------------------------------------------------------

class TestProperty19VisualizationOutputFileExistence:
    """Feature: image-denoising-comparison, Property 19: 可视化输出文件存在性

    对于任意对比过程，应该生成所有必需的可视化文件
    （对比柱状图、训练曲线图、去噪结果对比图）。

    Validates: Requirements 5.2, 5.4, 8.1, 8.2, 8.3, 8.4
    """

    @given(history=_training_history())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_training_curves_file_created(self, history):
        """**Validates: Requirements 8.1, 8.2**

        plot_training_curves should create an image file at the specified path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "training_curves.png")
            Visualizer.plot_training_curves(history, save_path)

            assert os.path.exists(save_path), (
                f"Training curves file not created: {save_path}"
            )
            assert os.path.getsize(save_path) > 0, "Training curves file is empty"

    @given(
        model_names=st.lists(_model_name(), min_size=2, max_size=4, unique=True),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_comparison_bar_file_created(self, model_names, results):
        """**Validates: Requirements 5.2, 8.2**

        plot_comparison_bar should create an image file at the specified path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_ids = []
            for name in model_names:
                config = {"model": {"name": name}, "training": {"learning_rate": 0.001, "num_epochs": 10}}
                exp_id = mgr.create_experiment(name, config)
                mgr.save_results(exp_id, results)
                exp_ids.append(exp_id)

            df = mgr.compare_experiments(exp_ids)
            save_path = os.path.join(tmpdir, "comparison_bar.png")
            Visualizer.plot_comparison_bar(df, save_path)

            assert os.path.exists(save_path), (
                f"Comparison bar chart not created: {save_path}"
            )
            assert os.path.getsize(save_path) > 0, "Comparison bar chart file is empty"

    @given(
        h=st.integers(min_value=16, max_value=64),
        w=st.integers(min_value=16, max_value=64),
        channels=st.sampled_from([1, 3]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_image_comparison_file_created(self, h, w, channels):
        """**Validates: Requirements 5.4, 8.3, 8.4**

        save_image_comparison should create an image file at the specified path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            if channels == 1:
                noisy = torch.rand(h, w)
                denoised = torch.rand(h, w)
                clean = torch.rand(h, w)
            else:
                noisy = torch.rand(channels, h, w)
                denoised = torch.rand(channels, h, w)
                clean = torch.rand(channels, h, w)

            save_path = os.path.join(tmpdir, "comparison.png")
            Visualizer.save_image_comparison(noisy, denoised, clean, save_path)

            assert os.path.exists(save_path), (
                f"Image comparison file not created: {save_path}"
            )
            assert os.path.getsize(save_path) > 0, "Image comparison file is empty"
