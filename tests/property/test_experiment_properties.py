"""属性测试 - 实验管理模块

Feature: image-denoising-comparison
Properties 22, 23, 24: 实验ID唯一性、实验输出完整性、实验查询一致性
"""

import os
import tempfile
import time

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from experiments.experiment_manager import ExperimentManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _experiment_name():
    """Generate valid experiment names."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    )


def _experiment_config():
    """Generate a plausible experiment config dict."""
    return st.fixed_dictionaries({
        "model": st.fixed_dictionaries({
            "name": st.sampled_from(["dncnn", "unet", "rednet", "ffdnet", "restormer", "sunet"]),
        }),
        "training": st.fixed_dictionaries({
            "learning_rate": st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
            "num_epochs": st.integers(min_value=1, max_value=500),
        }),
    })


def _experiment_results():
    """Generate plausible experiment results."""
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


# ---------------------------------------------------------------------------
# Property 22: 实验ID唯一性
# ---------------------------------------------------------------------------

class TestProperty22ExperimentIDUniqueness:
    """Feature: image-denoising-comparison, Property 22: 实验ID唯一性

    对于任意一系列实验创建操作，每个实验应该获得唯一的实验ID，
    且对应的输出目录应该被成功创建。

    Validates: Requirements 7.1
    """

    @given(
        names=st.lists(_experiment_name(), min_size=2, max_size=5),
        config=_experiment_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_all_experiment_ids_unique(self, names, config):
        """**Validates: Requirements 7.1**

        Creating multiple experiments should yield unique IDs and directories.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            ids = []
            for name in names:
                exp_id = mgr.create_experiment(name, config)
                ids.append(exp_id)

            # All IDs must be unique
            assert len(ids) == len(set(ids)), (
                f"Duplicate experiment IDs found: {ids}"
            )

            # Each experiment directory must exist
            for exp_id in ids:
                exp_dir = os.path.join(tmpdir, exp_id)
                assert os.path.isdir(exp_dir), (
                    f"Experiment directory not created: {exp_dir}"
                )

    @given(name=_experiment_name(), config=_experiment_config())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_experiment_directory_contains_metadata(self, name, config):
        """**Validates: Requirements 7.1**

        Each created experiment directory must contain a metadata file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_id = mgr.create_experiment(name, config)
            exp_dir = os.path.join(tmpdir, exp_id)

            meta_path = os.path.join(exp_dir, ExperimentManager.METADATA_FILE)
            assert os.path.exists(meta_path), "Metadata file not created"

            config_path = os.path.join(exp_dir, ExperimentManager.CONFIG_FILE)
            assert os.path.exists(config_path), "Config copy not created"


# ---------------------------------------------------------------------------
# Property 23: 实验输出完整性
# ---------------------------------------------------------------------------

class TestProperty23ExperimentOutputCompleteness:
    """Feature: image-denoising-comparison, Property 23: 实验输出完整性

    对于任意完成的实验，实验目录应该包含所有必需的文件
    （配置副本、元数据、评估结果、总结报告）。

    Validates: Requirements 7.2, 7.3, 7.4
    """

    @given(
        name=_experiment_name(),
        config=_experiment_config(),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_completed_experiment_has_all_files(self, name, config, results):
        """**Validates: Requirements 7.2, 7.3, 7.4**

        After saving results, the experiment directory must contain
        metadata, config, results, and summary files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_id = mgr.create_experiment(name, config)
            mgr.save_results(exp_id, results)

            exp_dir = os.path.join(tmpdir, exp_id)
            required_files = [
                ExperimentManager.METADATA_FILE,
                ExperimentManager.CONFIG_FILE,
                ExperimentManager.RESULTS_FILE,
                ExperimentManager.SUMMARY_FILE,
            ]
            for fname in required_files:
                fpath = os.path.join(exp_dir, fname)
                assert os.path.exists(fpath), f"Missing required file: {fname}"

    @given(
        name=_experiment_name(),
        config=_experiment_config(),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_metadata_contains_hardware_and_timestamps(self, name, config, results):
        """**Validates: Requirements 7.3**

        Metadata must include hardware info, creation time, and config snapshot.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_id = mgr.create_experiment(name, config)
            mgr.save_results(exp_id, results)

            metadata = mgr._load_metadata(exp_id)
            assert "hardware" in metadata, "Missing hardware info"
            assert "created_at" in metadata, "Missing created_at"
            assert "completed_at" in metadata, "Missing completed_at"
            assert "config_snapshot" in metadata, "Missing config snapshot"
            assert metadata["status"] == "completed"

    @given(
        name=_experiment_name(),
        config=_experiment_config(),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_results_roundtrip(self, name, config, results):
        """**Validates: Requirements 7.2**

        Saved results should be loadable and match the original data.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            exp_id = mgr.create_experiment(name, config)
            mgr.save_results(exp_id, results)

            loaded = mgr.load_results(exp_id)
            assert loaded == results, "Loaded results differ from saved results"


# ---------------------------------------------------------------------------
# Property 24: 实验查询一致性
# ---------------------------------------------------------------------------

class TestProperty24ExperimentQueryConsistency:
    """Feature: image-denoising-comparison, Property 24: 实验查询一致性

    对于任意已创建的实验集合，查询历史实验应该返回所有已创建的实验，
    且每个实验的元数据应该与创建时保存的一致。

    Validates: Requirements 7.5
    """

    @given(
        names=st.lists(_experiment_name(), min_size=1, max_size=5),
        config=_experiment_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_list_returns_all_created_experiments(self, names, config):
        """**Validates: Requirements 7.5**

        list_experiments should return all created experiments.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            created_ids = set()
            for name in names:
                exp_id = mgr.create_experiment(name, config)
                created_ids.add(exp_id)

            listed = mgr.list_experiments()
            listed_ids = {e["exp_id"] for e in listed}

            assert created_ids == listed_ids, (
                f"Mismatch: created={created_ids}, listed={listed_ids}"
            )

    @given(
        names=st.lists(_experiment_name(), min_size=1, max_size=4),
        config=_experiment_config(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_listed_metadata_matches_creation(self, names, config):
        """**Validates: Requirements 7.5**

        Each listed experiment's metadata should match what was set at creation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            created = {}
            for name in names:
                exp_id = mgr.create_experiment(name, config)
                created[exp_id] = name

            listed = mgr.list_experiments()
            for entry in listed:
                exp_id = entry["exp_id"]
                assert exp_id in created, f"Unknown experiment: {exp_id}"
                assert entry["name"] == created[exp_id], (
                    f"Name mismatch for {exp_id}: {entry['name']} != {created[exp_id]}"
                )
                assert entry["config_snapshot"] == config, (
                    f"Config mismatch for {exp_id}"
                )

    @given(
        names=st.lists(_experiment_name(), min_size=2, max_size=4),
        config=_experiment_config(),
        results=_experiment_results(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_compare_experiments_returns_all(self, names, config, results):
        """**Validates: Requirements 7.5**

        compare_experiments should return a DataFrame with a row per experiment.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            ids = []
            for name in names:
                exp_id = mgr.create_experiment(name, config)
                mgr.save_results(exp_id, results)
                ids.append(exp_id)

            df = mgr.compare_experiments(ids)
            assert len(df) == len(ids), (
                f"Expected {len(ids)} rows, got {len(df)}"
            )
            assert set(df["exp_id"].tolist()) == set(ids)
