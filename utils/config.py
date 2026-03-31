"""配置管理模块 - 支持YAML/JSON格式的配置文件加载、验证和保存。"""

import json
import os
from typing import Any, Dict, List

import yaml


# 配置文件中必需的顶层键及其必需子键
REQUIRED_KEYS = {
    "dataset": ["noisy_dir", "clean_dir", "train_split", "val_split", "test_split", "batch_size"],
    "model": ["name"],
    "training": ["num_epochs", "learning_rate", "checkpoint_dir"],
    "evaluation": ["metrics"],
    "experiment": ["name", "output_dir"],
}


class ConfigError(Exception):
    """配置相关错误。"""
    pass


class ConfigManager:
    """管理配置文件的加载、验证和保存。"""

    @staticmethod
    def load_config(path: str) -> dict:
        """加载YAML或JSON配置文件。

        Args:
            path: 配置文件路径。

        Returns:
            解析后的配置字典。

        Raises:
            FileNotFoundError: 配置文件不存在。
            ConfigError: 配置文件格式错误。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")

        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "r", encoding="utf-8") as f:
                if ext in (".yaml", ".yml"):
                    config = yaml.safe_load(f)
                elif ext == ".json":
                    config = json.load(f)
                else:
                    raise ConfigError(f"不支持的配置文件格式: {ext}，请使用 .yaml/.yml 或 .json")
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML解析错误: {e}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"JSON解析错误: {e}")

        if config is None:
            raise ConfigError("配置文件为空")

        return config

    @staticmethod
    def validate_config(config: dict) -> bool:
        """验证配置完整性，检查所有必需参数是否存在。

        Args:
            config: 配置字典。

        Returns:
            True 如果配置有效。

        Raises:
            ConfigError: 缺少必需参数时抛出，包含缺失参数列表。
        """
        if not isinstance(config, dict):
            raise ConfigError("配置必须是字典类型")

        missing: List[str] = []

        # 检查顶层必需键
        for section in ["dataset", "model", "training", "evaluation", "experiment"]:
            if section not in config:
                missing.append(section)

        if missing:
            raise ConfigError(f"缺少必需参数: {', '.join(missing)}")

        # 检查 dataset 配置
        dataset_cfg = config.get("dataset", {})
        if not isinstance(dataset_cfg, dict):
            missing.append("dataset (应为字典类型)")
        else:
            # 多数据集模式 vs 单数据集模式
            if "datasets" in dataset_cfg:
                # 多数据集模式：检查 datasets 列表
                if not isinstance(dataset_cfg["datasets"], list):
                    missing.append("dataset.datasets (应为列表类型)")
                elif len(dataset_cfg["datasets"]) == 0:
                    missing.append("dataset.datasets (列表不能为空)")
            else:
                # 单数据集模式：检查 noisy_dir 和 clean_dir
                for key in ["noisy_dir", "clean_dir"]:
                    if key not in dataset_cfg:
                        missing.append(f"dataset.{key}")
            
            # 检查通用必需参数
            for key in ["train_split", "val_split", "test_split", "batch_size"]:
                if key not in dataset_cfg:
                    missing.append(f"dataset.{key}")

        # 检查其他部分
        for section, keys in [
            ("model", ["name"]),
            ("training", ["num_epochs", "learning_rate", "checkpoint_dir"]),
            ("evaluation", ["metrics"]),
            ("experiment", ["name", "output_dir"]),
        ]:
            section_cfg = config.get(section, {})
            if not isinstance(section_cfg, dict):
                missing.append(f"{section} (应为字典类型)")
                continue
            for key in keys:
                if key not in section_cfg:
                    missing.append(f"{section}.{key}")

        if missing:
            raise ConfigError(f"缺少必需参数: {', '.join(missing)}")

        # 验证数据集划分比例
        splits = [
            dataset_cfg.get("train_split", 0),
            dataset_cfg.get("val_split", 0),
            dataset_cfg.get("test_split", 0),
        ]
        if any(s < 0 for s in splits):
            raise ConfigError("数据集划分比例不能为负数")
        total = sum(splits)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"数据集划分比例之和必须为1.0，当前为 {total}")

        return True

    @staticmethod
    def save_config(config: dict, path: str) -> None:
        """保存配置字典到YAML或JSON文件。

        Args:
            config: 配置字典。
            path: 保存路径。

        Raises:
            ConfigError: 不支持的文件格式。
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        ext = os.path.splitext(path)[1].lower()
        with open(path, "w", encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif ext == ".json":
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ConfigError(f"不支持的配置文件格式: {ext}，请使用 .yaml/.yml 或 .json")

    @staticmethod
    def get_default_config() -> dict:
        """返回默认配置字典。"""
        return {
            "dataset": {
                "noisy_dir": "./data/noisy",
                "clean_dir": "./data/clean",
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
                "batch_size": 16,
                "num_workers": 4,
            },
            "model": {
                "name": "dncnn",
                "params": {
                    "in_channels": 3,
                    "out_channels": 3,
                },
            },
            "training": {
                "num_epochs": 100,
                "learning_rate": 0.001,
                "scheduler": {
                    "type": "ReduceLROnPlateau",
                    "patience": 10,
                    "factor": 0.5,
                },
                "early_stopping": {
                    "patience": 20,
                },
                "checkpoint_dir": "./checkpoints",
                "save_frequency": 5,
            },
            "evaluation": {
                "metrics": ["psnr", "ssim", "mse"],
                "save_samples": True,
                "num_samples": 10,
            },
            "experiment": {
                "name": "exp_001",
                "description": "Baseline DnCNN training",
                "output_dir": "./experiments",
            },
        }
