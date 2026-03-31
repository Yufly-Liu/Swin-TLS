"""ConfigManager 单元测试。"""

import json
import os
import tempfile

import pytest
import yaml

from utils.config import ConfigError, ConfigManager


@pytest.fixture
def valid_config():
    """返回一个有效的配置字典。"""
    return ConfigManager.get_default_config()


@pytest.fixture
def tmp_dir():
    """创建临时目录。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestLoadConfig:
    def test_load_yaml(self, valid_config, tmp_dir):
        path = os.path.join(tmp_dir, "config.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(valid_config, f)
        loaded = ConfigManager.load_config(path)
        assert loaded == valid_config

    def test_load_json(self, valid_config, tmp_dir):
        path = os.path.join(tmp_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(valid_config, f)
        loaded = ConfigManager.load_config(path)
        assert loaded == valid_config

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config("nonexistent.yaml")

    def test_load_unsupported_format(self, tmp_dir):
        path = os.path.join(tmp_dir, "config.txt")
        with open(path, "w") as f:
            f.write("hello")
        with pytest.raises(ConfigError, match="不支持的配置文件格式"):
            ConfigManager.load_config(path)

    def test_load_invalid_yaml(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.yaml")
        with open(path, "w") as f:
            f.write(":\n  :\n    - :\n  bad: [")
        with pytest.raises(ConfigError, match="YAML解析错误"):
            ConfigManager.load_config(path)

    def test_load_empty_yaml(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.yaml")
        with open(path, "w") as f:
            f.write("")
        with pytest.raises(ConfigError, match="配置文件为空"):
            ConfigManager.load_config(path)


class TestValidateConfig:
    def test_valid_config(self, valid_config):
        assert ConfigManager.validate_config(valid_config) is True

    def test_missing_top_level_section(self, valid_config):
        del valid_config["dataset"]
        with pytest.raises(ConfigError, match="缺少必需参数.*dataset"):
            ConfigManager.validate_config(valid_config)

    def test_missing_nested_key(self, valid_config):
        del valid_config["dataset"]["noisy_dir"]
        with pytest.raises(ConfigError, match="dataset.noisy_dir"):
            ConfigManager.validate_config(valid_config)

    def test_invalid_split_sum(self, valid_config):
        valid_config["dataset"]["train_split"] = 0.5
        valid_config["dataset"]["val_split"] = 0.5
        valid_config["dataset"]["test_split"] = 0.5
        with pytest.raises(ConfigError, match="划分比例之和必须为1.0"):
            ConfigManager.validate_config(valid_config)

    def test_negative_split(self, valid_config):
        valid_config["dataset"]["train_split"] = -0.1
        with pytest.raises(ConfigError, match="不能为负数"):
            ConfigManager.validate_config(valid_config)

    def test_non_dict_config(self):
        with pytest.raises(ConfigError, match="字典类型"):
            ConfigManager.validate_config("not a dict")


class TestSaveConfig:
    def test_save_yaml(self, valid_config, tmp_dir):
        path = os.path.join(tmp_dir, "out.yaml")
        ConfigManager.save_config(valid_config, path)
        assert os.path.exists(path)
        loaded = ConfigManager.load_config(path)
        assert loaded == valid_config

    def test_save_json(self, valid_config, tmp_dir):
        path = os.path.join(tmp_dir, "out.json")
        ConfigManager.save_config(valid_config, path)
        assert os.path.exists(path)
        loaded = ConfigManager.load_config(path)
        assert loaded == valid_config

    def test_save_unsupported_format(self, valid_config, tmp_dir):
        path = os.path.join(tmp_dir, "out.txt")
        with pytest.raises(ConfigError, match="不支持的配置文件格式"):
            ConfigManager.save_config(valid_config, path)

    def test_save_creates_directories(self, valid_config, tmp_dir):
        path = os.path.join(tmp_dir, "sub", "dir", "config.yaml")
        ConfigManager.save_config(valid_config, path)
        assert os.path.exists(path)
