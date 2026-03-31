"""属性测试 - 配置管理模块

Feature: image-denoising-comparison
Properties 20, 21: 配置验证完整性 & 配置文件往返一致性
"""

import os
import tempfile

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.config import ConfigManager, ConfigError, REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Strategies (smart generators)
# ---------------------------------------------------------------------------

# Strategy: generate a valid split triple that sums to 1.0
def _valid_splits():
    """Generate (train, val, test) floats >= 0 that sum to 1.0."""
    return (
        st.tuples(
            st.floats(min_value=0.01, max_value=0.98),
            st.floats(min_value=0.01, max_value=0.98),
        )
        .filter(lambda t: t[0] + t[1] < 1.0)
        .map(lambda t: (round(t[0], 4), round(t[1], 4), round(1.0 - t[0] - t[1], 4)))
        .filter(lambda t: t[2] > 0)
    )


MODEL_NAMES = ["dncnn", "unet", "rednet", "ffdnet", "restormer", "sunet"]


def _valid_config_strategy():
    """Generate an arbitrary valid configuration dictionary."""
    return _valid_splits().flatmap(lambda splits: st.fixed_dictionaries({
        "dataset": st.fixed_dictionaries({
            "noisy_dir": st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-/"),
                min_size=1, max_size=30,
            ).map(lambda s: "./" + s),
            "clean_dir": st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-/"),
                min_size=1, max_size=30,
            ).map(lambda s: "./" + s),
            "train_split": st.just(splits[0]),
            "val_split": st.just(splits[1]),
            "test_split": st.just(splits[2]),
            "batch_size": st.integers(min_value=1, max_value=128),
        }),
        "model": st.fixed_dictionaries({
            "name": st.sampled_from(MODEL_NAMES),
        }),
        "training": st.fixed_dictionaries({
            "num_epochs": st.integers(min_value=1, max_value=500),
            "learning_rate": st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
            "checkpoint_dir": st.just("./checkpoints"),
        }),
        "evaluation": st.fixed_dictionaries({
            "metrics": st.just(["psnr", "ssim", "mse"]),
        }),
        "experiment": st.fixed_dictionaries({
            "name": st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
                min_size=1, max_size=20,
            ),
            "output_dir": st.just("./experiments"),
        }),
    }))


# ---------------------------------------------------------------------------
# Property 20: 配置验证完整性
# ---------------------------------------------------------------------------

class TestProperty20ConfigValidationCompleteness:
    """Feature: image-denoising-comparison, Property 20: 配置验证完整性

    对于任意缺少必需参数的配置文件，验证函数应该返回False并指出缺失的参数。

    Validates: Requirements 6.2, 6.3
    """

    @given(
        config=_valid_config_strategy(),
        section_to_remove=st.sampled_from(list(REQUIRED_KEYS.keys())),
    )
    @settings(max_examples=100)
    def test_missing_top_level_section_detected(self, config, section_to_remove):
        """**Validates: Requirements 6.2, 6.3**

        Removing any required top-level section should cause validate_config
        to raise ConfigError mentioning the missing section.
        """
        config = dict(config)  # shallow copy
        del config[section_to_remove]

        with pytest.raises(ConfigError) as exc_info:
            ConfigManager.validate_config(config)

        assert section_to_remove in str(exc_info.value)

    @given(data=st.data(), config=_valid_config_strategy())
    @settings(max_examples=100)
    def test_missing_sub_key_detected(self, data, config):
        """**Validates: Requirements 6.2, 6.3**

        Removing any required sub-key from a section should cause
        validate_config to raise ConfigError mentioning the missing key.
        """
        import copy
        config = copy.deepcopy(config)

        # Pick a random section that has sub-keys
        section = data.draw(st.sampled_from(list(REQUIRED_KEYS.keys())))
        sub_keys = REQUIRED_KEYS[section]
        key_to_remove = data.draw(st.sampled_from(sub_keys))

        del config[section][key_to_remove]

        with pytest.raises(ConfigError) as exc_info:
            ConfigManager.validate_config(config)

        error_msg = str(exc_info.value)
        assert key_to_remove in error_msg

    @given(config=_valid_config_strategy())
    @settings(max_examples=100)
    def test_valid_config_passes_validation(self, config):
        """**Validates: Requirements 6.2**

        A fully valid config should pass validation (returns True).
        """
        assert ConfigManager.validate_config(config) is True


# ---------------------------------------------------------------------------
# Property 21: 配置文件往返一致性
# ---------------------------------------------------------------------------

class TestProperty21ConfigRoundTrip:
    """Feature: image-denoising-comparison, Property 21: 配置文件往返一致性

    对于任意有效的配置字典，保存为文件后再加载，配置内容应该保持一致。

    Validates: Requirements 6.1, 6.5
    """

    @given(config=_valid_config_strategy(), ext=st.sampled_from([".yaml", ".json"]))
    @settings(max_examples=100)
    def test_save_then_load_roundtrip(self, config, ext):
        """**Validates: Requirements 6.1, 6.5**

        Saving a valid config and loading it back should yield identical content.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"config{ext}")
            ConfigManager.save_config(config, path)
            loaded = ConfigManager.load_config(path)

        assert loaded == config

    @given(config=_valid_config_strategy())
    @settings(max_examples=100)
    def test_roundtrip_preserves_validation(self, config):
        """**Validates: Requirements 6.1, 6.5**

        A config that passes validation before saving should also pass
        validation after a save/load round-trip.
        """
        assert ConfigManager.validate_config(config) is True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            ConfigManager.save_config(config, path)
            loaded = ConfigManager.load_config(path)

        assert ConfigManager.validate_config(loaded) is True
