"""属性测试 - 模型架构模块

Feature: image-denoising-comparison
Properties 5-7: 模型接口一致性、输入输出形状一致性、超参数可配置性
"""

import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from models.base_model import BaseModel
from models.dncnn import DnCNN
from models.unet import UNet
from models.rednet import REDNet
from models.ffdnet import FFDNet
from models.restormer import Restormer
from models.sunet import SUNet
from models import get_model, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 所有模型类列表（用于参数化）
ALL_MODEL_CLASSES = [DnCNN, UNet, REDNet, FFDNet, Restormer, SUNet]
ALL_MODEL_NAMES = ["dncnn", "unet", "rednet", "ffdnet", "restormer", "sunet"]

# SUNet 需要输入尺寸是 patch_size * 2^depth 的倍数，使用较大的固定尺寸
# 为了测试效率，使用较小的模型配置
LIGHTWEIGHT_CONFIGS = {
    "dncnn": {"in_channels": 3, "out_channels": 3, "num_layers": 5, "num_features": 16},
    "unet": {"in_channels": 3, "out_channels": 3, "base_features": 16, "depth": 2},
    "rednet": {"in_channels": 3, "out_channels": 3, "num_layers": 5, "num_features": 16},
    "ffdnet": {"in_channels": 3, "out_channels": 3, "num_layers": 5, "num_features": 16},
    "restormer": {"in_channels": 3, "out_channels": 3, "dim": 12, "num_blocks": [1, 1, 1, 1], "num_heads": [1, 2, 4, 4]},
    "sunet": {"in_channels": 3, "out_channels": 3, "embed_dim": 12, "depths": [1, 1, 1, 1], "num_heads": [3, 6, 12, 12], "window_size": 4, "patch_size": 4},
}


# ---------------------------------------------------------------------------
# Property 5: 模型接口一致性
# ---------------------------------------------------------------------------

class TestProperty5ModelInterfaceConsistency:
    """Feature: image-denoising-comparison, Property 5: 模型接口一致性

    对于所有实现的模型（DnCNN、U-Net、RED-Net、FFDNet、Restormer、SUNet），
    每个模型都应该实现BaseModel接口的所有必需方法
    （forward、get_loss_function、get_optimizer）。

    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
    """

    @given(model_name=st.sampled_from(ALL_MODEL_NAMES))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_all_models_implement_base_interface(self, model_name):
        """**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

        Every model must be a subclass of BaseModel and implement
        forward, get_loss_function, and get_optimizer.
        """
        config = LIGHTWEIGHT_CONFIGS[model_name]
        model = get_model(model_name, config)

        # Must be a BaseModel subclass
        assert isinstance(model, BaseModel), f"{model_name} is not a BaseModel subclass"

        # Must have required methods
        assert callable(getattr(model, "forward", None)), f"{model_name} missing forward"
        assert callable(getattr(model, "get_loss_function", None)), f"{model_name} missing get_loss_function"
        assert callable(getattr(model, "get_optimizer", None)), f"{model_name} missing get_optimizer"

        # get_loss_function must return an nn.Module
        loss_fn = model.get_loss_function()
        assert isinstance(loss_fn, torch.nn.Module), f"{model_name} loss_fn is not nn.Module"

        # get_optimizer must return an Optimizer
        optimizer = model.get_optimizer(lr=1e-3)
        assert isinstance(optimizer, torch.optim.Optimizer), f"{model_name} optimizer is not Optimizer"


# ---------------------------------------------------------------------------
# Property 6: 模型输入输出形状一致性
# ---------------------------------------------------------------------------

class TestProperty6ModelIOShapeConsistency:
    """Feature: image-denoising-comparison, Property 6: 模型输入输出形状一致性

    对于任意模型和输入张量，模型的输出形状应该与输入形状相同（去噪任务的特性）。

    Validates: Requirements 2.1-2.6
    """

    @given(
        model_name=st.sampled_from(ALL_MODEL_NAMES),
        batch_size=st.integers(min_value=1, max_value=2),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_output_shape_matches_input(self, model_name, batch_size):
        """**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

        For any model, the output tensor shape must equal the input tensor shape.
        """
        config = LIGHTWEIGHT_CONFIGS[model_name]
        model = get_model(model_name, config)
        model.eval()

        # Use 64x64 which is divisible by all required factors
        H, W = 64, 64
        x = torch.randn(batch_size, 3, H, W)

        with torch.no_grad():
            out = model(x)

        assert out.shape == x.shape, (
            f"{model_name}: expected output shape {x.shape}, got {out.shape}"
        )


# ---------------------------------------------------------------------------
# Property 7: 模型超参数可配置性
# ---------------------------------------------------------------------------

class TestProperty7ModelHyperparameterConfigurability:
    """Feature: image-denoising-comparison, Property 7: 模型超参数可配置性

    对于任意支持的超参数配置，使用不同超参数初始化的模型应该具有不同的
    架构特征（如层数、通道数）。

    Validates: Requirements 2.8
    """

    @given(
        num_features=st.sampled_from([16, 32, 64]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_different_configs_produce_different_param_counts(self, num_features):
        """**Validates: Requirements 2.8**

        Models initialized with different hyperparameters (e.g. num_features)
        should have different parameter counts, proving the config is effective.
        """
        # Use DnCNN as representative: varying num_features changes param count
        config_small = {"num_features": 8, "num_layers": 5}
        config_varied = {"num_features": num_features, "num_layers": 5}

        model_small = DnCNN(config_small)
        model_varied = DnCNN(config_varied)

        params_small = model_small.count_parameters()
        params_varied = model_varied.count_parameters()

        if num_features != 8:
            assert params_small != params_varied, (
                f"Different num_features ({8} vs {num_features}) should yield "
                f"different param counts, but both have {params_small}"
            )
        else:
            assert params_small == params_varied
