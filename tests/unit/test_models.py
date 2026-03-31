"""模型架构单元测试 - 验证每个模型的前向传播和基本功能。"""

import pytest
import torch

from models import get_model, MODEL_REGISTRY
from models.base_model import BaseModel
from models.dncnn import DnCNN
from models.unet import UNet
from models.rednet import REDNet
from models.ffdnet import FFDNet
from models.restormer import Restormer
from models.sunet import SUNet


# 轻量配置，加速测试
LIGHTWEIGHT_CONFIGS = {
    "dncnn": {"in_channels": 3, "out_channels": 3, "num_layers": 5, "num_features": 16},
    "unet": {"in_channels": 3, "out_channels": 3, "base_features": 16, "depth": 2},
    "rednet": {"in_channels": 3, "out_channels": 3, "num_layers": 5, "num_features": 16},
    "ffdnet": {"in_channels": 3, "out_channels": 3, "num_layers": 5, "num_features": 16},
    "restormer": {"in_channels": 3, "out_channels": 3, "dim": 12, "num_blocks": [1, 1, 1, 1], "num_heads": [1, 2, 4, 4]},
    "sunet": {"in_channels": 3, "out_channels": 3, "embed_dim": 12, "depths": [1, 1, 1, 1], "num_heads": [3, 6, 12, 12], "window_size": 4, "patch_size": 4},
}

ALL_MODEL_NAMES = list(LIGHTWEIGHT_CONFIGS.keys())


class TestModelForwardPass:
    """验证每个模型的前向传播输出形状正确。"""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_forward_shape_64x64(self, model_name):
        config = LIGHTWEIGHT_CONFIGS[model_name]
        model = get_model(model_name, config)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, f"{model_name}: {out.shape} != {x.shape}"

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_forward_shape_128x128(self, model_name):
        config = LIGHTWEIGHT_CONFIGS[model_name]
        model = get_model(model_name, config)
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, f"{model_name}: {out.shape} != {x.shape}"

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_forward_batch_size_2(self, model_name):
        config = LIGHTWEIGHT_CONFIGS[model_name]
        model = get_model(model_name, config)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestModelLossFunction:
    """验证每个模型返回正确的损失函数。"""

    @pytest.mark.parametrize("model_name,expected_loss", [
        ("dncnn", torch.nn.MSELoss),
        ("unet", torch.nn.MSELoss),
        ("rednet", torch.nn.MSELoss),
        ("ffdnet", torch.nn.L1Loss),
        ("sunet", torch.nn.L1Loss),
    ])
    def test_loss_function_type(self, model_name, expected_loss):
        model = get_model(model_name, LIGHTWEIGHT_CONFIGS[model_name])
        loss_fn = model.get_loss_function()
        assert isinstance(loss_fn, expected_loss)

    def test_restormer_charbonnier_loss(self):
        from models.restormer import CharbonnierLoss
        model = get_model("restormer", LIGHTWEIGHT_CONFIGS["restormer"])
        loss_fn = model.get_loss_function()
        assert isinstance(loss_fn, CharbonnierLoss)


class TestModelRegistry:
    """验证模型注册表功能。"""

    def test_all_six_models_registered(self):
        assert len(MODEL_REGISTRY) == 6
        for name in ALL_MODEL_NAMES:
            assert name in MODEL_REGISTRY

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="未知的模型类型"):
            get_model("nonexistent")

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_count_parameters_positive(self, model_name):
        model = get_model(model_name, LIGHTWEIGHT_CONFIGS[model_name])
        assert model.count_parameters() > 0
