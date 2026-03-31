# 模型模块
from models.base_model import BaseModel
from models.dncnn import DnCNN
from models.unet import UNet
from models.rednet import REDNet
from models.ffdnet import FFDNet
from models.restormer import Restormer
from models.sunet import SUNet

# 模型注册表
MODEL_REGISTRY = {
    "dncnn": DnCNN,
    "unet": UNet,
    "rednet": REDNet,
    "ffdnet": FFDNet,
    "restormer": Restormer,
    "sunet": SUNet,
}


def get_model(name: str, config: dict = None) -> BaseModel:
    """根据名称获取模型实例。

    Args:
        name: 模型名称（dncnn, unet, rednet, ffdnet, restormer, sunet）
        config: 模型配置字典

    Returns:
        模型实例

    Raises:
        ValueError: 未知的模型名称
    """
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"未知的模型类型: {name}。支持的模型: {supported}")
    return MODEL_REGISTRY[name_lower](config)
