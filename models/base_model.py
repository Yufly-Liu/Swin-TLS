"""模型基类 - 所有去噪模型的抽象基类。"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer, Adam


class BaseModel(nn.Module, ABC):
    """所有去噪模型的抽象基类。

    子类必须实现:
        - forward(x): 前向传播
        - get_loss_function(): 返回该模型推荐的损失函数
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化模型。

        Args:
            config: 模型配置字典，包含超参数（如 in_channels, out_channels 等）。
        """
        super().__init__()
        self.config = config or {}
        self.in_channels = self.config.get("in_channels", 3)
        self.out_channels = self.config.get("out_channels", 3)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        Args:
            x: 输入噪声图像张量，形状 (B, C, H, W)

        Returns:
            去噪后的图像张量，形状与输入相同 (B, C, H, W)
        """
        ...

    @abstractmethod
    def get_loss_function(self) -> nn.Module:
        """返回该模型官方推荐的损失函数。

        Returns:
            PyTorch 损失函数模块
        """
        ...

    def get_optimizer(self, lr: float = 1e-3) -> Optimizer:
        """返回优化器（默认 Adam）。

        Args:
            lr: 学习率

        Returns:
            PyTorch 优化器
        """
        return Adam(self.parameters(), lr=lr)

    def count_parameters(self) -> int:
        """返回模型可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
