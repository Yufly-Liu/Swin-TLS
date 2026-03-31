"""DnCNN - Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.

参考: Zhang et al., 2017
官方架构: 17层卷积网络，批归一化，ReLU激活，残差学习。
"""

from typing import Dict, Any

import torch.nn as nn
from torch import Tensor

from models.base_model import BaseModel


class DnCNN(BaseModel):
    """DnCNN 去噪网络。

    架构: Conv+ReLU → (Conv+BN+ReLU) × (depth-2) → Conv
    使用残差学习策略：网络学习噪声残差，输出 = 输入 - 残差。

    Config keys:
        in_channels (int): 输入通道数，默认 3
        out_channels (int): 输出通道数，默认 3
        num_layers (int): 网络层数，默认 17
        num_features (int): 中间层特征通道数，默认 64
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.num_layers = self.config.get("num_layers", 17)
        self.num_features = self.config.get("num_features", 64)

        layers = []
        # 第一层: Conv + ReLU (无BN)
        layers.append(nn.Conv2d(self.in_channels, self.num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间层: Conv + BN + ReLU
        for _ in range(self.num_layers - 2):
            layers.append(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(self.num_features))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层: Conv (无BN, 无ReLU)
        layers.append(nn.Conv2d(self.num_features, self.out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播 - 残差学习: output = input - noise_residual"""
        noise = self.dncnn(x)
        return x - noise

    def get_loss_function(self) -> nn.Module:
        """DnCNN 使用 MSE 损失函数。"""
        return nn.MSELoss()
