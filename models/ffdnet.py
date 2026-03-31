"""FFDNet - Toward a Fast and Flexible Solution for CNN-Based Image Denoising.

参考: Zhang et al., 2018
官方架构: 支持噪声级别输入的灵活去噪网络。
核心思想: 将噪声级别图作为额外输入通道，使网络能处理不同噪声水平。
"""

from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from models.base_model import BaseModel


class FFDNet(BaseModel):
    """FFDNet 去噪网络。

    架构特点:
    1. 像素重排（下采样）减小空间分辨率
    2. 噪声级别图作为额外输入通道
    3. DnCNN风格的卷积网络处理
    4. 像素重排（上采样）恢复分辨率

    Config keys:
        in_channels (int): 输入通道数，默认 3
        out_channels (int): 输出通道数，默认 3
        num_layers (int): 中间卷积层数，默认 15
        num_features (int): 中间层特征通道数，默认 64
        noise_level (float): 默认噪声级别 (0-1)，默认 0.1
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.num_layers = self.config.get("num_layers", 15)
        self.num_features = self.config.get("num_features", 64)
        self.noise_level = self.config.get("noise_level", 0.1)

        # 像素重排后通道数 × 4，加上噪声级别图通道
        reshuffled_channels = self.in_channels * 4 + 1

        layers = []
        # 第一层
        layers.append(nn.Conv2d(reshuffled_channels, self.num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间层
        for _ in range(self.num_layers - 2):
            layers.append(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(self.num_features))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层
        layers.append(nn.Conv2d(self.num_features, self.out_channels * 4, kernel_size=3, padding=1, bias=False))

        self.body = nn.Sequential(*layers)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor, noise_level: float = None) -> Tensor:
        """前向传播。

        Args:
            x: 输入图像 (B, C, H, W)，H和W应为偶数
            noise_level: 噪声级别，None则使用默认值
        """
        B, C, H, W = x.shape
        sigma = noise_level if noise_level is not None else self.noise_level

        # 像素重排下采样 (B, C, H, W) -> (B, 4C, H/2, W/2)
        # 手动实现 pixel unshuffle
        x_reshaped = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x_down = x_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)

        # 创建噪声级别图
        noise_map = torch.full((B, 1, H // 2, W // 2), sigma, device=x.device, dtype=x.dtype)

        # 拼接输入和噪声级别图
        x_in = torch.cat([x_down, noise_map], dim=1)

        # 通过网络
        out = self.body(x_in)

        # 像素重排上采样 (B, 4C, H/2, W/2) -> (B, C, H, W)
        out = self.pixel_shuffle(out)

        return out

    def get_loss_function(self) -> nn.Module:
        """FFDNet 使用 L1 损失函数。"""
        return nn.L1Loss()
