"""U-Net - Convolutional Networks for Biomedical Image Segmentation (adapted for denoising).

参考: Ronneberger et al., 2015
官方架构: 编码器-解码器结构，带跳跃连接。
"""

from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from models.base_model import BaseModel


class DoubleConv(nn.Module):
    """两个连续的 Conv-BN-ReLU 块。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet(BaseModel):
    """U-Net 去噪网络。

    编码器-解码器结构，4级下采样/上采样，跳跃连接。

    Config keys:
        in_channels (int): 输入通道数，默认 3
        out_channels (int): 输出通道数，默认 3
        base_features (int): 第一层特征数，默认 64
        depth (int): 编码器深度（下采样次数），默认 4
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.base_features = self.config.get("base_features", 64)
        self.depth = self.config.get("depth", 4)

        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = self.in_channels
        features = self.base_features
        for _ in range(self.depth):
            self.encoders.append(DoubleConv(in_ch, features))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = features
            features *= 2

        # 瓶颈层
        self.bottleneck = DoubleConv(in_ch, features)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(self.depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(features, features // 2))
            features //= 2

        # 输出层
        self.out_conv = nn.Conv2d(features, self.out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # 保存输入用于残差连接
        residual = x

        # 编码器路径
        skip_connections = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # 解码器路径
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skip_connections)):
            x = upconv(x)
            # 处理尺寸不匹配
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.out_conv(x)
        return x + residual

    def get_loss_function(self) -> nn.Module:
        """U-Net 使用 MSE 损失函数。"""
        return nn.MSELoss()
