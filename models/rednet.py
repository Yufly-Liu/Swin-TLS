"""RED-Net - Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks
with Symmetric Skip Connections.

参考: Mao et al., 2016
官方架构: 对称的卷积-反卷积网络，带跳跃连接。
"""

from typing import Dict, Any

import torch.nn as nn
from torch import Tensor

from models.base_model import BaseModel


class REDNet(BaseModel):
    """RED-Net 去噪网络。

    对称的编码器（卷积）-解码器（反卷积）结构，
    编码器和解码器之间有对称的跳跃连接。

    Config keys:
        in_channels (int): 输入通道数，默认 3
        out_channels (int): 输出通道数，默认 3
        num_layers (int): 编码器层数（解码器层数相同），默认 15
        num_features (int): 中间层特征通道数，默认 64
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.num_layers = self.config.get("num_layers", 15)
        self.num_features = self.config.get("num_features", 64)

        # 编码器: 卷积层 + ReLU
        self.encoder_convs = nn.ModuleList()
        self.encoder_relus = nn.ModuleList()

        # 第一层
        self.encoder_convs.append(
            nn.Conv2d(self.in_channels, self.num_features, kernel_size=3, padding=1, bias=False)
        )
        self.encoder_relus.append(nn.ReLU(inplace=True))

        # 后续编码器层
        for _ in range(1, self.num_layers):
            self.encoder_convs.append(
                nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1, bias=False)
            )
            self.encoder_relus.append(nn.ReLU(inplace=True))

        # 解码器: 反卷积层 + ReLU
        self.decoder_convs = nn.ModuleList()
        self.decoder_relus = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.decoder_convs.append(
                nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1, bias=False)
            )
            self.decoder_relus.append(nn.ReLU(inplace=True))

        # 最后一层解码器输出到原始通道数
        self.decoder_convs.append(
            nn.ConvTranspose2d(self.num_features, self.out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.decoder_relus.append(nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        # 编码器路径，保存跳跃连接
        skip_connections = []
        out = x
        for conv, relu in zip(self.encoder_convs, self.encoder_relus):
            out = relu(conv(out))
            skip_connections.append(out)

        # 解码器路径，对称跳跃连接
        for i, (deconv, relu) in enumerate(zip(self.decoder_convs[:-1], self.decoder_relus[:-1])):
            out = deconv(out)
            # 对称跳跃连接
            skip_idx = self.num_layers - 1 - i
            if 0 <= skip_idx < len(skip_connections):
                out = out + skip_connections[skip_idx]
            out = relu(out)
        
        # 最后一层：不用 ReLU，直接输出
        out = self.decoder_convs[-1](out)
        
        # 残差学习：输出噪声，加到输入上得到去噪结果
        # 但要 clamp 到 [0, 1]
        out = (x - out).clamp(0, 1)
        
        return out

    def get_loss_function(self) -> nn.Module:
        """RED-Net 使用 MSE 损失函数。"""
        return nn.MSELoss()
