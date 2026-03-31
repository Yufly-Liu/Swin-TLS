"""Restormer - Efficient Transformer for High-Resolution Image Restoration.

参考: Zamir et al., 2022
官方架构: 基于Transformer的图像恢复网络，使用多头转置自注意力(MDTA)和
门控前馈网络(GDFN)。
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base_model import BaseModel


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 的平滑近似)。"""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class LayerNorm2d(nn.Module):
    """适用于图像特征的 LayerNorm (B, C, H, W)。"""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, H, W, C) -> norm -> (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention (MDTA)。

    转置自注意力: 在通道维度而非空间维度计算注意力，
    降低高分辨率图像的计算复杂度。
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, -1, H * W)
        k = k.reshape(B, self.num_heads, -1, H * W)
        v = v.reshape(B, self.num_heads, -1, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 转置注意力: (C/h, HW) × (HW, C/h) -> (C/h, C/h)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network (GDFN)。"""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Restormer Transformer Block: LayerNorm → MDTA → LayerNorm → GDFN。"""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Restormer(BaseModel):
    """Restormer 图像恢复网络。

    U-Net风格的多尺度Transformer架构:
    - 编码器: 逐级下采样，增加通道数
    - 解码器: 逐级上采样，减少通道数
    - 每级包含多个Transformer块

    Config keys:
        in_channels (int): 输入通道数，默认 3
        out_channels (int): 输出通道数，默认 3
        dim (int): 基础特征维度，默认 48
        num_blocks (list): 每级Transformer块数，默认 [4, 6, 6, 8]
        num_heads (list): 每级注意力头数，默认 [1, 2, 4, 8]
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        dim = self.config.get("dim", 48)
        num_blocks = self.config.get("num_blocks", [4, 6, 6, 8])
        num_heads = self.config.get("restormer_num_heads", self.config.get("num_heads", [1, 2, 4, 8]))

        # 输入投影
        self.input_proj = nn.Conv2d(self.in_channels, dim, kernel_size=3, padding=1, bias=False)

        # 编码器
        self.encoder1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1, bias=False)

        # 瓶颈
        self.bottleneck = nn.Sequential(*[TransformerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])

        # 解码器
        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2, bias=False)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.decoder2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2, bias=False)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.decoder1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[3])])

        # 输出投影
        self.output_proj = nn.Conv2d(dim, self.out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.input_proj(x)

        # 编码器
        enc1 = self.encoder1(x)
        x = self.down1(enc1)

        enc2 = self.encoder2(x)
        x = self.down2(enc2)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码器
        x = self.up2(x)
        if x.shape != enc2.shape:
            x = F.interpolate(x, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        x = self.reduce2(torch.cat([x, enc2], dim=1))
        x = self.decoder2(x)

        x = self.up1(x)
        if x.shape != enc1.shape:
            x = F.interpolate(x, size=enc1.shape[2:], mode="bilinear", align_corners=False)
        x = self.reduce1(torch.cat([x, enc1], dim=1))
        x = self.decoder1(x)

        x = self.output_proj(x)
        return x + residual

    def get_loss_function(self) -> nn.Module:
        """Restormer 使用 Charbonnier 损失函数。"""
        return CharbonnierLoss()
