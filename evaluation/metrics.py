"""评估指标模块 - 实现PSNR、SSIM、MSE等图像质量评估指标。"""

import torch
import torch.nn.functional as F
from torch import Tensor


class MetricError(Exception):
    """指标计算相关错误。"""
    pass


class Metrics:
    """图像质量评估指标集合。

    所有方法接受形状为 (C, H, W) 或 (B, C, H, W) 的张量，
    值范围应为 [0, 1]。
    """

    @staticmethod
    def calculate_psnr(img1: Tensor, img2: Tensor, max_val: float = 1.0) -> float:
        """计算峰值信噪比 (PSNR)。

        PSNR = 10 * log10(MAX^2 / MSE)

        Args:
            img1: 图像张量
            img2: 图像张量（与 img1 形状相同）
            max_val: 像素最大值，默认 1.0

        Returns:
            PSNR 值 (dB)。相同图像返回 100.0 作为近似无穷大。

        Raises:
            MetricError: 图像形状不匹配
        """
        if img1.shape != img2.shape:
            raise MetricError(
                f"图像形状不匹配: {img1.shape} vs {img2.shape}"
            )

        mse = torch.mean((img1.float() - img2.float()) ** 2).item()
        if mse < 1e-10:
            return 100.0
        psnr = 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()
        return psnr

    @staticmethod
    def calculate_ssim(
        img1: Tensor,
        img2: Tensor,
        window_size: int = 11,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ) -> float:
        """计算结构相似性指数 (SSIM)。

        使用滑动窗口高斯加权计算局部统计量。

        Args:
            img1: 图像张量，形状 (C, H, W) 或 (B, C, H, W)
            img2: 图像张量（与 img1 形状相同）
            window_size: 高斯窗口大小
            C1: 亮度稳定常数
            C2: 对比度稳定常数

        Returns:
            SSIM 值，范围 [0, 1]。相同图像返回 1.0。

        Raises:
            MetricError: 图像形状不匹配
        """
        if img1.shape != img2.shape:
            raise MetricError(
                f"图像形状不匹配: {img1.shape} vs {img2.shape}"
            )

        # Ensure 4D: (B, C, H, W)
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        img1 = img1.float()
        img2 = img2.float()

        channels = img1.size(1)

        # Create Gaussian window
        window = Metrics._gaussian_window(window_size, channels, img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

        # Clamp variances to avoid negative values from numerical errors
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        ssim_val = ssim_map.mean().item()

        return max(0.0, min(1.0, ssim_val))

    @staticmethod
    def _gaussian_window(size: int, channels: int, device: torch.device) -> Tensor:
        """创建高斯窗口核。"""
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        return window

    @staticmethod
    def calculate_mse(img1: Tensor, img2: Tensor) -> float:
        """计算均方误差 (MSE)。

        Args:
            img1: 图像张量
            img2: 图像张量（与 img1 形状相同）

        Returns:
            MSE 值，非负。相同图像返回 0.0。

        Raises:
            MetricError: 图像形状不匹配
        """
        if img1.shape != img2.shape:
            raise MetricError(
                f"图像形状不匹配: {img1.shape} vs {img2.shape}"
            )

        mse = torch.mean((img1.float() - img2.float()) ** 2).item()
        return mse
