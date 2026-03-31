"""可视化工具模块 - 绘制训练曲线、模型对比图表和去噪前后对比图。"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from utils.logger import Logger


class Visualizer:
    """生成训练过程和评估结果的可视化图表。

    所有方法均为静态方法，支持将结果保存为图像文件。
    """

    _logger = None

    @classmethod
    def _get_logger(cls):
        if cls._logger is None:
            cls._logger = Logger.get_logger("visualizer")
        return cls._logger

    @staticmethod
    def plot_training_curves(history: Dict[str, list], save_path: str) -> None:
        """绘制训练损失和验证指标曲线。

        Args:
            history: 训练历史字典，包含 'epoch', 'train_loss',
                     以及可选的 'val_psnr', 'val_ssim', 'learning_rate' 等。
            save_path: 图像保存路径。
        """
        epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

        # Count how many subplots we need
        has_loss = "train_loss" in history and len(history["train_loss"]) > 0
        has_psnr = "val_psnr" in history and len(history["val_psnr"]) > 0
        has_ssim = "val_ssim" in history and len(history["val_ssim"]) > 0
        has_lr = "learning_rate" in history and len(history["learning_rate"]) > 0

        n_plots = sum([has_loss, has_psnr, has_ssim, has_lr])
        if n_plots == 0:
            Visualizer._get_logger().warning("训练历史为空，跳过绘图")
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), squeeze=False)
        axes = axes.flatten()
        idx = 0

        if has_loss:
            axes[idx].plot(epochs[:len(history["train_loss"])], history["train_loss"], "b-", label="Train Loss")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Loss")
            axes[idx].set_title("Training Loss")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1

        if has_psnr:
            axes[idx].plot(epochs[:len(history["val_psnr"])], history["val_psnr"], "g-", label="Val PSNR")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("PSNR (dB)")
            axes[idx].set_title("Validation PSNR")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1

        if has_ssim:
            axes[idx].plot(epochs[:len(history["val_ssim"])], history["val_ssim"], "r-", label="Val SSIM")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("SSIM")
            axes[idx].set_title("Validation SSIM")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1

        if has_lr:
            axes[idx].plot(epochs[:len(history["learning_rate"])], history["learning_rate"], "m-", label="Learning Rate")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("LR")
            axes[idx].set_title("Learning Rate Schedule")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        Visualizer._get_logger().info(f"训练曲线已保存: {save_path}")

    @staticmethod
    def plot_comparison_bar(results: pd.DataFrame, save_path: str,
                           metrics: Optional[List[str]] = None) -> None:
        """绘制模型对比柱状图。

        Args:
            results: 包含模型名称和指标的 DataFrame。
                     需要 'name' 或 'model' 列作为模型标识。
            save_path: 图像保存路径。
            metrics: 要绘制的指标列名列表。默认自动检测。
        """
        # Determine model label column
        label_col = "name" if "name" in results.columns else "model"
        if label_col not in results.columns:
            Visualizer._get_logger().warning("DataFrame 缺少 name/model 列，跳过绘图")
            return

        if metrics is None:
            # Auto-detect numeric metric columns
            exclude = {"exp_id", "name", "model", "status", "created_at"}
            metrics = [c for c in results.columns
                       if c not in exclude and pd.api.types.is_numeric_dtype(results[c])]

        if not metrics:
            Visualizer._get_logger().warning("没有可绘制的指标列，跳过绘图")
            return

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
        axes = axes.flatten()

        labels = results[label_col].tolist()
        x = np.arange(len(labels))

        for i, metric in enumerate(metrics):
            values = results[metric].tolist()
            bars = axes[i].bar(x, values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(labels, rotation=45, ha="right")
            axes[i].set_title(metric)
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        Visualizer._get_logger().info(f"对比柱状图已保存: {save_path}")

    @staticmethod
    def save_image_comparison(noisy: Tensor, denoised: Tensor,
                              clean: Tensor, save_path: str) -> None:
        """保存去噪前后对比图（噪声图 | 去噪图 | 干净图）。

        Args:
            noisy: 噪声图像张量，形状 (C, H, W) 或 (H, W)。
            denoised: 去噪后图像张量。
            clean: 干净图像张量。
            save_path: 图像保存路径。
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ["Noisy", "Denoised", "Clean"]
        images = [noisy, denoised, clean]

        for ax, img, title in zip(axes, images, titles):
            img_np = Visualizer._tensor_to_numpy(img)
            if img_np.ndim == 2:
                ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img_np)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        Visualizer._get_logger().info(f"对比图像已保存: {save_path}")

    @staticmethod
    def _tensor_to_numpy(tensor: Tensor) -> np.ndarray:
        """将 PyTorch 张量转换为 numpy 数组用于 matplotlib 显示。

        Args:
            tensor: 形状 (C, H, W) 或 (H, W) 的张量。

        Returns:
            形状 (H, W, C) 或 (H, W) 的 numpy 数组，值裁剪到 [0, 1]。
        """
        img = tensor.detach().cpu().float()
        img = torch.clamp(img, 0.0, 1.0)
        if img.dim() == 3:
            # (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0)
        return img.numpy()
