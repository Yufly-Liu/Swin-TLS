"""评估引擎模块 - 在测试集上评估模型性能并保存去噪样本。"""

import os
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.base_model import BaseModel
from evaluation.metrics import Metrics, MetricError
from utils.logger import Logger


class Evaluator:
    """评估训练好的模型性能。

    在测试集上计算 PSNR、SSIM、MSE 指标的均值和标准差，
    并支持保存去噪结果样本图像。

    Args:
        model: 要评估的 BaseModel 实例
        test_loader: 测试数据 DataLoader
    """

    def __init__(self, model: BaseModel, test_loader: DataLoader):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = Logger.get_logger("evaluator")

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """在测试集上评估模型。

        Returns:
            指标字典，格式:
            {
                'psnr': {'mean': float, 'std': float},
                'ssim': {'mean': float, 'std': float},
                'mse': {'mean': float, 'std': float},
            }

        Raises:
            DatasetError: 测试集为空
        """
        self.model.eval()

        psnr_values: List[float] = []
        ssim_values: List[float] = []
        mse_values: List[float] = []

        with torch.no_grad():
            for noisy, clean in self.test_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                output = self.model(noisy)
                # Clamp output to [0, 1]
                output = output.clamp(0.0, 1.0)

                batch_size = output.size(0)
                for i in range(batch_size):
                    psnr_values.append(Metrics.calculate_psnr(output[i], clean[i]))
                    ssim_values.append(Metrics.calculate_ssim(output[i], clean[i]))
                    mse_values.append(Metrics.calculate_mse(output[i], clean[i]))

        if not psnr_values:
            from data.dataset import DatasetError
            raise DatasetError("测试集为空，无法评估")

        results = {
            "psnr": {
                "mean": float(np.mean(psnr_values)),
                "std": float(np.std(psnr_values)),
            },
            "ssim": {
                "mean": float(np.mean(ssim_values)),
                "std": float(np.std(ssim_values)),
            },
            "mse": {
                "mean": float(np.mean(mse_values)),
                "std": float(np.std(mse_values)),
            },
        }

        self.logger.info(
            f"评估完成 - PSNR: {results['psnr']['mean']:.4f}±{results['psnr']['std']:.4f}, "
            f"SSIM: {results['ssim']['mean']:.4f}±{results['ssim']['std']:.4f}, "
            f"MSE: {results['mse']['mean']:.6f}±{results['mse']['std']:.6f}"
        )

        return results

    def save_sample_results(
        self, output_dir: str, num_samples: int = 10
    ) -> List[str]:
        """保存去噪结果样本图像。

        对每个样本保存一张包含 noisy / denoised / clean 的对比图。

        Args:
            output_dir: 输出目录
            num_samples: 要保存的样本数量

        Returns:
            保存的文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        saved_paths: List[str] = []
        count = 0

        with torch.no_grad():
            for noisy, clean in self.test_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                output = self.model(noisy).clamp(0.0, 1.0)

                batch_size = output.size(0)
                for i in range(batch_size):
                    if count >= num_samples:
                        break

                    # Stack noisy, denoised, clean side by side
                    comparison = torch.cat(
                        [noisy[i].cpu(), output[i].cpu(), clean[i].cpu()], dim=2
                    )
                    path = os.path.join(output_dir, f"sample_{count:04d}.png")
                    save_image(comparison, path)
                    saved_paths.append(path)
                    count += 1

                if count >= num_samples:
                    break

        self.logger.info(f"已保存 {len(saved_paths)} 个去噪样本到 {output_dir}")
        return saved_paths
