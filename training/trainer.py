"""训练引擎模块 - 管理模型训练、验证、检查点和学习率调度。"""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.base_model import BaseModel
from utils.logger import Logger


class Trainer:
    """管理模型训练流程。

    支持:
        - 训练循环和验证循环
        - 学习率调度（StepLR、ReduceLROnPlateau）
        - 检查点保存和加载（含最佳模型自动保存）
        - GPU/CPU 自动切换
        - 训练历史记录

    Args:
        model: 要训练的 BaseModel 实例
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        config: 训练配置字典，支持的键:
            - learning_rate (float): 学习率，默认 1e-3
            - num_epochs (int): 训练轮数，默认 100
            - checkpoint_dir (str): 检查点保存目录，默认 './checkpoints'
            - save_frequency (int): 每隔多少 epoch 保存检查点，默认 5
            - scheduler (dict): 学习率调度器配置
                - type (str): 'StepLR' 或 'ReduceLROnPlateau'
                - step_size (int): StepLR 的步长
                - gamma (float): StepLR 的衰减因子
                - patience (int): ReduceLROnPlateau 的耐心值
                - factor (float): ReduceLROnPlateau 的衰减因子
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict = None,
    ):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger.get_logger("trainer")

        # 模型
        self.model = model.to(self.device)
        
        # 损失函数：优先使用配置中的自定义损失，否则使用模型默认损失
        loss_config = self.config.get("loss", {})
        if loss_config:
            from models.losses import get_loss_function
            loss_type = loss_config.get("type", "mse")
            loss_params = loss_config.get("params", {})
            # 如果是组合损失，把模型原始损失传进去
            if loss_type == "combined":
                base_loss = model.get_loss_function()
                loss_params["base_loss"] = base_loss
                self.logger.info(f"组合损失基础项: {base_loss.__class__.__name__}")
            self.loss_fn = get_loss_function(loss_type, **loss_params).to(self.device)
            # 日志中过滤掉 base_loss 对象，避免输出不可读
            log_params = {k: v for k, v in loss_params.items() if not isinstance(v, nn.Module)}
            self.logger.info(f"使用自定义损失函数: {loss_type}, 参数: {log_params}")
        else:
            self.loss_fn = model.get_loss_function().to(self.device)
            self.logger.info(f"使用模型默认损失函数: {self.loss_fn.__class__.__name__}")

        # 数据
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化器
        lr = self.config.get("learning_rate", 1e-3)
        self.optimizer = model.get_optimizer(lr=lr)

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 检查点
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        self.save_frequency = self.config.get("save_frequency", 5)

        # TensorBoard
        self.use_tensorboard = self.config.get("use_tensorboard", True)
        if self.use_tensorboard:
            log_dir = os.path.join(self.checkpoint_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"TensorBoard 日志目录: {log_dir}")
        else:
            self.writer = None

        # 训练状态
        self.current_epoch = 0
        self.best_val_psnr = -float("inf")
        self.best_val_ssim = -float("inf")
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_psnr": [],
            "val_ssim": [],
            "learning_rate": [],
        }

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _create_scheduler(self):
        """根据配置创建学习率调度器（支持 warmup）。
        
        warmup 配置项（在 scheduler 下）：
            warmup_epochs (int): warmup 轮数，默认 0（不使用）
            warmup_start_lr (float): warmup 起始学习率，默认 1e-7
        """
        sched_cfg = self.config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "ReduceLROnPlateau")
        
        # warmup 配置
        warmup_epochs = sched_cfg.get("warmup_epochs", 0)
        warmup_start_lr = sched_cfg.get("warmup_start_lr", 1e-7)
        target_lr = self.config.get("learning_rate", 1e-3)

        # 创建主调度器
        if sched_type == "StepLR":
            step_size = sched_cfg.get("step_size", 30)
            gamma = sched_cfg.get("gamma", 0.1)
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif sched_type == "ReduceLROnPlateau":
            patience = sched_cfg.get("patience", 10)
            factor = sched_cfg.get("factor", 0.5)
            min_lr = sched_cfg.get("min_lr", 1e-7)
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=patience, 
                factor=factor, min_lr=min_lr
            )
        elif sched_type == "CosineAnnealingLR":
            T_max = sched_cfg.get("T_max", self.config.get("num_epochs", 100))
            eta_min = sched_cfg.get("eta_min", 1e-6)
            # 如果有 warmup，余弦退火的周期要减去 warmup 轮数
            if warmup_epochs > 0:
                T_max = max(1, T_max - warmup_epochs)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        else:
            self.logger.warning(
                f"未知的调度器类型 '{sched_type}'，使用 ReduceLROnPlateau"
            )
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=10, factor=0.5
            )
        
        # 如果配置了 warmup，用 SequentialLR 串联
        if warmup_epochs > 0 and not isinstance(
            main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            # LinearLR: 从 start_factor 线性增长到 1.0
            start_factor = warmup_start_lr / target_lr
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=max(start_factor, 1e-7),
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            self.logger.info(
                f"Warmup: {warmup_epochs} epochs, "
                f"{warmup_start_lr:.1e} → {target_lr:.1e}, 然后 {sched_type}"
            )
            return scheduler
        
        return main_scheduler

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self) -> float:
        """训练一个 epoch。

        Returns:
            该 epoch 的平均训练损失。
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 累积各子损失（用于组合损失的分项统计）
        sub_loss_totals = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", 
                    leave=False)
        
        # 计算每 10% 记录一次的步数
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # 每 10% 记录一次
        
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(noisy)
            loss = self.loss_fn(output, clean)
            
            # 跳过 NaN loss，避免污染模型参数
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # 收集各子损失
            if hasattr(self.loss_fn, 'last_loss_details'):
                for k, v in self.loss_fn.last_loss_details.items():
                    sub_loss_totals[k] = sub_loss_totals.get(k, 0.0) + v
            
            # 更新进度条显示当前loss和各子损失
            postfix = {'loss': f'{loss.item():.4f}'}
            if hasattr(self.loss_fn, 'last_loss_details'):
                for k, v in self.loss_fn.last_loss_details.items():
                    postfix[k] = f'{v:.4f}' if v >= 0.0001 else f'{v:.2e}'
            pbar.set_postfix(postfix)
            
            # 每 10% 记录一次到 TensorBoard
            if self.writer is not None and (batch_idx + 1) % log_interval == 0:
                global_step = (self.current_epoch - 1) * total_batches + batch_idx
                self.writer.add_scalar(
                    "Loss/train_step", 
                    loss.item(), 
                    global_step
                )

        avg_loss = total_loss / max(num_batches, 1)
        
        # 计算各子损失的平均值并保存
        self._last_sub_losses = {}
        if sub_loss_totals and num_batches > 0:
            for k, v in sub_loss_totals.items():
                self._last_sub_losses[k] = v / num_batches
        
        return avg_loss

    def validate(self) -> Dict[str, float]:
        """在验证集上评估模型。

        Returns:
            指标字典: {'val_psnr': float, 'val_ssim': float, 'val_loss': float}
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        num_samples = 0

        with torch.no_grad():
            for noisy, clean in tqdm(self.val_loader, desc="Validating", 
                                     leave=False):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                output = self.model(noisy)
                loss = self.loss_fn(output, clean)
                total_loss += loss.item()
                num_batches += 1

                # 计算 PSNR 和 SSIM（逐图像）
                batch_size = output.size(0)
                num_samples += batch_size
                for i in range(batch_size):
                    total_psnr += self._calculate_psnr(output[i], clean[i])
                    total_ssim += self._calculate_ssim(output[i], clean[i])

        num_samples = max(num_samples, 1)
        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_psnr": total_psnr / num_samples,
            "val_ssim": total_ssim / num_samples,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = None) -> Dict:
        """完整训练流程。

        Args:
            num_epochs: 训练轮数，默认从配置读取。

        Returns:
            训练历史字典。
        """
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 100)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        start_epoch = self.current_epoch + 1  # 从检查点恢复时接着训练
        self.logger.info(
            f"开始训练: epoch {start_epoch}-{num_epochs}, device={self.device}"
        )

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            
            # 更新损失函数的训练进度（阶段式，只在阶段切换时变化）
            if hasattr(self.loss_fn, 'set_training_progress'):
                progress = (epoch - 1) / max(num_epochs - 1, 1)
                old_stage = getattr(self.loss_fn, '_current_stage', -1)
                self.loss_fn.set_training_progress(progress)
                new_stage = getattr(self.loss_fn, '_current_stage', -1)
                if new_stage != old_stage and hasattr(self.loss_fn, 'get_stage_info'):
                    self.logger.info(f"损失函数切换: {self.loss_fn.get_stage_info()}")

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_metrics = self.validate()
            val_psnr = val_metrics["val_psnr"]
            val_ssim = val_metrics["val_ssim"]

            # 当前学习率
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 记录历史
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_psnr"].append(val_psnr)
            self.history["val_ssim"].append(val_ssim)
            self.history["learning_rate"].append(current_lr)

            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"loss: {train_loss:.6f}, "
                f"val_psnr: {val_psnr:.4f}, "
                f"val_ssim: {val_ssim:.4f}, "
                f"lr: {current_lr:.2e}"
            )
            
            # 打印各子损失分项
            if hasattr(self, '_last_sub_losses') and self._last_sub_losses:
                parts = [f"{k}: {v:.6f}" for k, v in self._last_sub_losses.items()]
                self.logger.info(f"  子损失 - {', '.join(parts)}")

            # 记录到 TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Metrics/PSNR", val_psnr, epoch)
                self.writer.add_scalar("Metrics/SSIM", val_ssim, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                
                # 记录各子损失到 TensorBoard
                if hasattr(self, '_last_sub_losses') and self._last_sub_losses:
                    for k, v in self._last_sub_losses.items():
                        self.writer.add_scalar(f"SubLoss/{k}", v, epoch)
                
                # 记录损失权重（阶段式，不频繁变化）
                if hasattr(self.loss_fn, '_get_current_edge_weight'):
                    self.writer.add_scalar("Loss_Weights/edge_weight", 
                                          self.loss_fn._get_current_edge_weight(), epoch)
                
                # 每个 epoch 都记录图像可视化
                self._log_images_to_tensorboard(epoch)

            # 更新学习率调度器
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_psnr)
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.SequentialLR):
                # SequentialLR 内部自动切换，直接 step
                self.scheduler.step()
            else:
                self.scheduler.step()

            # 保存最佳模型
            is_best = val_psnr > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_psnr
                self.best_val_ssim = val_ssim
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                    is_best=True,
                )
            else:
                self.logger.debug(
                    f"未更新最佳模型: val_psnr={val_psnr:.4f} <= best={self.best_val_psnr:.4f}"
                )

            # 定期保存检查点
            if epoch % self.save_frequency == 0:
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                )

        self.logger.info("训练完成")
        
        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return self.history

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """保存训练检查点。

        Args:
            path: 检查点文件路径。
            is_best: 是否为最佳模型。
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_psnr": self.best_val_psnr,
            "best_val_ssim": self.best_val_ssim,
            "history": self.history,
            "config": self.config,
            "is_best": is_best,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"检查点已保存: {path}" + (" (最佳模型)" if is_best else ""))

    def load_checkpoint(self, path: str) -> None:
        """加载检查点恢复训练。

        Args:
            path: 检查点文件路径。

        Raises:
            FileNotFoundError: 检查点文件不存在。
            RuntimeError: 检查点文件损坏或不兼容。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"检查点加载失败: {e}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # scheduler 加载：格式不兼容时跳过（如旧检查点无 warmup，新配置有 warmup）
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (KeyError, TypeError, ValueError) as e:
            self.logger.warning(f"Scheduler state_dict 不兼容，将使用新配置的调度器: {e}")
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_psnr = checkpoint["best_val_psnr"]
        self.best_val_ssim = checkpoint.get("best_val_ssim", -float("inf"))
        self.history = checkpoint["history"]

        self.logger.info(
            f"检查点已加载: {path}, epoch={self.current_epoch}, "
            f"best_psnr={self.best_val_psnr:.4f}"
        )

    # ------------------------------------------------------------------
    # TensorBoard 图像可视化
    # ------------------------------------------------------------------

    def _log_images_to_tensorboard(self, epoch: int, num_images: int = 4):
        """记录去噪效果图像到 TensorBoard。
        
        Args:
            epoch: 当前 epoch
            num_images: 记录多少张图像
        """
        if self.writer is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            # 从验证集取几张图像
            for i, (noisy, clean) in enumerate(self.val_loader):
                if i >= num_images:
                    break
                
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # 去噪
                denoised = self.model(noisy).clamp(0.0, 1.0)
                
                # 只取 batch 中的第一张图
                noisy_img = noisy[0]
                clean_img = clean[0]
                denoised_img = denoised[0]
                
                # 拼接成对比图: [noisy | denoised | clean]
                comparison = torch.cat([noisy_img, denoised_img, clean_img], dim=2)
                
                # 记录到 TensorBoard
                self.writer.add_image(
                    f"Denoising/sample_{i+1}",
                    comparison,
                    epoch,
                    dataformats='CHW'
                )
        
        self.model.train()

    # ------------------------------------------------------------------
    # Metric helpers (lightweight, no external dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """计算两张图像的 PSNR（峰值信噪比）。

        Args:
            img1, img2: 形状 (C, H, W)，值范围 [0, 1]。

        Returns:
            PSNR 值（dB）。
        """
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse < 1e-10:
            return 100.0  # 近似无穷大
        return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

    @staticmethod
    def _calculate_ssim(
        img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
    ) -> float:
        """计算两张图像的 SSIM（结构相似性）。

        简化实现，使用均值和方差近似。

        Args:
            img1, img2: 形状 (C, H, W)，值范围 [0, 1]。

        Returns:
            SSIM 值，范围 [0, 1]。
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = img1.var()
        sigma2_sq = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_val = (numerator / denominator).item()
        return max(0.0, min(1.0, ssim_val))
