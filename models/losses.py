"""自定义损失函数 - 针对细线条恢复优化的损失函数"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightedMSELoss(nn.Module):
    """加权 MSE 损失 - 对亮像素（线条）赋予更高权重
    
    适用场景：黑色背景 + 白色细线条
    原理：对真值中的亮像素（线条区域）赋予更高的权重，
         强制模型更关注线条的恢复而不是背景
    
    输出已归一化：除以平均权重，使输出量级与标准 MSE 一致。
    
    Args:
        foreground_weight: 前景（亮像素）权重，默认 10.0
        threshold: 判断前景的阈值（0-1），默认 0.5
    """
    
    def __init__(self, foreground_weight: float = 10.0, threshold: float = 0.5):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.threshold = threshold
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: 预测图像 (B, C, H, W)
            target: 真值图像 (B, C, H, W)
        """
        # 创建权重图：真值中亮的地方权重高
        target_gray = target.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        weight_map = torch.where(
            target_gray > self.threshold,
            torch.tensor(self.foreground_weight, device=target.device, dtype=target.dtype),
            torch.tensor(1.0, device=target.device, dtype=target.dtype)
        )
        
        # 加权 MSE，归一化使量级与标准 MSE 一致
        mse = (pred - target) ** 2
        weighted_mse = (mse * weight_map).sum() / (weight_map.sum() + 1e-8)
        return weighted_mse


class EdgePreservingLoss(nn.Module):
    """边缘保持损失 - MSE + 边缘损失
    
    适用场景：需要保持细节和边缘的去噪任务
    原理：在 MSE 基础上增加边缘损失，强制模型保持边缘清晰
    
    Args:
        edge_weight: 边缘损失权重，默认 0.5
        mse_weight: MSE 损失权重，默认 1.0
    """
    
    def __init__(self, edge_weight: float = 0.5, mse_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        
        # Sobel 算子用于边缘检测
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0))
    
    def _compute_edges(self, img: Tensor) -> Tensor:
        """计算图像边缘强度"""
        gray = img.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 确保 Sobel 算子在正确的设备上
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # 边缘强度（加 eps 避免 sqrt(0) 的梯度问题）
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edges
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # MSE 损失
        mse = self.mse_loss(pred, target)
        
        # 边缘损失 — 只在有边缘的区域计算，避免被大量黑色背景稀释
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        
        # 用真值边缘作为 mask：边缘强度 > 阈值的区域才参与计算
        edge_mask = (target_edges > 0.01).float()
        mask_sum = edge_mask.sum() + 1e-8
        
        # 在边缘区域计算 MSE
        edge_diff = (pred_edges - target_edges) ** 2
        edge_loss = (edge_diff * edge_mask).sum() / mask_sum
        
        # 数值稳定性检查
        if torch.isnan(mse) or torch.isnan(edge_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 组合
        total_loss = self.mse_weight * mse + self.edge_weight * edge_loss
        return total_loss


class FocalMSELoss(nn.Module):
    """Focal MSE 损失 - 对难样本（误差大的像素）赋予更高权重
    
    适用场景：模型容易忽略的细节区域（如细线条）
    原理：借鉴 Focal Loss 思想，对误差大的像素赋予更高权重，
         强制模型关注难以恢复的区域
    
    输出已归一化：除以平均权重，使输出量级与标准 MSE 一致。
    
    Args:
        gamma: 聚焦参数，默认 2.0（越大越关注难样本）
    """
    
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = (pred - target) ** 2
        
        # Focal 权重：误差越大，权重越高
        mse_detached = mse.detach()
        mse_max = mse_detached.max()
        if mse_max < 1e-8:
            return mse.mean()
        
        mse_normalized = mse_detached / (mse_max + 1e-8)
        focal_weight = (1 - torch.exp(-mse_normalized)) ** self.gamma
        
        # 归一化：除以平均权重，使量级与标准 MSE 一致
        weight_mean = focal_weight.mean() + 1e-8
        focal_weight = focal_weight / weight_mean
        
        result = (mse * focal_weight).mean()
        
        if torch.isnan(result) or torch.isinf(result):
            return mse.mean()
        
        return result


class CombinedLoss(nn.Module):
    """组合损失 - 模型原始损失 + 加权 MSE + 边缘保持 + Focal
    
    在模型原本的损失函数基础上，叠加针对细线条恢复的增强损失。
    各子损失已归一化到与标准 MSE 相同量级，再用权重系数加权平均。
    
    Args:
        base_loss: 模型原本的损失函数（MSE/L1/Charbonnier），默认 None（用 MSE）
        base_weight: 原始损失的权重，默认 1.0
        foreground_weight: 前景权重（传给 WeightedMSE），默认 10.0
        weighted_mse_weight: WeightedMSE 的混合系数，默认 0.5
        edge_weight: 边缘损失的最终混合系数，默认 0.3
        focal_gamma: Focal 参数，默认 2.0
        focal_weight: Focal 损失的混合系数，默认 0.2
        use_edge: 是否使用边缘损失，默认 True
        use_focal: 是否使用 Focal 损失，默认 True
        num_stages: 训练阶段数，默认 3
        stage_ratios: 各阶段占比列表，默认 [0.4, 0.3, 0.3]
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        base_weight: float = 1.0,
        foreground_weight: float = 10.0,
        weighted_mse_weight: float = 0.5,
        edge_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.2,
        use_edge: bool = True,
        use_focal: bool = True,
        num_stages: int = 3,
        stage_ratios: list = None,
        # 兼容旧参数
        edge_weight_schedule: str = None,
        foreground_weight_schedule: str = None
    ):
        super().__init__()
        # 模型原始损失
        self.base_loss = base_loss if base_loss is not None else nn.MSELoss()
        self.base_weight = base_weight
        
        # 增强损失
        self.weighted_mse = WeightedMSELoss(foreground_weight=foreground_weight)
        self.weighted_mse_weight = weighted_mse_weight
        self.edge_loss = EdgePreservingLoss(edge_weight=1.0, mse_weight=0.0) if use_edge else None
        self.focal_loss = FocalMSELoss(gamma=focal_gamma) if use_focal else None
        self.use_edge = use_edge
        self.use_focal = use_focal
        
        self._edge_weight_max = edge_weight
        self._focal_weight = focal_weight
        
        # 阶段式调整
        self.num_stages = max(1, num_stages)
        if stage_ratios is not None:
            total = sum(stage_ratios)
            self.stage_ratios = [r / total for r in stage_ratios]
        else:
            if self.num_stages == 1:
                self.stage_ratios = [1.0]
            elif self.num_stages == 2:
                self.stage_ratios = [0.5, 0.5]
            else:
                self.stage_ratios = [0.4, 0.3, 0.3]
        
        self._stage_edge_weights = self._compute_stage_weights()
        self._current_stage = 0
        self._current_edge_weight = self._stage_edge_weights[0]
    
    def _compute_stage_weights(self) -> list:
        if self.num_stages == 1:
            return [self._edge_weight_max]
        weights = []
        for i in range(self.num_stages):
            w = self._edge_weight_max * i / (self.num_stages - 1)
            weights.append(w)
        return weights
    
    def set_training_progress(self, progress: float):
        progress = max(0.0, min(1.0, progress))
        cumulative = 0.0
        new_stage = self.num_stages - 1
        for i, ratio in enumerate(self.stage_ratios):
            cumulative += ratio
            if progress < cumulative:
                new_stage = i
                break
        if new_stage != self._current_stage:
            self._current_stage = new_stage
            self._current_edge_weight = self._stage_edge_weights[new_stage]
    
    def _get_current_edge_weight(self) -> float:
        return self._current_edge_weight
    
    def get_stage_info(self) -> str:
        return (f"阶段 {self._current_stage + 1}/{self.num_stages}, "
                f"边缘权重={self._current_edge_weight:.3f}")
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # 记录各子损失乘权重后的值，加起来 = 总 loss
        self.last_loss_details = {}
        
        # 1. 模型原始损失（基础项）
        base_val = self.base_loss(pred, target)
        if torch.isnan(base_val) or torch.isinf(base_val):
            base_val = F.mse_loss(pred, target)
        weighted_base = self.base_weight * base_val
        self.last_loss_details['base'] = weighted_base.item()
        loss = weighted_base
        
        # 2. 加权 MSE（已归一化）
        wmse = self.weighted_mse(pred, target)
        if not (torch.isnan(wmse) or torch.isinf(wmse)):
            weighted_wmse = self.weighted_mse_weight * wmse
            self.last_loss_details['w_mse'] = weighted_wmse.item()
            loss = loss + weighted_wmse
        
        # 3. 边缘损失（阶段式权重）
        if self.use_edge and self.edge_loss is not None:
            edge_val = self.edge_loss(pred, target)
            if not (torch.isnan(edge_val) or torch.isinf(edge_val)):
                weighted_edge = self._current_edge_weight * edge_val
                self.last_loss_details['edge'] = weighted_edge.item()
                if self._current_edge_weight > 0:
                    loss = loss + weighted_edge
        
        # 4. Focal 损失（已归一化）
        if self.use_focal and self.focal_loss is not None:
            focal_val = self.focal_loss(pred, target)
            if not (torch.isnan(focal_val) or torch.isinf(focal_val)):
                weighted_focal = self._focal_weight * focal_val
                self.last_loss_details['focal'] = weighted_focal.item()
                loss = loss + weighted_focal
        
        if torch.isnan(loss) or torch.isinf(loss):
            return F.mse_loss(pred, target)
        
        return loss


class PerceptualWeightedLoss(nn.Module):
    """感知加权损失 - 基于真值的亮度分布自适应调整权重
    
    适用场景：自动识别重要区域（线条）并赋予更高权重
    
    Args:
        percentile: 用于确定前景的百分位数，默认 90（前 10% 最亮的像素）
        min_weight: 背景最小权重，默认 1.0
        max_weight: 前景最大权重，默认 20.0
    """
    
    def __init__(
        self,
        percentile: float = 90.0,
        min_weight: float = 1.0,
        max_weight: float = 20.0
    ):
        super().__init__()
        self.percentile = percentile
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # 转为灰度图
        target_gray = target.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 计算阈值（每个 batch 独立计算）
        B = target.size(0)
        weight_map = torch.ones_like(target_gray)
        
        for i in range(B):
            # 计算该图像的亮度阈值
            threshold = torch.quantile(
                target_gray[i].flatten(),
                self.percentile / 100.0
            )
            
            # 创建权重图：线性映射
            # 亮度越高，权重越大
            weight_map[i] = torch.where(
                target_gray[i] > threshold,
                self.min_weight + (self.max_weight - self.min_weight) * 
                (target_gray[i] - threshold) / (1.0 - threshold + 1e-8),
                torch.tensor(self.min_weight, device=target.device)
            )
        
        # 加权 MSE
        mse = (pred - target) ** 2
        weighted_mse = mse * weight_map
        return weighted_mse.mean()


# 便捷函数
def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """获取损失函数
    
    Args:
        loss_type: 损失函数类型
            - 'mse': 标准 MSE
            - 'weighted_mse': 加权 MSE（推荐用于细线条）
            - 'edge_preserving': 边缘保持损失
            - 'focal_mse': Focal MSE
            - 'combined': 组合损失（最强，推荐）
            - 'perceptual_weighted': 感知加权损失（自适应）
        **kwargs: 损失函数参数
    
    Returns:
        损失函数实例
    """
    loss_map = {
        'mse': nn.MSELoss,
        'weighted_mse': WeightedMSELoss,
        'edge_preserving': EdgePreservingLoss,
        'focal_mse': FocalMSELoss,
        'combined': CombinedLoss,
        'perceptual_weighted': PerceptualWeightedLoss,
    }
    
    if loss_type not in loss_map:
        raise ValueError(
            f"未知的损失函数类型: {loss_type}. "
            f"可选: {list(loss_map.keys())}"
        )
    
    return loss_map[loss_type](**kwargs)
