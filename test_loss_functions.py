"""测试自定义损失函数"""

import torch
from models.losses import (
    WeightedMSELoss,
    EdgePreservingLoss,
    FocalMSELoss,
    CombinedLoss,
    PerceptualWeightedLoss,
    get_loss_function
)

def create_test_data():
    """创建测试数据：黑色背景 + 白色线条"""
    # 模拟真值：黑色背景 + 白色线条
    target = torch.zeros(2, 3, 64, 64)
    target[:, :, 30:34, :] = 1.0  # 水平线条
    target[:, :, :, 30:34] = 1.0  # 垂直线条
    
    # 模拟预测1：完全黑色（去掉了白斑，但也去掉了线条）
    pred_bad = torch.zeros_like(target)
    
    # 模拟预测2：保留了线条
    pred_good = target.clone()
    pred_good += torch.randn_like(pred_good) * 0.1  # 添加小噪声
    pred_good = pred_good.clamp(0, 1)
    
    return target, pred_bad, pred_good

def test_loss_function(loss_fn, name):
    """测试单个损失函数"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    target, pred_bad, pred_good = create_test_data()
    
    # 计算损失
    loss_bad = loss_fn(pred_bad, target).item()
    loss_good = loss_fn(pred_good, target).item()
    
    print(f"预测1（全黑，丢失线条）: Loss = {loss_bad:.6f}")
    print(f"预测2（保留线条）:       Loss = {loss_good:.6f}")
    print(f"差异: {abs(loss_bad - loss_good):.6f}")
    
    # 理想情况：loss_good < loss_bad（保留线条的预测应该有更低的损失）
    if loss_good < loss_bad:
        print("✅ 正确：保留线条的预测损失更低")
    else:
        print("❌ 错误：全黑预测损失更低（损失函数无效）")
    
    return loss_bad, loss_good

def main():
    print("="*60)
    print("自定义损失函数测试")
    print("="*60)
    print("\n场景：黑色背景 + 白色线条")
    print("预测1：全黑（去掉白斑，但也去掉线条）")
    print("预测2：保留线条（正确的去噪）")
    print("\n期望：预测2的损失应该更低")
    
    # 1. 标准 MSE（基线）
    mse_loss = torch.nn.MSELoss()
    test_loss_function(mse_loss, "标准 MSE（基线）")
    
    # 2. 加权 MSE
    weighted_mse = WeightedMSELoss(foreground_weight=10.0)
    test_loss_function(weighted_mse, "加权 MSE (weight=10)")
    
    # 3. 边缘保持
    edge_loss = EdgePreservingLoss(edge_weight=0.5)
    test_loss_function(edge_loss, "边缘保持损失")
    
    # 4. Focal MSE
    focal_loss = FocalMSELoss(gamma=2.0)
    test_loss_function(focal_loss, "Focal MSE")
    
    # 5. 组合损失
    combined_loss = CombinedLoss(
        foreground_weight=15.0,
        edge_weight=0.5,
        focal_gamma=2.0
    )
    test_loss_function(combined_loss, "组合损失（最强）")
    
    # 6. 感知加权
    perceptual_loss = PerceptualWeightedLoss(percentile=90.0, max_weight=20.0)
    test_loss_function(perceptual_loss, "感知加权损失")
    
    # 7. 测试 get_loss_function
    print(f"\n{'='*60}")
    print("测试 get_loss_function 工厂函数")
    print(f"{'='*60}")
    
    loss_types = ['mse', 'weighted_mse', 'edge_preserving', 'focal_mse', 'combined', 'perceptual_weighted']
    for loss_type in loss_types:
        try:
            loss_fn = get_loss_function(loss_type)
            print(f"✅ {loss_type}: {loss_fn.__class__.__name__}")
        except Exception as e:
            print(f"❌ {loss_type}: {e}")
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")
    print("\n建议：")
    print("1. 如果标准 MSE 无法区分好坏预测，使用加权 MSE")
    print("2. 如果需要最佳效果，使用组合损失")
    print("3. 如果数据集亮度不一致，使用感知加权损失")

if __name__ == "__main__":
    main()
