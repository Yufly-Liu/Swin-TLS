"""测试阶段式权重调整功能"""

import torch
from models.losses import CombinedLoss

def test_stage_weights():
    print("=" * 60)
    print("测试阶段式权重调整")
    print("=" * 60)
    
    # 3 阶段，比例 40%/30%/30%
    loss_fn = CombinedLoss(
        foreground_weight=10.0,
        edge_weight=0.5,
        focal_gamma=2.0,
        focal_weight=0.2,
        use_edge=True,
        use_focal=True,
        num_stages=3,
        stage_ratios=[0.4, 0.3, 0.3]
    )
    
    print(f"\n阶段配置: {loss_fn.num_stages} 阶段")
    print(f"阶段占比: {loss_fn.stage_ratios}")
    print(f"各阶段边缘权重: {loss_fn._stage_edge_weights}")
    
    print("\n模拟 100 epoch 训练:")
    print("-" * 50)
    
    total_epochs = 100
    last_stage = -1
    for epoch in range(1, total_epochs + 1):
        progress = (epoch - 1) / (total_epochs - 1)
        loss_fn.set_training_progress(progress)
        
        if loss_fn._current_stage != last_stage:
            print(f"Epoch {epoch:3d}: {loss_fn.get_stage_info()}")
            last_stage = loss_fn._current_stage
    
    # 测试数值稳定性
    print("\n" + "=" * 60)
    print("测试数值稳定性")
    print("=" * 60)
    
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)
    
    for stage in range(loss_fn.num_stages):
        # 手动设置到每个阶段
        progress = sum(loss_fn.stage_ratios[:stage]) + 0.01
        loss_fn.set_training_progress(progress)
        
        loss = loss_fn(pred, target)
        print(f"阶段 {stage+1}: loss={loss.item():.6f}, "
              f"edge_weight={loss_fn._get_current_edge_weight():.3f}, "
              f"NaN={torch.isnan(loss).item()}")
    
    # 测试 num_stages=1（固定权重）
    print("\n" + "=" * 60)
    print("测试固定权重 (num_stages=1)")
    print("=" * 60)
    
    loss_fn_fixed = CombinedLoss(
        foreground_weight=10.0,
        edge_weight=0.5,
        num_stages=1
    )
    
    for epoch in [1, 50, 100]:
        progress = (epoch - 1) / 99
        loss_fn_fixed.set_training_progress(progress)
        loss = loss_fn_fixed(pred, target)
        print(f"Epoch {epoch:3d}: loss={loss.item():.6f}, "
              f"edge_weight={loss_fn_fixed._get_current_edge_weight():.3f}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_stage_weights()
