"""图像去噪网络对比系统 - 主程序入口

支持三种运行模式:
    - train:    训练指定的去噪模型
    - evaluate: 在测试集上评估已训练的模型
    - compare:  对比多个实验的结果

用法示例:
    python main.py train --config config.yaml
    python main.py evaluate --config config.yaml --checkpoint checkpoints/best_model.pth
    python main.py compare --experiments exp1 exp2 --output_dir ./comparison
"""

import argparse
import os
import sys
import time
import json

import torch
from torch.utils.data import DataLoader

from utils.config import ConfigManager, ConfigError
from utils.logger import Logger
from utils.visualization import Visualizer
from data.dataset import DenoisingDataset, DatasetError, split_dataset
from data.transforms import DataTransforms
from models import get_model, MODEL_REGISTRY
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from experiments.experiment_manager import ExperimentManager


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="图像去噪网络对比系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="运行模式")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="训练去噪模型")
    train_parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    train_parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")

    # ---- evaluate ----
    eval_parser = subparsers.add_parser("evaluate", help="评估已训练的模型")
    eval_parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    eval_parser.add_argument("--output_dir", type=str, default=None, help="评估结果输出目录")

    # ---- compare ----
    cmp_parser = subparsers.add_parser("compare", help="对比多个实验结果")
    cmp_parser.add_argument("--experiments", nargs="+", required=True, help="要对比的实验ID列表")
    cmp_parser.add_argument("--exp_dir", type=str, default="./experiments", help="实验根目录")
    cmp_parser.add_argument("--output_dir", type=str, default="./comparison", help="对比结果输出目录")

    return parser


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------

def cmd_train(args):
    """执行训练流程。"""
    logger = Logger.get_logger("main")

    # 1. 加载并验证配置
    config = ConfigManager.load_config(args.config)
    ConfigManager.validate_config(config)
    logger.info(f"配置已加载: {args.config}")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    exp_cfg = config["experiment"]

    # 2. 创建实验
    exp_manager = ExperimentManager(base_dir=exp_cfg.get("output_dir", "./experiments"))
    
    # 实验名称自动包含模型名
    exp_name = exp_cfg.get("name", "experiment")
    model_name = model_cfg["name"]
    if model_name.lower() not in exp_name.lower():
        exp_name = f"{model_name}_{exp_name}"
    
    exp_id = exp_manager.create_experiment(exp_name, config)
    exp_dir = os.path.join(exp_manager.base_dir, exp_id)
    logger.info(f"实验已创建: {exp_id}")

    # 3. 加载数据集
    patches_per_image = dataset_cfg.get("patches_per_image", 1)
    
    train_transform = DataTransforms.get_train_transforms(
        target_size=dataset_cfg.get("patch_size", 512),
        use_random_crop=dataset_cfg.get("use_random_crop", False),
        patches_per_image=patches_per_image
    )
    val_transform = DataTransforms.get_val_transforms(
        target_size=dataset_cfg.get("patch_size", 512)
    )

    # 支持多数据集合并
    if "datasets" in dataset_cfg:
        # 多数据集模式
        from data.dataset import create_multi_dataset
        dataset = create_multi_dataset(dataset_cfg["datasets"])
        logger.info(f"已加载 {len(dataset_cfg['datasets'])} 个数据集，总计 {len(dataset)} 对图像")
    else:
        # 单数据集模式
        dataset = DenoisingDataset(
            noisy_dir=dataset_cfg["noisy_dir"],
            clean_dir=dataset_cfg["clean_dir"],
        )
        logger.info(f"已加载数据集: {len(dataset)} 对图像")
    
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=dataset_cfg["train_split"],
        val_ratio=dataset_cfg["val_split"],
        test_ratio=dataset_cfg["test_split"],
    )

    # 为训练集应用 MultiPatchDataset（每张图返回多个patch）
    from data.dataset import MultiPatchDataset
    if patches_per_image > 1 and dataset_cfg.get("use_random_crop", False):
        train_ds = MultiPatchDataset(train_set, patches_per_image, train_transform)
        logger.info(f"训练集: {len(train_set)} 张图像 × {patches_per_image} patches = {len(train_ds)} 个训练样本")
    else:
        train_ds = _TransformSubset(train_set, train_transform)
        logger.info(f"训练集: {len(train_set)} 个样本")
    
    val_ds = _TransformSubset(val_set, val_transform)

    batch_size = dataset_cfg.get("batch_size", 16)
    num_workers = dataset_cfg.get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"数据加载器已创建: 训练={len(train_set)}, 验证={len(val_set)}, 测试={len(test_set)}")

    # 4. 创建模型
    model = get_model(model_cfg["name"], model_cfg.get("params", {}))
    param_count = sum(p.numel() for p in model.parameters())
    
    # 5. 打印训练配置摘要
    logger.info("=" * 80)
    logger.info("训练配置摘要")
    logger.info("=" * 80)
    logger.info(f"实验名称: {exp_cfg['name']}")
    logger.info(f"实验ID: {exp_id}")
    logger.info(f"输出目录: {exp_dir}")
    logger.info("-" * 80)
    logger.info(f"模型: {model_cfg['name'].upper()}")
    logger.info(f"参数量: {param_count:,}")
    loss_type = train_cfg.get("loss", {}).get("type", "")
    if loss_type:
        base_loss_name = model.get_loss_function().__class__.__name__
        logger.info(f"损失函数: {loss_type} (基础: {base_loss_name})")
    else:
        logger.info(f"损失函数: {model.get_loss_function().__class__.__name__}")
    logger.info("-" * 80)
    logger.info(f"数据集:")
    if "datasets" in dataset_cfg:
        for ds_cfg in dataset_cfg["datasets"]:
            logger.info(f"  - {ds_cfg.get('name', 'unknown')}: {ds_cfg.get('num_samples', 'all')} 张")
    logger.info(f"  总计: {len(dataset)} 对图像")
    logger.info(f"  训练集: {len(train_set)} 张图像")
    if patches_per_image > 1:
        logger.info(f"  每张图裁剪: {patches_per_image} 个 patch")
        logger.info(f"  训练样本总数: {len(train_ds)} 个 patch")
    logger.info(f"  验证集: {len(val_set)} 张图像")
    logger.info(f"  测试集: {len(test_set)} 张图像")
    logger.info(f"  Patch尺寸: {dataset_cfg.get('patch_size', 512)}×{dataset_cfg.get('patch_size', 512)}")
    logger.info(f"  随机裁剪: {'是' if dataset_cfg.get('use_random_crop', False) else '否'}")
    logger.info("-" * 80)
    logger.info(f"训练参数:")
    logger.info(f"  Epochs: {train_cfg.get('num_epochs', 100)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  学习率: {train_cfg.get('learning_rate', 0.001)}")
    logger.info(f"  优化器: {model.get_optimizer(0.001).__class__.__name__}")
    logger.info(f"  学习率调度: {train_cfg.get('scheduler', {}).get('type', 'ReduceLROnPlateau')}")
    logger.info(f"  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info("=" * 80)
    
    # 6. 训练
    train_cfg["checkpoint_dir"] = os.path.join(exp_dir, "checkpoints")
    trainer = Trainer(model, train_loader, val_loader, config=train_cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"从检查点恢复: {args.resume}")

    start_time = time.time()
    history = trainer.train(num_epochs=train_cfg.get("num_epochs", 100))
    training_time = time.time() - start_time

    # 7. 保存训练曲线
    curves_path = os.path.join(exp_dir, "training_curves.png")
    Visualizer.plot_training_curves(history, curves_path)

    # 8. 保存实验结果
    results = {
        "model_name": model_cfg["name"],
        "model_params": param_count,
        "training_time": training_time,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "best_val_psnr": trainer.best_val_psnr,
        "best_val_ssim": trainer.best_val_ssim,
        "history": history,
    }
    exp_manager.save_results(exp_id, results)

    logger.info(f"训练完成 - 耗时: {training_time:.1f}s, 最佳PSNR: {trainer.best_val_psnr:.4f}, 最佳SSIM: {trainer.best_val_ssim:.4f}")
    logger.info(f"实验目录: {exp_dir}")
    return exp_id


# ---------------------------------------------------------------------------
# Evaluate command
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    """执行评估流程。"""
    logger = Logger.get_logger("main")

    # 1. 加载配置
    config = ConfigManager.load_config(args.config)
    ConfigManager.validate_config(config)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    eval_cfg = config.get("evaluation", {})

    # 2. 加载数据集（仅需测试集）
    val_transform = DataTransforms.get_val_transforms(
        target_size=dataset_cfg.get("patch_size", 512)
    )

    # 支持多数据集合并
    if "datasets" in dataset_cfg:
        from data.dataset import create_multi_dataset
        dataset = create_multi_dataset(dataset_cfg["datasets"])
    else:
        dataset = DenoisingDataset(
            noisy_dir=dataset_cfg["noisy_dir"],
            clean_dir=dataset_cfg["clean_dir"],
        )
    
    _, _, test_set = split_dataset(
        dataset,
        train_ratio=dataset_cfg["train_split"],
        val_ratio=dataset_cfg["val_split"],
        test_ratio=dataset_cfg["test_split"],
    )

    test_ds = _TransformSubset(test_set, val_transform)
    batch_size = dataset_cfg.get("batch_size", 16)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"测试集大小: {len(test_set)}")

    # 3. 加载模型和检查点
    model = get_model(model_cfg["name"], model_cfg.get("params", {}))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"模型已加载: {args.checkpoint}")

    # 4. 评估
    evaluator = Evaluator(model, test_loader)
    results = evaluator.evaluate()

    # 5. 保存去噪样本
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval_results")
    num_samples = eval_cfg.get("num_samples", 10)
    if eval_cfg.get("save_samples", True):
        samples_dir = os.path.join(output_dir, "samples")
        evaluator.save_sample_results(samples_dir, num_samples=num_samples)

    # 6. 保存评估结果
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    eval_output = {
        "model_name": model_cfg["name"],
        "checkpoint": args.checkpoint,
        "test_metrics": results,
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)

    logger.info(f"评估完成 - PSNR: {results['psnr']['mean']:.4f}, SSIM: {results['ssim']['mean']:.4f}")
    logger.info(f"结果已保存: {results_path}")
    return results


# ---------------------------------------------------------------------------
# Compare command
# ---------------------------------------------------------------------------

def cmd_compare(args):
    """执行对比流程。"""
    logger = Logger.get_logger("main")

    exp_manager = ExperimentManager(base_dir=args.exp_dir)
    exp_ids = args.experiments

    # 1. 生成对比表格
    df = exp_manager.compare_experiments(exp_ids)
    logger.info(f"对比 {len(exp_ids)} 个实验:")
    print(df.to_string(index=False))

    # 2. 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)

    # JSON
    json_path = os.path.join(output_dir, "comparison.json")
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    # 3. 生成对比柱状图
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    Visualizer.plot_comparison_bar(df, chart_path)

    logger.info(f"对比结果已保存: {output_dir}")
    return df


# ---------------------------------------------------------------------------
# Helper: apply transform to a Subset
# ---------------------------------------------------------------------------

class _TransformSubset:
    """为 Subset 包装一个 transform，使其在 __getitem__ 时应用。"""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        noisy, clean = self.subset[idx]
        if self.transform is not None:
            from PIL import Image as PILImage
            import torchvision.transforms.functional as TF_util
            if isinstance(noisy, torch.Tensor):
                # 已经是 tensor，需要转回 PIL 再应用 transform
                noisy = TF_util.to_pil_image(noisy)
                clean = TF_util.to_pil_image(clean)
            return self.transform(noisy, clean)
        return noisy, clean

    def __len__(self):
        return len(self.subset)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "train":
            cmd_train(args)
        elif args.command == "evaluate":
            cmd_evaluate(args)
        elif args.command == "compare":
            cmd_compare(args)
        else:
            parser.print_help()
            sys.exit(1)
    except (ConfigError, DatasetError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练已被用户中断")
        sys.exit(0)


if __name__ == "__main__":
    main()
