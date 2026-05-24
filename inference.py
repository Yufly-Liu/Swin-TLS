"""推理脚本 - 将大图裁剪为 512x512 小块推理后再拼接

用法:
    python inference.py --input image.png --checkpoint_dir ./checkpoints --output ./results
    python inference.py --input assets/a.BMP --checkpoint_dir weight --region 900 193 800 800 --output ./results --gt assets\GT.bmp
"""

import argparse
import os
import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from models import get_model, MODEL_REGISTRY
from utils.config import ConfigManager


PATCH_SIZE = 512
OVERLAP = 64  # 拼接重叠区域大小


def load_image(path):
    """加载图像并转为 tensor [0, 1]"""
    img = Image.open(path).convert('RGB')
    img_tensor = TF.to_tensor(img)  # (C, H, W), [0, 1]
    return img, img_tensor


def tile_image_with_overlap(img_tensor, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """将图像裁剪为多个带重叠的 patch，使用边缘扩展避免拼接痕迹

    Args:
        img_tensor: (C, H, W)
        patch_size: 裁剪大小
        overlap: 重叠区域大小

    Returns:
        patches: list of (C, patch_size, patch_size)
        positions: list of (x, y, actual_h, actual_w)
    """
    C, H, W = img_tensor.shape
    stride = patch_size - overlap

    patches = []
    positions = []

    # 填充到 patch_size 的倍数
    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else 0
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else 0

    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))
    C, H, W = img_tensor.shape

    y = 0
    while y < H - patch_size + 1:
        x = 0
        while x < W - patch_size + 1:
            patch = img_tensor[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((x, y, patch_size, patch_size))
            x += stride
        y += stride

    return patches, positions, (pad_h, pad_w)


def untile_image(patches, positions, orig_h, orig_w, pad_info, overlap=OVERLAP):
    """将多个 patch 拼接回原始图像尺寸，使用权重平均处理重叠区域

    Args:
        patches: list of (C, H, W) tensors (所有 patch 都是 patch_size x patch_size)
        positions: list of (x, y, h, w)
        orig_h, orig_w: 原始图像尺寸（不含填充）
        pad_info: (pad_h, pad_w) 填充信息
        overlap: 重叠区域大小
    """
    pad_h, pad_w = pad_info
    patch_size = PATCH_SIZE

    # 输出尺寸 = 原始尺寸（不含填充）
    output = torch.zeros((3, orig_h, orig_w))
    weight_map = torch.zeros((3, orig_h, orig_w))

    # 使用渐变权重，距离边缘越近权重越小
    for patch, (x, y, h, w) in zip(patches, positions):
        weight = torch.ones((3, h, w))

        # 边缘区域渐变权重
        if overlap > 0 and h > overlap and w > overlap:
            fade_range = min(overlap, h // 2, w // 2)
            for i in range(fade_range):
                fade = i / fade_range
                weight[:, i, :] *= fade
                weight[:, h-1-i, :] *= fade
                weight[:, :, i] *= fade
                weight[:, :, w-1-i] *= fade

        # 只写入 orig_h x orig_w 范围内
        end_y = min(y + h, orig_h)
        end_x = min(x + w, orig_w)
        actual_h = end_y - y
        actual_w = end_x - x

        if actual_h > 0 and actual_w > 0:
            output[:, y:end_y, x:end_x] += patch[:, :actual_h, :actual_w] * weight[:, :actual_h, :actual_w]
            weight_map[:, y:end_y, x:end_x] += weight[:, :actual_h, :actual_w]

    # 平均重叠区域
    weight_map[weight_map == 0] = 1.0
    output = output / weight_map

    return output


def psnr_gray(pred, target, data_range=1.0):
    """计算灰度 PSNR"""
    # (C, H, W) -> (H, W)
    pred_gray = pred.mean(axis=0)
    target_gray = target.mean(axis=0)

    mse = np.mean((pred_gray - target_gray) ** 2)
    if mse == 0:
        return float('inf')

    max_val = data_range
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def ssim_gray(pred, target, data_range=1.0):
    """计算灰度 SSIM（简化版）"""
    pred_gray = pred.mean(axis=0)
    target_gray = target.mean(axis=0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    pred_gray = pred_gray.astype(np.float64)
    target_gray = target_gray.astype(np.float64)

    mu_pred = pred_gray.mean()
    mu_target = target_gray.mean()

    var_pred = pred_gray.var()
    var_target = target_gray.var()
    cov = np.mean((pred_gray - mu_pred) * (target_gray - mu_target))

    num = (2 * mu_pred * mu_target + C1) * (2 * cov + C2)
    den = (mu_pred**2 + mu_target**2 + C1) * (var_pred + var_target + C2)

    ssim = num / den
    return ssim


def psnr_color(pred, target, data_range=1.0):
    """计算彩色 PSNR（逐通道平均）"""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mse_r = np.mean((pred[:,:,0] - target[:,:,0]) ** 2)
    mse_g = np.mean((pred[:,:,1] - target[:,:,1]) ** 2)
    mse_b = np.mean((pred[:,:,2] - target[:,:,2]) ** 2)

    mse = (mse_r + mse_g + mse_b) / 3.0
    if mse == 0:
        return float('inf')

    max_val = data_range
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def ssim_color(pred, target, data_range=1.0):
    """计算彩色 SSIM（逐通道平均）"""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    ssim_r = ssim_gray(pred[:,:,0], target[:,:,0], data_range)
    ssim_g = ssim_gray(pred[:,:,1], target[:,:,1], data_range)
    ssim_b = ssim_gray(pred[:,:,2], target[:,:,2], data_range)

    return (ssim_r + ssim_g + ssim_b) / 3.0


def get_metrics(pred_np, target_np):
    """计算 PSNR, SSIM, MSE（去噪前后对比）

    注意：这是无参考的对比，仅衡量去噪平滑程度
    """
    psnr = psnr_color(pred_np, target_np)
    ssim = ssim_color(pred_np, target_np)
    mse = np.mean((pred_np - target_np) ** 2)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse
    }


def main():
    parser = argparse.ArgumentParser(description='使用多个模型权重去噪（大图裁剪推理）')
    parser.add_argument('--input', '-i', required=True, help='输入图像路径')
    parser.add_argument('--checkpoint_dir', '-c', required=True, help='模型权重目录')
    parser.add_argument('--output', '-o', default='./inference_results', help='输出目录')
    parser.add_argument('--region', nargs=4, type=int, default=None,
                        help='矩形区域 [x y width height]，不指定则处理整张图')
    parser.add_argument('--gt', '-g', default=None, help='真值图像路径（用于计算指标）')
    parser.add_argument('--config', default='configs/config_default_loss.yaml', help='配置文件')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--save_images', action='store_true', default=True, help='保存去噪图像')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1. 加载图像
    orig_pil, img_tensor = load_image(args.input)
    orig_full_w, orig_full_h = orig_pil.size
    orig_np_full = np.array(orig_pil).astype(np.float32) / 255.0

    # 确定处理区域（边界裁剪）
    if args.region:
        proc_x, proc_y, proc_w, proc_h = args.region
        # 边界裁剪，确保不超出图像范围
        proc_x = min(proc_x, orig_full_w - 1)
        proc_y = min(proc_y, orig_full_h - 1)
        proc_w = min(proc_w, orig_full_w - proc_x)
        proc_h = min(proc_h, orig_full_h - proc_y)
        img_tensor_proc = img_tensor[:, proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]
        proc_pil = orig_pil.crop((proc_x, proc_y, proc_x+proc_w, proc_y+proc_h))
        orig_np = np.array(proc_pil).astype(np.float32) / 255.0
        print(f"指定区域(已裁剪): ({proc_x}, {proc_y}) -> {proc_w}x{proc_h}")
    else:
        proc_x, proc_y, proc_w, proc_h = 0, 0, orig_full_w, orig_full_h
        img_tensor_proc = img_tensor
        orig_np = orig_np_full
        print(f"处理整张图: {orig_full_w}x{orig_full_h}")

    # 加载真值图像（用于计算指标）
    gt_np = None
    if args.gt:
        gt_pil, gt_tensor = load_image(args.gt)
        if args.region:
            gt_tensor = gt_tensor[:, proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]
        else:
            gt_tensor = gt_tensor[:, :proc_h, :proc_w]
        gt_np = gt_tensor.numpy()
        if gt_np.ndim == 3:
            gt_np = gt_np.transpose(1, 2, 0)
        gt_np = np.clip(gt_np, 0, 1)
        print(f"加载真值: {args.gt}")

    print(f"处理尺寸: {proc_w}x{proc_h}")
    print(f"模型输入尺寸: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"重叠区域: {OVERLAP}")

    # 2. 裁剪为 patch（带重叠）
    # 如果图像尺寸 <= patch_size，pad 到 patch_size
    if proc_w <= PATCH_SIZE and proc_h <= PATCH_SIZE:
        # pad 到 patch_size
        pad_h = PATCH_SIZE - proc_h
        pad_w = PATCH_SIZE - proc_w
        img_padded = torch.nn.functional.pad(img_tensor_proc, (0, pad_w, 0, pad_h))
        patches = [img_padded]
        positions = [(0, 0, PATCH_SIZE, PATCH_SIZE)]
        pad_info = (pad_h, pad_w)
        print(f"图像尺寸 < {PATCH_SIZE}，pad 到 {PATCH_SIZE}x{PATCH_SIZE}")
    else:
        patches, positions, pad_info = tile_image_with_overlap(img_tensor_proc, PATCH_SIZE, OVERLAP)
    print(f"裁剪为 {len(patches)} 个 {PATCH_SIZE}x{PATCH_SIZE} patch (重叠 {OVERLAP})")

    # 3. 找 checkpoint
    checkpoint_paths = glob.glob(os.path.join(args.checkpoint_dir, '**/*.pth'), recursive=True)
    checkpoint_paths = sorted([p for p in checkpoint_paths])

    if not checkpoint_paths:
        print(f"在 {args.checkpoint_dir} 中未找到 .pth 文件")
        return

    print(f"找到 {len(checkpoint_paths)} 个 checkpoint")

    # 4. 加载配置 - 优先从 checkpoint 读取，以确保模型结构一致
    config = ConfigManager.load_config(args.config)
    model_cfg = config['model']
    model_name = model_cfg['name']
    model_params = model_cfg.get('params', {}).copy()

    # 加载第一个 checkpoint 获取训练时的模型参数
    first_ckpt = torch.load(checkpoint_paths[0], map_location='cpu', weights_only=False)
    if 'config' in first_ckpt:
        ckpt_model_cfg = first_ckpt['config'].get('model', {})
        ckpt_params = ckpt_model_cfg.get('params', {})

        # 用 checkpoint 里的 SCUNet 参数覆盖配置
        if 'scunet_config' in ckpt_params:
            model_params['scunet_config'] = ckpt_params['scunet_config']
            print(f"    从 checkpoint 读取 scunet_config: {ckpt_params['scunet_config']}")
        if 'scunet_dim' in ckpt_params:
            model_params['scunet_dim'] = ckpt_params['scunet_dim']
        if 'scunet_input_resolution' in ckpt_params:
            model_params['scunet_input_resolution'] = ckpt_params['scunet_input_resolution']
        if 'scunet_drop_path_rate' in ckpt_params:
            model_params['scunet_drop_path_rate'] = ckpt_params['scunet_drop_path_rate']

    device = torch.device(args.device)
    base_model = get_model(model_name, model_params)
    base_model.to(device)

    # 5. 逐个 checkpoint 推理
    results = []
    denoised_images = []

    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).stem
        print(f"\n推理: {ckpt_name}")

        try:
            # 加载权重
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']

            # 检查 key 前缀
            sample_key = list(state_dict.keys())[0]
            print(f"    [DEBUG] checkpoint key sample: {sample_key}")

            # 从 checkpoint 读取模型配置，创建对应类型的模型
            ckpt_cfg = checkpoint.get('config', {})
            ckpt_model_cfg = ckpt_cfg.get('model', {})
            ckpt_model_name = ckpt_model_cfg.get('name', model_name)
            ckpt_params = ckpt_model_cfg.get('params', {}).copy()

            # 如果 config 里没有有效的模型名，从 state_dict key 推断
            if ckpt_model_name == 'scunet' or not ckpt_model_name:
                first_key = sample_key
                if first_key.startswith('dncnn.'):
                    ckpt_model_name = 'dncnn'
                elif first_key.startswith('body.') or first_key.startswith('pixel_unshuffle'):
                    ckpt_model_name = 'ffdnet'
                elif first_key.startswith('encoders.') or first_key.startswith('decoders.'):
                    ckpt_model_name = 'unet'
                elif first_key.startswith('restormer.'):
                    ckpt_model_name = 'restormer'
                elif first_key.startswith('swin_'):
                    ckpt_model_name = 'sunet'
                print(f"    [INFO] 从 key 推断模型: {ckpt_model_name}")

            # 从 config 读取对应模型的参数
            base_params = model_params.get('params', {}).copy()

            # 用 checkpoint 里的 scunet 参数覆盖（其他模型用 config 默认值）
            if ckpt_model_name == 'scunet':
                if 'scunet_config' in ckpt_params:
                    base_params['scunet_config'] = ckpt_params['scunet_config']
                if 'scunet_dim' in ckpt_params:
                    base_params['scunet_dim'] = ckpt_params['scunet_dim']
                if 'scunet_input_resolution' in ckpt_params:
                    base_params['scunet_input_resolution'] = ckpt_params['scunet_input_resolution']
                if 'scunet_drop_path_rate' in ckpt_params:
                    base_params['scunet_drop_path_rate'] = ckpt_params['scunet_drop_path_rate']

            # 为这个 checkpoint 创建对应的模型（使用 config 中的默认参数）
            infer_model = get_model(ckpt_model_name, base_params)
            infer_model.to(device)
            infer_model.eval()

            # 获取模型期望的 key
            model_keys = set(infer_model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())

            # 检查缺失的 key
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys

            if missing:
                print(f"    [WARNING] 缺失 {len(missing)} 个 key，例如: {list(missing)[:3]}")
            if unexpected:
                print(f"    [WARNING] 多余 {len(unexpected)} 个 key，例如: {list(unexpected)[:3]}")

            # 尝试 strict=True 加载
            try:
                infer_model.load_state_dict(state_dict, strict=True)
                print(f"    加载成功 (strict=True), 模型={ckpt_model_name}")
            except RuntimeError as e:
                err_msg = str(e)
                if "size mismatch" in err_msg:
                    # 形状不匹配，checkpoint 和模型的层数/通道数不一致，跳过
                    print(f"    形状不匹配: {err_msg.split('size mismatch')[1].split(',')[0].strip()}")
                    print(f"    跳过此 checkpoint (架构不匹配)")
                    continue
                print(f"    strict=True 失败，尝试 strict=False...")
                infer_model.load_state_dict(state_dict, strict=False)
                print(f"    已加载，忽略不匹配的 key")

            # 逐 patch 推理
            output_patches = []
            with torch.no_grad():
                for i, patch in enumerate(patches):
                    input_tensor = patch.unsqueeze(0).to(device)
                    output_patch = infer_model(input_tensor).squeeze(0).cpu()
                    output_patches.append(output_patch)

            # 拼接回原图尺寸 (proc_h, proc_w 是原始区域尺寸)
            output_full = untile_image(output_patches, positions, proc_h, proc_w, pad_info, OVERLAP)

            # 归一化到 [0, 1]，转 (H, W, C)
            output_np = output_full.numpy()
            output_np = np.clip(output_np, 0, 1)
            output_np = output_np.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

            # 转 PIL (H, W, C) -> (H, W, C) for display
            output_pil = Image.fromarray((output_np * 255).astype(np.uint8))

            # 保存图像
            if args.save_images:
                out_path = os.path.join(args.output, f"{ckpt_name}_denoised.png")
                output_pil.save(out_path)
                print(f"  保存: {out_path}")

            # 计算指标（如果提供了真值，用真值计算；否则用输入图）
            target_np = gt_np if gt_np is not None else orig_np
            metrics = get_metrics(output_np, target_np)
            print(f"  去噪前 -> 去噪后:")
            print(f"  去噪后 PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}, MSE: {metrics['mse']:.6f}")

            results.append({
                'checkpoint': ckpt_name,
                **metrics
            })

            denoised_images.append(output_pil)

            if args.save_images:
                out_path = os.path.join(args.output, f"{ckpt_name}_denoised.png")
                output_pil.save(out_path)
                print(f"  保存: {out_path}")

        except Exception as e:
            import traceback
            print(f"  错误: {e}")
            traceback.print_exc()
            continue

    # 6. 打印汇总表
    if results:
        print("\n" + "=" * 65)
        print("推理结果汇总")
        print("=" * 65)
        print(f"{'Checkpoint':<35} {'PSNR':>10} {'SSIM':>10} {'MSE':>12}")
        print("-" * 65)
        for r in results:
            print(f"{r['checkpoint']:<35} {r['psnr']:>10.2f} {r['ssim']:>10.4f} {r['mse']:>12.6f}")
        print("-" * 65)

        # 保存 CSV
        import csv
        csv_path = os.path.join(args.output, 'inference_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['checkpoint', 'psnr', 'ssim', 'mse'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV 已保存: {csv_path}")

        # 可视化对比
        n = len(denoised_images) + 1
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        # 如果指定了区域，使用裁剪后的区域图作为原图显示
        display_pil = proc_pil if args.region else orig_pil
        axes[0].imshow(display_pil)
        axes[0].set_title('Original (Noisy)', fontsize=12)
        axes[0].axis('off')

        for i, (img, metrics, name) in enumerate(zip(denoised_images, results, [r['checkpoint'] for r in results])):
            psnr_val = metrics['psnr']
            ssim_val = metrics['ssim']
            title = f"{name}"
            if not np.isnan(psnr_val):
                title += f"\nPSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f}"
            axes[i+1].set_title(title, fontsize=10)
            axes[i+1].axis('off')

        plt.tight_layout()
        chart_path = os.path.join(args.output, 'comparison.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比图已保存: {chart_path}")


if __name__ == '__main__':
    main()