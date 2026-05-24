# Swin-TLS: Temporal Line-Shifting with Swin Transformer for 3D Reconstruction under Strong Interreflection

**Swin-TLS** is a modular deep learning processing platform developed to address the **strong interreflection** issue on highly reflective metal surfaces (e.g., deep holes, grooves) in Fringe Projection Profilometry (FPP). This project focuses entirely on the algorithmic and data-processing level, providing an end-to-end image restoration pipeline to recover pure direct radiation components from contaminated temporal line-shifting fringe images. The system supports training, evaluating, and comparing multiple classic and state-of-the-art denoising networks.

<p align="center">
  <img src="assets/display.png" width="800">
</p>

<p align="center">
  <img src="assets/compare.png" width="800">
  <img src="assets/a.gif" width="300">
</p>

## ✨ Core Features

- **Optimized for Line-Shifting Structured Light**: By utilizing the sparse characteristics of binary multi-line patterns, the system effectively frames interreflection suppression as a deep learning-based structured noise removal task. This algorithmic choice completely bypasses the large-scale illumination aliasing common in global coding methods (e.g., Gray codes).
- **Hybrid Dataset Support (Virtual & Real)**:
  - **Real-captured Data**: A high-fidelity non-reflection Ground Truth (GT) acquisition strategy designed based on the reflection uniqueness of convex surfaces.
  - **Simulated Data**: Large-scale pixel-aligned simulation datasets generated using the **Blender Cycles** path-tracing renderer by precisely controlling the number of light bounces to decouple direct and indirect illumination.
- **Custom Composite Loss Function**: Designed for extreme class imbalance in structured light restoration (predominantly dark background, sparse bright lines). The strategy uses a **Composite Loss (Model Default + WeightedMSE + Edge-aware + Focal)** with phased weight adjustments to prevent detail loss and "brightness collapse" caused by over-suppressing large-scale highlights.
- **Powerful Model Comparison Matrix**: Built-in support for 4 classic and frontier denoising models: **DnCNN, U-Net, FFDNet, and SUNet**. It particularly validates the superior performance of **SUNet** in modeling long-range spatial dependencies and handling high-resolution structured noise.
- **Comprehensive Engineering Pipeline**: Supports multi-dataset merged training, oversampling/undersampling, Patch-based training, data augmentation, and integrated TensorBoard monitoring with checkpoint resume functionality.

## 🧰 Supported Models

| **Model Architecture** | **Features**                                                 | **Default Loss** | **Reference**            |
| ---------------------- | ------------------------------------------------------------ | ---------------- | ------------------------ |
| **DnCNN**              | 17-layer CNN, residual learning; stable for Gaussian noise.  | MSE              | Zhang et al., 2017       |
| **U-Net**              | Encoder-decoder with skip connections; preserves multi-scale features. | MSE              | Ronneberger et al., 2015 |
| **FFDNet**             | Noise-level-aware; high computational efficiency.            | L1               | Zhang et al., 2018       |
| **RED-Net**            | Symmetric encoder-decoder with residual learning.            | MSE              | Mao et al., 2016         |
| **Restormer**          | Transformer-based restoration with attention mechanisms.    | Charbonnier      | Zamir et al., 2021       |
| **SUNet**              | Swin Transformer U-Net; combines global modeling with local refinement. | L1               | Official Impl.           |
| **SCUNet**             | **(Recommended)** Structured Crowd U-Net; hybrid Swin + Conv blocks. | L1               | SCUNet-main              |

**Note:** SUNet and SCUNet are recommended for structured noise (interreflection patterns). SCUNet has ~9.6M parameters and uses a hybrid Swin-Transformer + Convolution architecture.

## 🚀 Installation

Bash

```
# Clone the repository
git clone <repo-url>
cd Swin-TLS

# Create conda environment
conda create -n Swin-TLS python=3.10
conda activate Swin-TLS

# Install PyTorch (CUDA 11.8, optimized for Quadro RTX 5000)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

*Key dependencies: PyTorch 2.7.1, TensorBoard 2.20, scikit-image 0.25, matplotlib 3.10, PyYAML 6.0, tqdm 4.67.*

## 📥 Dataset

Training data is not included in the repository due to size. Download from cloud drive:

**Link**: [Baidu Netdisk]() | **Code / 提取码**: `XXXX`

The dataset covers metal workpieces with various surface roughness (planar milling, end milling, grinding). Supported formats: PNG, JPG, BMP, TIFF.

After downloading, extract to `data/`:
```
data/
├── aluminum/          # Aluminum 6061 (Real-captured, roughness 0.8~3.2)
│   ├── input/         # Interreflection-polluted images
│   └── target/        # Clean images (via differential strategy)
├── iron/              # 45# Carbon Steel
│   ├── input/
│   └── target/
└── Synthesis/         # Blender-rendered (Complex geometry)
    ├── input/         # Reflection: ON
    └── target/        # Reflection: OFF
```

*Note: During training, images (e.g., 2048x1376) are randomly cropped into 512x512 patches.*

## ⚙️ Configuration

The project uses YAML configuration files. You can switch models by modifying `model.name`.

| **Config File**                     | **Description**                                              |
| ----------------------------------- | ------------------------------------------------------------ |
| `configs/config_combined_loss.yaml` | **Combined Loss (Recommended)**: Model Default + WeightedMSE + Edge + Focal. Optimized for extreme class imbalance. |
| `configs/config_default_loss.yaml`  | Uses the model's standard loss function (e.g., MSE or L1).   |
| `configs/config_test_loss.yaml`     | For quick testing with small samples.                        |

**Switch models with one line:**

YAML

```
model:
  name: "sunet"  # Options: dncnn, ffdnet, sunet, unet
```

## 💻 Usage

**Training**

```bash
# Train with Combined Loss (Recommended for fine-line restoration)
python main.py train --config configs/config_combined_loss.yaml

# Train with default loss
python main.py train --config configs/config_default_loss.yaml

# Resume training from a checkpoint
python main.py train --config configs/config_combined_loss.yaml --resume experiments/<exp_id>/checkpoints/checkpoint_epoch_25.pth
```

**Inference (Large Image Tiling)**

```bash
# Denoise an image with all checkpoints in the directory
python inference.py --input path/to/image.png --checkpoint_dir ./checkpoints --output ./results

# Specify a region [x y width height]
python inference.py --input path/to/image.png --checkpoint_dir ./checkpoints --region 0 0 512 512

# With ground truth for metric calculation
python inference.py --input path/to/image.png --checkpoint_dir ./checkpoints --gt path/to/gt.png
```

**Data Augmentation & LR Strategy**

- **Augmentation**: Random cropping (512x512), random horizontal/vertical flipping, and random 90° rotation.
- **LR Scheduler**: Default uses `CosineAnnealingLR` decaying to `eta_min: 0.000001`.

## 🎯 Loss Function Guide

The project supports multiple loss functions optimized for structured light restoration (dark background + sparse bright lines):

| Loss Type | Description | Best For |
|-----------|-------------|----------|
| `combined` | Model Default + WeightedMSE + Edge + Focal | **Recommended** for fine-line restoration |
| `weighted_mse` | Spatial weighting (foreground 15x) | Quick verification |
| `edge_preserving` | MSE + Sobel edge loss | Sharp line edges |
| `focal_mse` | Adaptive focusing on hard samples | Mixed difficulty |
| `perceptual_weighted` | Auto-threshold per image | Varying brightness |

**Combined Loss Parameters:**

```yaml
training:
  loss:
    type: "combined"
    params:
      foreground_weight: 10.0    # Line region weight multiplier
      edge_weight: 0.5            # Edge loss weight
      focal_gamma: 2.0            # Focal focusing parameter
      use_edge: true              # Enable edge loss
      use_focal: true            # Enable focal loss
      num_stages: 3              # Phased weight adjustment
```

**Key insight**: Standard MSE treats all pixels equally. Since dark background dominates (~90%), the model learns to just remove noise without restoring lines. Weighted MSE amplifies line region error by 10-15x, forcing the model to prioritize line recovery.

## 📊 Performance & 3D Reconstruction Results

Key metrics on the test set using the **Combined Loss** strategy:

- **SUNet\***: PSNR 25.49 dB / SSIM 0.9212 (Best performing)
- **U-Net\***: PSNR 23.69 dB / SSIM 0.8422

By inputting the network's output ("cleaned" images) into the multi-frequency heterodyne reconstruction pipeline, 3D point cloud accuracy is significantly improved. In standard metal step-block measurements, the height measurement error was reduced from **0.081mm to 0.021mm** (a **74% reduction**).

## 📂 Project Structure

Plaintext

```
├── main.py                    # Main entry (train/evaluate/compare)
├── inference.py               # Large image inference with tiling/overlap
├── configs/                   # Configuration files