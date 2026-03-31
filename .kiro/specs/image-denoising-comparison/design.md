# 设计文档

## 概述

本项目实现一个模块化的图像去噪网络对比系统，支持训练和评估多个典型的深度学习去噪模型。系统采用PyTorch框架，提供统一的接口来管理数据集、训练模型、评估性能和生成对比报告。

核心设计目标：
- 模块化架构，易于扩展新模型
- 统一的训练和评估流程
- 完整的实验管理和结果追踪
- 支持多种评估指标和可视化

## 架构

系统采用分层架构，主要包含以下模块：

```
image-denoising-comparison/
├── data/
│   ├── dataset.py          # 数据集加载和预处理
│   └── transforms.py       # 数据增强
├── models/
│   ├── base_model.py       # 模型基类
│   ├── dncnn.py           # DnCNN实现
│   ├── unet.py            # U-Net实现
│   ├── rednet.py          # RED-Net实现
│   ├── ffdnet.py          # FFDNet实现
│   ├── restormer.py       # Restormer实现
│   └── sunet.py           # SUNet实现
├── training/
│   ├── trainer.py         # 训练引擎
│   └── losses.py          # 损失函数
├── evaluation/
│   ├── evaluator.py       # 评估引擎
│   └── metrics.py         # 评估指标
├── utils/
│   ├── config.py          # 配置管理
│   ├── logger.py          # 日志记录
│   └── visualization.py   # 可视化工具
├── experiments/
│   └── experiment_manager.py  # 实验管理
└── main.py                # 主入口
```

### 数据流

1. 配置加载 → 数据集初始化 → 模型创建
2. 训练循环：数据加载 → 前向传播 → 损失计算 → 反向传播 → 参数更新
3. 验证循环：数据加载 → 前向传播 → 指标计算 → 检查点保存
4. 评估流程：模型加载 → 测试集推理 → 指标计算 → 结果保存
5. 对比生成：收集所有模型结果 → 生成对比表格和图表


## 组件和接口

### 1. 数据集模块 (data/)

#### DenoisingDataset
负责加载和管理图像对数据集。

```python
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir: str, clean_dir: str, transform=None):
        """
        参数:
            noisy_dir: 噪声图像目录
            clean_dir: 干净图像目录
            transform: 数据增强变换
        """
        pass
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """返回 (noisy_image, clean_image) 对"""
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
```

#### DataTransforms
提供数据增强和预处理功能。

```python
class DataTransforms:
    @staticmethod
    def get_train_transforms() -> Compose:
        """返回训练时的数据增强（翻转、旋转、光照变化）"""
        pass
    
    @staticmethod
    def get_val_transforms() -> Compose:
        """返回验证/测试时的预处理（仅归一化）"""
        pass
```

### 2. 模型模块 (models/)

#### BaseModel
所有去噪模型的抽象基类。

```python
class BaseModel(nn.Module):
    def __init__(self, config: dict):
        """初始化模型"""
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        pass
    
    def get_loss_function(self) -> nn.Module:
        """返回该模型官方推荐的损失函数"""
        pass
    
    def get_optimizer(self, lr: float) -> Optimizer:
        """返回优化器"""
        pass
```

#### 具体模型实现

每个模型继承BaseModel并实现具体架构。所有模型实现应该直接使用官方论文和代码仓库中的架构：

- **DnCNN**: 使用官方架构（17层卷积网络，批归一化和ReLU激活），损失函数为MSE
  - 参考: Zhang et al. "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
  
- **U-Net**: 使用官方架构（编码器-解码器结构，带跳跃连接），损失函数为MSE
  - 参考: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
  
- **RED-Net**: 使用官方架构（对称的卷积-反卷积网络，带跳跃连接），损失函数为MSE
  - 参考: Mao et al. "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections"
  
- **FFDNet**: 使用官方架构（灵活的去噪网络，支持噪声级别输入），损失函数为L1
  - 参考: Zhang et al. "FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising"
  
- **Restormer**: 使用官方架构（基于Transformer的恢复网络，多头自注意力），损失函数为Charbonnier Loss
  - 参考: Zamir et al. "Restormer: Efficient Transformer for High-Resolution Image Restoration"
  
- **SUNet**: 使用官方架构（Swin Transformer U-Net，结合局部和全局特征），损失函数为L1
  - 参考: 官方SUNet实现

注意：实现时应该参考各模型的官方代码仓库，确保架构细节（层数、通道数、激活函数等）与官方版本一致。


### 3. 训练模块 (training/)

#### Trainer
管理模型训练流程。

```python
class Trainer:
    def __init__(self, model: BaseModel, train_loader: DataLoader, 
                 val_loader: DataLoader, config: dict):
        """
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置（学习率、epoch数等）
        """
        self.model = model
        self.loss_fn = model.get_loss_function()
        self.optimizer = model.get_optimizer(config['lr'])
        self.scheduler = self._create_scheduler(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_epoch(self) -> float:
        """训练一个epoch，返回平均损失"""
        pass
    
    def validate(self) -> Dict[str, float]:
        """在验证集上评估，返回指标字典"""
        pass
    
    def train(self, num_epochs: int) -> None:
        """完整训练流程"""
        pass
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """保存模型检查点"""
        pass
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点恢复训练"""
        pass
```

### 4. 评估模块 (evaluation/)

#### Evaluator
评估训练好的模型性能。

```python
class Evaluator:
    def __init__(self, model: BaseModel, test_loader: DataLoader):
        """
        参数:
            model: 要评估的模型
            test_loader: 测试数据加载器
        """
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate(self) -> Dict[str, float]:
        """
        在测试集上评估模型
        返回: {'psnr': float, 'ssim': float, 'mse': float}
        """
        pass
    
    def save_sample_results(self, output_dir: str, num_samples: int = 10) -> None:
        """保存去噪结果样本图像"""
        pass
```

#### Metrics
实现各种评估指标。

```python
class Metrics:
    @staticmethod
    def calculate_psnr(img1: Tensor, img2: Tensor) -> float:
        """计算峰值信噪比"""
        pass
    
    @staticmethod
    def calculate_ssim(img1: Tensor, img2: Tensor) -> float:
        """计算结构相似性指数"""
        pass
    
    @staticmethod
    def calculate_mse(img1: Tensor, img2: Tensor) -> float:
        """计算均方误差"""
        pass
```


### 5. 实验管理模块 (experiments/)

#### ExperimentManager
管理实验运行和结果追踪。

```python
class ExperimentManager:
    def __init__(self, base_dir: str = './experiments'):
        """
        参数:
            base_dir: 实验输出根目录
        """
        self.base_dir = base_dir
    
    def create_experiment(self, name: str, config: dict) -> str:
        """
        创建新实验
        返回: 实验ID
        """
        pass
    
    def save_results(self, exp_id: str, results: dict) -> None:
        """保存实验结果"""
        pass
    
    def load_results(self, exp_id: str) -> dict:
        """加载实验结果"""
        pass
    
    def list_experiments(self) -> List[dict]:
        """列出所有实验"""
        pass
    
    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """对比多个实验结果"""
        pass
```

### 6. 工具模块 (utils/)

#### ConfigManager
管理配置文件。

```python
class ConfigManager:
    @staticmethod
    def load_config(path: str) -> dict:
        """加载YAML配置文件"""
        pass
    
    @staticmethod
    def validate_config(config: dict) -> bool:
        """验证配置完整性"""
        pass
    
    @staticmethod
    def save_config(config: dict, path: str) -> None:
        """保存配置文件"""
        pass
```

#### Visualizer
生成可视化图表。

```python
class Visualizer:
    @staticmethod
    def plot_training_curves(history: dict, save_path: str) -> None:
        """绘制训练损失和验证指标曲线"""
        pass
    
    @staticmethod
    def plot_comparison_bar(results: pd.DataFrame, save_path: str) -> None:
        """绘制模型对比柱状图"""
        pass
    
    @staticmethod
    def save_image_comparison(noisy: Tensor, denoised: Tensor, 
                             clean: Tensor, save_path: str) -> None:
        """保存去噪前后对比图"""
        pass
```


## 数据模型

### 配置文件结构 (config.yaml)

```yaml
# 数据集配置
dataset:
  noisy_dir: "./data/noisy"
  clean_dir: "./data/clean"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  batch_size: 16
  num_workers: 4

# 模型配置
model:
  name: "dncnn"  # 可选: dncnn, unet, rednet, ffdnet, restormer, sunet
  params:
    in_channels: 3
    out_channels: 3
    # 模型特定参数

# 训练配置
training:
  num_epochs: 100
  learning_rate: 0.001
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 10
    factor: 0.5
  early_stopping:
    patience: 20
  checkpoint_dir: "./checkpoints"
  save_frequency: 5

# 评估配置
evaluation:
  metrics: ["psnr", "ssim", "mse"]
  save_samples: true
  num_samples: 10

# 实验配置
experiment:
  name: "exp_001"
  description: "Baseline DnCNN training"
  output_dir: "./experiments"
```

### 训练历史数据结构

```python
{
    "epoch": [1, 2, 3, ...],
    "train_loss": [0.05, 0.04, 0.03, ...],
    "val_psnr": [28.5, 29.2, 30.1, ...],
    "val_ssim": [0.85, 0.87, 0.89, ...],
    "learning_rate": [0.001, 0.001, 0.0005, ...]
}
```

### 评估结果数据结构

```python
{
    "model_name": "dncnn",
    "test_metrics": {
        "psnr": {"mean": 30.5, "std": 2.1},
        "ssim": {"mean": 0.89, "std": 0.05},
        "mse": {"mean": 0.002, "std": 0.0005}
    },
    "training_time": 3600.5,  # 秒
    "inference_time": 0.05,   # 每张图像的秒数
    "model_params": 556032,   # 参数量
    "timestamp": "2024-01-15T10:30:00"
}
```

### 对比报告数据结构

```python
{
    "experiments": [
        {
            "exp_id": "exp_001",
            "model": "dncnn",
            "psnr": 30.5,
            "ssim": 0.89,
            "mse": 0.002,
            "params": 556032,
            "train_time": 3600.5,
            "inference_time": 0.05
        },
        # 其他模型...
    ],
    "best_model": {
        "by_psnr": "restormer",
        "by_ssim": "restormer",
        "by_speed": "dncnn"
    }
}
```


## 正确性属性

属性是一种特征或行为，应该在系统的所有有效执行中保持为真——本质上是关于系统应该做什么的形式化陈述。属性作为人类可读规范和机器可验证正确性保证之间的桥梁。

### 数据管理属性

**属性 1: 数据集加载完整性**
*对于任意*有效的数据集目录，加载后的图像对数量应该等于目录中配对图像的数量，且每个噪声图像都有对应的干净图像。
**验证需求: 1.1, 1.2**

**属性 2: 数据集划分不重叠性**
*对于任意*数据集和划分比例，划分后的训练集、验证集和测试集应该没有重叠，且三个子集的大小之和应该等于原始数据集大小。
**验证需求: 1.3**

**属性 3: 数据增强保持配对关系**
*对于任意*图像对，应用相同的数据增强变换后，噪声图像和干净图像应该保持空间对应关系（相同的翻转、旋转等）。
**验证需求: 1.5**

**属性 4: 图像预处理输出一致性**
*对于任意*输入图像，预处理后的输出应该具有指定的尺寸和值范围（通常是[0, 1]或[-1, 1]）。
**验证需求: 1.4**

### 模型架构属性

**属性 5: 模型接口一致性**
*对于所有*实现的模型（DnCNN、U-Net、RED-Net、FFDNet、Restormer、SUNet），每个模型都应该实现BaseModel接口的所有必需方法（forward、get_loss_function、get_optimizer）。
**验证需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

**属性 6: 模型输入输出形状一致性**
*对于任意*模型和输入张量，模型的输出形状应该与输入形状相同（去噪任务的特性）。
**验证需求: 2.1-2.6**

**属性 7: 模型超参数可配置性**
*对于任意*支持的超参数配置，使用不同超参数初始化的模型应该具有不同的架构特征（如层数、通道数）。
**验证需求: 2.8**

### 训练流程属性

**属性 8: 训练检查点往返一致性**
*对于任意*训练状态，保存检查点后再加载，恢复的训练状态（epoch、优化器状态、模型参数）应该与保存前完全一致。
**验证需求: 3.7**

**属性 9: 验证指标记录完整性**
*对于任意*训练过程，每个epoch后的训练历史应该包含该epoch的损失值、验证指标和学习率。
**验证需求: 3.2, 3.5**

**属性 10: 学习率调度单调性**
*对于任意*使用ReduceLROnPlateau策略的训练过程，当验证指标在patience个epoch内没有改善时，学习率应该按照factor比例减小。
**验证需求: 3.3**

**属性 11: 最佳模型保存正确性**
*对于任意*训练过程，保存的最佳模型检查点应该对应验证集上性能最好的epoch。
**验证需求: 3.4**


### 评估指标属性

**属性 12: PSNR计算正确性**
*对于任意*两张相同的图像，PSNR值应该趋向无穷大；对于完全不同的图像，PSNR值应该较低。PSNR计算应该满足：PSNR = 10 * log10(MAX^2 / MSE)。
**验证需求: 4.1**

**属性 13: SSIM计算边界性**
*对于任意*两张图像，SSIM值应该在[0, 1]范围内，且两张相同图像的SSIM值应该等于1。
**验证需求: 4.2**

**属性 14: MSE计算正确性**
*对于任意*两张相同的图像，MSE值应该等于0；MSE值应该始终非负。
**验证需求: 4.3**

**属性 15: 评估结果统计完整性**
*对于任意*测试集评估结果，返回的指标字典应该包含每个指标的均值和标准差。
**验证需求: 4.6**

**属性 16: 样本图像保存数量一致性**
*对于任意*评估过程，如果指定保存N个样本，则应该生成恰好N个去噪结果图像文件。
**验证需求: 4.5**

### 对比和报告属性

**属性 17: 对比表格完整性**
*对于任意*一组模型评估结果，生成的对比表格应该包含所有模型的所有指标（PSNR、SSIM、MSE、参数量、训练时间、推理时间）。
**验证需求: 5.1, 5.3**

**属性 18: 对比数据序列化往返一致性**
*对于任意*对比报告数据，导出为JSON/CSV后再导入，数据内容应该保持一致（数值精度在可接受范围内）。
**验证需求: 5.5**

**属性 19: 可视化输出文件存在性**
*对于任意*对比过程，应该生成所有必需的可视化文件（对比柱状图、训练曲线图、去噪结果对比图）。
**验证需求: 5.2, 5.4, 8.1, 8.2, 8.3, 8.4**

### 配置管理属性

**属性 20: 配置验证完整性**
*对于任意*缺少必需参数的配置文件，验证函数应该返回False并指出缺失的参数。
**验证需求: 6.2, 6.3**

**属性 21: 配置文件往返一致性**
*对于任意*有效的配置字典，保存为文件后再加载，配置内容应该保持一致。
**验证需求: 6.1, 6.5**

### 实验管理属性

**属性 22: 实验ID唯一性**
*对于任意*一系列实验创建操作，每个实验应该获得唯一的实验ID，且对应的输出目录应该被成功创建。
**验证需求: 7.1**

**属性 23: 实验输出完整性**
*对于任意*完成的实验，实验目录应该包含所有必需的文件（配置副本、模型检查点、训练日志、评估结果、元数据、总结报告）。
**验证需求: 7.2, 7.3, 7.4**

**属性 24: 实验查询一致性**
*对于任意*已创建的实验集合，查询历史实验应该返回所有已创建的实验，且每个实验的元数据应该与创建时保存的一致。
**验证需求: 7.5**


## 错误处理

### 数据加载错误

1. **缺失图像对**: 当噪声图像没有对应的干净图像时，抛出`DatasetError`并列出缺失的配对
2. **图像格式错误**: 当图像文件无法读取时，记录警告并跳过该图像对
3. **空数据集**: 当数据集目录为空时，抛出`DatasetError`并提示用户检查路径

### 模型错误

1. **未知模型类型**: 当配置中指定的模型名称不在支持列表中时，抛出`ModelNotFoundError`并列出支持的模型
2. **模型初始化失败**: 当模型参数不合法时，抛出`ModelInitError`并说明参数要求
3. **内存不足**: 当GPU内存不足时，自动降级到CPU并记录警告

### 训练错误

1. **检查点加载失败**: 当检查点文件损坏或不兼容时，抛出`CheckpointError`并提示从头开始训练
2. **NaN损失**: 当训练过程中出现NaN损失时，停止训练并保存当前状态，建议降低学习率
3. **磁盘空间不足**: 当保存检查点时磁盘空间不足，抛出`IOError`并保留最近的检查点

### 配置错误

1. **配置文件不存在**: 抛出`FileNotFoundError`并提示正确的配置文件路径
2. **配置格式错误**: 抛出`ConfigError`并指出具体的格式问题（如YAML语法错误）
3. **缺失必需参数**: 抛出`ConfigError`并列出所有缺失的参数

### 评估错误

1. **模型文件不存在**: 当尝试加载不存在的模型检查点时，抛出`FileNotFoundError`
2. **测试集为空**: 当测试集没有数据时，抛出`DatasetError`
3. **指标计算失败**: 当图像尺寸不匹配导致指标无法计算时，抛出`MetricError`

## 测试策略

### 双重测试方法

本项目采用单元测试和基于属性的测试相结合的方法：

- **单元测试**: 验证特定示例、边缘情况和错误条件
- **属性测试**: 通过随机化输入验证通用属性，确保全面覆盖

单元测试有助于捕获具体的bug，而属性测试验证一般正确性。两者互补，共同提供全面的测试覆盖。

### 单元测试重点

单元测试应该专注于：
- 特定示例，展示正确行为
- 组件之间的集成点
- 边缘情况和错误条件

避免编写过多的单元测试——基于属性的测试已经处理了大量输入的覆盖。

### 基于属性的测试配置

- 使用`pytest-hypothesis`作为Python的属性测试库
- 每个属性测试最少运行100次迭代（由于随机化）
- 每个测试必须引用其设计文档中的属性
- 标签格式: **Feature: image-denoising-comparison, Property {number}: {property_text}**
- 每个正确性属性必须由单个基于属性的测试实现

### 测试组织

```
tests/
├── unit/
│   ├── test_dataset.py          # 数据集单元测试
│   ├── test_models.py           # 模型单元测试
│   ├── test_trainer.py          # 训练器单元测试
│   ├── test_evaluator.py        # 评估器单元测试
│   └── test_config.py           # 配置管理单元测试
├── property/
│   ├── test_data_properties.py      # 数据管理属性测试
│   ├── test_model_properties.py     # 模型架构属性测试
│   ├── test_training_properties.py  # 训练流程属性测试
│   ├── test_eval_properties.py      # 评估指标属性测试
│   └── test_experiment_properties.py # 实验管理属性测试
└── integration/
    ├── test_end_to_end.py       # 端到端集成测试
    └── test_comparison.py       # 模型对比集成测试
```

### 测试数据

- 使用小型合成数据集进行快速测试（10-20张图像对）
- 使用固定随机种子确保测试可重现
- 为属性测试生成各种尺寸和内容的随机图像
- 模拟各种配置和超参数组合

### 持续集成

- 所有测试应该在CI/CD管道中自动运行
- 单元测试和属性测试都必须通过才能合并代码
- 集成测试可以在夜间构建中运行（耗时较长）
- 保持测试覆盖率在80%以上
