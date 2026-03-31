# 需求文档

## 简介

本项目旨在实现一个图像去噪和图像恢复网络对比系统。系统将训练多个典型的深度学习网络（如DnCNN、U-Net、RED-Net等），并对比它们在图像去噪任务上的性能表现。用户已准备好包含噪声图像和干净图像对的数据集，系统需要提供完整的训练、评估和对比功能。

## 术语表

- **System**: 图像去噪对比系统
- **Dataset**: 包含噪声图像和干净图像对的训练数据集
- **Model**: 深度学习去噪网络模型（如DnCNN、U-Net、RED-Net等）
- **Training_Engine**: 负责模型训练的组件
- **Evaluation_Engine**: 负责模型评估的组件
- **Metrics**: 评估指标（PSNR、SSIM、MSE等）
- **Checkpoint**: 训练过程中保存的模型状态
- **Comparison_Report**: 多个模型性能对比报告

## 需求

### 需求 1：数据集管理

**用户故事：** 作为研究人员，我想要加载和管理图像数据集，以便能够训练和评估去噪模型。

#### 验收标准

1. WHEN 提供数据集路径时，THE System SHALL 加载所有噪声图像和对应的干净图像对
2. WHEN 加载数据集时，THE System SHALL 验证每个噪声图像都有对应的干净图像
3. THE System SHALL 将数据集划分为训练集、验证集和测试集
4. WHEN 图像尺寸不一致时，THE System SHALL 提供图像预处理功能（裁剪、缩放、归一化）
5. THE System SHALL 支持批量加载和数据增强（翻转、旋转、明暗光照变化等）

### 需求 2：模型架构实现

**用户故事：** 作为研究人员，我想要实现多个典型的去噪网络架构，以便进行性能对比。

#### 验收标准

1. THE System SHALL 实现 DnCNN 网络架构
2. THE System SHALL 实现 U-Net 网络架构
3. THE System SHALL 实现 RED-Net 网络架构
4. THE System SHALL 实现 FFDNet 网络架构
5. THE System SHALL 实现 Restormer 网络架构
6. THE System SHALL 实现 SUNet 网络架构
7. WHERE 用户需要添加新模型时，THE System SHALL 提供统一的模型接口
8. WHEN 初始化模型时，THE System SHALL 支持自定义超参数（层数、通道数等）

### 需求 3：模型训练

**用户故事：** 作为研究人员，我想要训练去噪模型，以便学习从噪声图像恢复干净图像的映射关系。

#### 验收标准

1. WHEN 开始训练时，THE Training_Engine SHALL 使用每个模型官方推荐的损失函数
2. WHEN 训练过程中，THE Training_Engine SHALL 在每个epoch后计算验证集上的性能指标
3. THE Training_Engine SHALL 支持学习率调度策略（StepLR、ReduceLROnPlateau等）
4. WHEN 验证性能提升时，THE Training_Engine SHALL 自动保存最佳模型检查点
5. THE Training_Engine SHALL 记录训练日志（损失值、学习率、指标等）
6. WHERE 支持GPU时，THE Training_Engine SHALL 使用GPU加速训练
7. WHEN 训练中断时，THE Training_Engine SHALL 支持从检查点恢复训练

### 需求 4：模型评估

**用户故事：** 作为研究人员，我想要评估训练好的模型性能，以便了解模型的去噪效果。

#### 验收标准

1. WHEN 评估模型时，THE Evaluation_Engine SHALL 计算 PSNR（峰值信噪比）指标
2. WHEN 评估模型时，THE Evaluation_Engine SHALL 计算 SSIM（结构相似性）指标
3. WHEN 评估模型时，THE Evaluation_Engine SHALL 计算 MSE（均方误差）指标
4. THE Evaluation_Engine SHALL 在测试集上评估模型性能
5. WHEN 评估完成时，THE Evaluation_Engine SHALL 保存去噪后的图像样本
6. THE Evaluation_Engine SHALL 计算每个模型在测试集上的平均指标和标准差

### 需求 5：模型对比

**用户故事：** 作为研究人员，我想要对比多个模型的性能，以便选择最佳的去噪方案。

#### 验收标准

1. WHEN 对比多个模型时，THE System SHALL 生成包含所有模型指标的对比表格
2. THE System SHALL 生成可视化对比图表（柱状图、折线图等）
3. WHEN 生成对比报告时，THE System SHALL 包含训练时间、推理速度和模型参数量信息
4. THE System SHALL 保存并排显示不同模型的去噪结果图像
5. THE Comparison_Report SHALL 以结构化格式（JSON、CSV）导出对比数据

### 需求 6：配置管理

**用户故事：** 作为研究人员，我想要通过配置文件管理实验参数，以便轻松复现和调整实验。

#### 验收标准

1. THE System SHALL 支持通过配置文件指定数据集路径、模型类型和训练参数
2. WHEN 读取配置文件时，THE System SHALL 验证所有必需参数是否存在
3. IF 配置文件格式错误，THEN THE System SHALL 返回清晰的错误信息
4. THE System SHALL 支持 YAML 或 JSON 格式的配置文件
5. WHEN 开始实验时，THE System SHALL 保存使用的配置文件副本以便复现

### 需求 7：实验管理

**用户故事：** 作为研究人员，我想要管理多个实验运行，以便追踪和对比不同配置的结果。

#### 验收标准

1. WHEN 开始新实验时，THE System SHALL 创建唯一的实验ID和输出目录
2. THE System SHALL 在实验目录中保存模型检查点、日志和评估结果
3. THE System SHALL 记录实验的元数据（开始时间、配置、硬件信息等）
4. WHEN 实验完成时，THE System SHALL 生成实验总结报告
5. THE System SHALL 支持列出和查询历史实验记录

### 需求 8：可视化

**用户故事：** 作为研究人员，我想要可视化训练过程和结果，以便直观了解模型性能。

#### 验收标准

1. WHEN 训练过程中，THE System SHALL 实时绘制损失曲线
2. THE System SHALL 绘制验证集上的指标变化曲线（PSNR、SSIM）
3. WHEN 评估完成时，THE System SHALL 生成去噪前后的对比图像
4. THE System SHALL 支持将可视化结果保存为图像文件
5. WHERE 支持TensorBoard时，THE System SHALL 集成TensorBoard日志记录
