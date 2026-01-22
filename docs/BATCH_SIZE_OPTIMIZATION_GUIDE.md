# Batch Size优化指南

## 📖 概述

本项目包含一个基于**Google Deep Learning Tuning Playbook**原则的自动batch size优化脚本，可以科学地找到最适合你硬件和项目的batch size。

## 🎯 脚本功能

`scripts/find_optimal_batch_size.py` 自动化完成以下任务：

1. **固定所有超参数** - 学习率、优化器、正则化等全部固定
2. **测试多个batch size** - 默认测试 [4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
3. **测量训练吞吐量** - 核心指标：samples/second
4. **分析scaling效率** - 识别Perfect Scaling区间和Critical Batch Size
5. **智能推荐** - 基于吞吐量、GPU利用率和内存安全性推荐最优值
6. **生成详细报告** - Markdown格式，包含所有数据和分析

## 🚀 使用方法

### 快速开始

```bash
# 直接运行脚本
python scripts/find_optimal_batch_size.py
```

### 测试过程

脚本会：
1. ✅ 检测GPU硬件信息
2. 🔒 显示固定的超参数配置
3. 🧪 逐个测试每个batch size：
   - 预热10步（忽略不稳定数据）
   - 测量50步（计算吞吐量）
   - 记录GPU利用率和内存使用
4. 📊 分析scaling效率
5. 🎯 推荐最优batch size
6. 📄 生成详细报告

### 预期运行时间

- 每个batch size测试：约2-5分钟
- 总计10个batch size：约20-50分钟
- 如遇到OOM会自动停止

## 📊 输出示例

### 控制台输出

```
======================================================================
🧪 测试 Batch Size = 32
======================================================================
🔧 加载模型...
📊 创建固定测试数据集：1000 样本
🚀 开始训练测量...
   - 预热步数: 10
   - 测量步数: 50
100%|████████████████████████| 60/60 [00:02<00:00, 25.31it/s]

✅ 测试完成
   训练吞吐量: 1024.5 samples/sec
   时间/步: 0.0312s
   GPU利用率: 85.3%
   峰值内存: 8192MB (66.7%)
```

### 生成的报告

报告保存在 `docs/reports/optimal_batch_size_YYYYMMDD_HHMMSS.md`，包含：

1. **硬件环境**: GPU型号、显存、CUDA版本
2. **固定超参数**: 学习率、优化器等配置
3. **测试结果表格**: 所有batch size的完整数据
4. **Scaling效率分析**: Perfect Scaling区间识别
5. **推荐结果**: 最优batch size及理由
6. **实施指南**: 如何应用到config.py
7. **完整数据**: JSON格式的详细记录

## 🔍 关键概念

### 1. Perfect Scaling

在Perfect Scaling区间内：
- 吞吐量随batch size线性增长
- Scaling效率 ≥ 80%
- 例如：batch size从16翻倍到32，吞吐量也翻倍

### 2. Critical Batch Size

超过此值后：
- 吞吐量不再线性增长
- Scaling效率 < 80%
- 继续增大batch size没有意义

### 3. 训练吞吐量

核心指标，定义为：
```
训练吞吐量 = 每秒处理的样本数 (samples/second)
```

更高的吞吐量 = 更快的训练速度

## 💡 Google Playbook核心原则

### 原则1: Batch Size不影响最终精度

> 只要其他超参数调优得当（特别是学习率和正则化），使用任何batch size都能达到相同的最终性能。

**含义**: 
- Batch size只是速度优化的工具
- 不应该用batch size来调优验证集性能

### 原则2: 找到最大可用的Batch Size

> 通常，硬件支持的最大batch size就是最优选择。

**原因**:
- 更大的batch size更好地利用GPU
- 减少训练时间
- 加快开发迭代

### 原则3: 改变Batch Size需要重新调优

> ⚠️ **重要**: 如果你改变batch size，必须重新调优：
> - 学习率
> - 正则化参数
> - 可能需要调整训练步数

**建议**: 
- 在项目早期确定batch size
- 避免频繁改变

## 🎯 推荐策略

脚本会按以下优先级推荐：

1. **最优选择**: Perfect Scaling区间内的最大batch size
2. **次优选择**: 吞吐量最高的batch size
3. **安全边界**: 内存利用率 < 90%

## 📈 示例结果解读

假设测试结果：

| Batch Size | 吞吐量 | Scaling效率 | GPU利用率 | 内存利用 |
|------------|--------|------------|----------|---------|
| 4 | 100 | 100% | 15% | 20% |
| 8 | 200 | 100% | 25% | 25% |
| 16 | 400 | 100% | 45% | 35% |
| 32 | 780 | 97.5% | 80% | 55% |
| 64 | 1400 | 87.5% | 92% | 75% |
| 128 | 1800 | 56.3% | 95% | 88% |

**分析**:
- Perfect Scaling范围: 4-32 (效率≥80%)
- Critical Batch Size: 128 (效率骤降)
- **推荐**: Batch Size = 64
  - 仍在良好scaling区间
  - GPU利用率高(92%)
  - 内存安全(75%)

## ⚠️ 注意事项

### 1. 测试环境要求

- ✅ 需要CUDA GPU
- ✅ 需要安装sentence-transformers
- ✅ 需要足够的GPU内存
- ✅ 建议关闭其他GPU占用程序

### 2. 测试期间

- 📊 测试会占用GPU，不要同时运行其他训练
- ⏱️ 每个batch size测试2-5分钟
- 💾 会生成临时模型文件（自动清理）
- 🔄 遇到OOM会自动停止

### 3. 应用推荐结果

使用推荐的batch size：

```python
# 在 config.py 中设置
TRAIN_BATCH_SIZE = 64  # 替换为推荐值
EVAL_BATCH_SIZE = 64   # 通常与训练batch size相同
```

⚠️ 保持其他超参数不变，因为测试时用的就是当前配置！

## 🔧 自定义配置

如果需要修改测试参数，编辑脚本中的配置：

```python
class BatchSizeOptimizer:
    def __init__(self):
        # 要测试的batch size列表
        self.BATCH_SIZES_TO_TEST = [4, 8, 16, 32, 48, 64, 96, 128]
        
        # 测试数据集大小
        self.DATASET_SIZE = 1000
        
        # 预热和测量步数
        self.NUM_WARMUP_STEPS = 10
        self.NUM_MEASUREMENT_STEPS = 50
```

## 📚 参考资料

- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Choosing the Batch Size](https://github.com/google-research/tuning_playbook#choosing-the-batch-size)
- [Shallue et al. 2018 - Measuring the Effects of Data Parallelism](https://arxiv.org/abs/1811.03600)

## 🤔 常见问题

### Q: 为什么我的GPU利用率很低？

A: 可能原因：
- Batch size太小
- 数据加载瓶颈（增加num_workers）
- 模型太简单（对于强大的GPU）

### Q: 所有batch size都OOM怎么办？

A: 考虑：
- 使用混合精度训练（FP16）
- 减小模型大小
- 使用梯度累积（但会牺牲速度）

### Q: 测试结果与实际训练不一致？

A: 可能因为：
- 实际训练使用了不同的超参数
- 实际数据集更大/更复杂
- 使用了不同的模型

### Q: 需要多久重新测试一次？

A: 需要重新测试当：
- 更换GPU硬件
- 改变模型架构
- 改变数据集大小

## ✅ 总结

这个脚本提供了一种**科学、自动化**的方法来找到最优batch size，帮助你：

1. ✅ 充分利用GPU硬件
2. ✅ 减少训练时间
3. ✅ 避免手动试错
4. ✅ 遵循最佳实践

立即运行 `python scripts/find_optimal_batch_size.py` 开始优化！🚀
