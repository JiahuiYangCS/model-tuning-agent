# Model Tuning Agent with GPU Monitoring

🚀 一个用于自动化模型微调和GPU监控的Agent系统

## ✨ 主要功能

- **自动化模型微调**: 基于Sentence Transformers进行模型训练和评估
- **独立GPU监控**: 实时监控GPU使用情况，训练结束后生成详细报告
- **智能优化建议**: 基于GPU状态提供batch size、混合精度等优化建议
- **OpenRouter API集成**: 支持GPT等大模型API调用

## 📁 项目结构

```
model-tuning-agent/
├── agents/              # Agent模块（GPT Agent等）
├── core/               # 核心训练逻辑
├── gpu_monitor/        # GPU监控模块
├── models/             # 模型注册和保存目录
├── scripts/            # 工具脚本
├── utils/              # 工具函数
├── config.py           # 配置文件
├── run.py              # 主运行脚本
└── gpu_monitor_daemon.py  # GPU监控守护进程
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd model-tuning-agent

# 安装依赖
pip install torch sentence-transformers nvidia-ml-py openai

# 配置API密钥（可选，用于GPT Agent）
python setup_openrouter_api_key.py
```

### 2. 运行训练（带GPU监控）

**终端1 - 启动GPU监控**：
```bash
python gpu_monitor_daemon.py --output training_log.json --interval 1.0
```

**终端2 - 运行训练**：
```bash
python run.py
```

训练完成后，在终端1按`Ctrl+C`停止监控，然后生成报告：
```bash
python gpu_monitor_daemon.py --analyze training_log.json --report report.md
```

### 3. 查看监控报告

生成的`report.md`包含：
- 整体GPU使用统计（平均、最大、最小值）
- 时间段详细分析（自适应时间段）
- 完整历史记录
- 峰值和异常分析

## ⚙️ 配置说明

编辑`config.py`调整训练参数：

```python
# 快速测试模式（使用小数据集）
QUICK_TEST_MODE = False

# 训练参数
NUM_TRAIN_EPOCHS = 4
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

# GPU监控
GPU_ID = 0
```

## 📊 GPU监控特性

### 自动化监控
- ✅ 独立进程，不影响训练性能
- ✅ 低CPU开销（<0.5%）
- ✅ 实时显示GPU状态

### 智能分析
- ✅ **自适应时间段**: 根据训练时长自动调整（总时长的10%，30-120秒）
- ✅ 峰值检测：自动识别GPU利用率、显存、温度、功耗峰值
- ✅ 低利用率统计：识别GPU未充分利用的时段

### 详细报告
- 📈 完整的时序数据
- 📊 统计摘要（平均、最大、最小、中位数）
- 🔍 时间段分析
- ⚡ 峰值时刻定位

## 🛠️ GPU优化建议

系统会根据GPU状态自动提供优化建议：

- **Batch Size优化**: 根据显存使用情况计算最优batch size
- **混合精度训练**: 检测Compute Capability是否支持AMP
- **DataLoader优化**: 自动配置num_workers和pin_memory
- **cuDNN优化**: 启用benchmark模式加速训练

## 📝 高级用法

### 自定义监控间隔
```bash
# 高频监控（0.5秒）
python gpu_monitor_daemon.py --output log.json --interval 0.5

# 低频监控（2秒）
python gpu_monitor_daemon.py --output log.json --interval 2.0
```

### 多GPU监控
```bash
# 监控GPU 0
python gpu_monitor_daemon.py --output gpu0.json --gpu 0

# 监控GPU 1
python gpu_monitor_daemon.py --output gpu1.json --gpu 1
```

## 🎯 使用场景

1. **模型微调实验**: 训练Sentence Transformer模型
2. **GPU性能分析**: 分析GPU利用率，优化训练配置
3. **资源监控**: 长时间训练的GPU状态追踪
4. **对比实验**: 不同配置下的GPU使用对比

## 📄 核心模块说明

### gpu_monitor/
- `pynvml_monitor.py`: 基于NVIDIA官方pynvml的高性能监控
- `gpu_optimizer.py`: GPU优化建议系统

### core/
- `training.py`: Sentence Transformer训练逻辑

### agents/
- `gpt_agent.py`: GPT Agent实现

## 🔧 故障排查

### 问题1：GPU监控无法启动

**解决方案**：
```bash
# 检查pynvml是否安装
pip show nvidia-ml-py

# 如果未安装
pip install nvidia-ml-py
```

### 问题2：训练速度慢

**解决方案**：
1. 查看GPU监控报告，检查GPU利用率
2. 如果利用率低（<50%），增大batch size
3. 启用混合精度训练（AMP）
4. 检查DataLoader的num_workers设置

### 问题3：显存不足

**解决方案**：
1. 减小batch size
2. 启用梯度累积
3. 使用混合精度训练
4. 减少模型大小

## 📚 相关文档

- [GPU监控使用说明](GPU_MONITOR_USAGE.md) - 详细的GPU监控系统使用指南
- [docs/](docs/) - 项目文档和示例

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📝 许可证

MIT License

## 🔗 依赖项

- PyTorch >= 2.0
- sentence-transformers >= 2.0
- nvidia-ml-py (pynvml)
- openai (可选，用于GPT Agent)

---

**开始使用**：
```bash
# 终端1
python gpu_monitor_daemon.py

# 终端2  
python run.py
```

Happy training! 🎉
