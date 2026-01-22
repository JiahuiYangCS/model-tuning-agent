# GPU监控系统使用说明

## 📖 概述

本GPU监控系统提供独立的GPU后台监控进程，可与模型微调训练并行运行，静默记录GPU状态并在训练结束后生成详细的历史分析报告。

**核心特性：**
- ✅ **后台静默运行**：监控进程与训练进程完全分离，互不干扰
- ✅ **持续记录**：静默记录GPU利用率、显存、温度、功耗等指标
- ✅ **详细报告**：训练结束后生成包含每轮、每时段的详细历史报告
- ✅ **高性能**：基于NVIDIA官方pynvml库，CPU开销<0.5%
- ✅ **无干扰**：不实时显示数据，避免终端输出混乱

---

## 🚀 快速开始

### 1. 启动GPU后台监控（独立终端1）

在第一个终端窗口中启动GPU后台监控进程：

```powershell
# 基本用法
python gpu_monitor_daemon.py

# 自定义配置
python gpu_monitor_daemon.py --output my_training.json --interval 0.5 --gpu 0
```

**参数说明：**
- `--output, -o`: 输出JSON文件路径（默认：`gpu_monitor_log.json`）
- `--interval, -i`: 采样间隔，单位秒（默认：`1.0`秒）
- `--gpu, -g`: GPU设备ID（默认：`0`）

**监控特点：**
- 后台静默运行，不实时显示数据
- 每30秒显示一次简要状态（不影响训练）
- 完整数据保存在JSON文件中

### 2. 运行训练脚本（独立终端2）

在第二个终端窗口中运行模型训练：

```powershell
# 运行训练
python run.py
```

### 3. 训练结束后生成报告

训练完成后，按`Ctrl+C`停止监控，然后生成详细报告：

```powershell
# 生成分析报告
python gpu_monitor_daemon.py --analyze gpu_monitor_log.json --report training_report.md
```

---

## 📊 监控报告内容

生成的报告包含以下详细内容：

### 1. **整体统计摘要**
- GPU利用率：平均值、最小值、最大值、中位数
- 显存使用：平均值、最小值、最大值、中位数
- 温度：平均值、最小值、最大值、中位数
- 功耗：平均值、最小值、最大值、中位数

### 2. **时间段分析**
- 按每60秒划分时间段
- 每个时段的平均GPU利用率、显存使用、温度、功耗
- 便于观察训练不同阶段的GPU使用模式

### 3. **详细历史记录**
- 完整的时序数据表格
- 每个采样点的具体时刻和各项指标
- 支持追溯任意时刻的GPU状态

### 4. **峰值与异常分析**
- GPU利用率峰值时刻及数值
- 显存使用峰值时刻及数值
- 温度峰值时刻及数值
- 功耗峰值时刻及数值
- GPU低利用率时段统计（<20%）

### 5. **趋势图数据**
- GPU利用率随时间变化的ASCII趋势图
- 便于快速识别训练过程中的利用率模式

---

## 💡 使用场景示例

### 场景1：标准训练监控

```powershell
# 终端1：启动监控（1秒采样间隔）
python gpu_monitor_daemon.py --output train_20250121.json --interval 1.0

# 终端2：运行训练
python run.py

# 训练结束后，在终端1按Ctrl+C停止监控

# 终端1：生成报告
python gpu_monitor_daemon.py --analyze train_20250121.json --report train_20250121_report.md
```

### 场景2：高频监控（0.5秒采样）

```powershell
# 终端1：启动高频监控
python gpu_monitor_daemon.py --output train_highfreq.json --interval 0.5

# 终端2：运行训练
python run.py

# 停止监控后生成报告
python gpu_monitor_daemon.py --analyze train_highfreq.json --report highfreq_report.md
```

### 场景3：多GPU监控

```powershell
# 终端1：监控GPU 0
python gpu_monitor_daemon.py --output gpu0.json --gpu 0

# 终端2：监控GPU 1
python gpu_monitor_daemon.py --output gpu1.json --gpu 1

# 终端3：运行训练（使用GPU 0或1）
python run.py

# 分别生成两个GPU的报告
python gpu_monitor_daemon.py --analyze gpu0.json --report gpu0_report.md
python gpu_monitor_daemon.py --analyze gpu1.json --report gpu1_report.md
```

---

## 🔧 技术架构

### 核心组件

1. **PyNVMLMonitor**（[pynvml_monitor.py](gpu_monitor/pynvml_monitor.py)）
   - 基于NVIDIA官方pynvml库
   - 高性能、低CPU开销（<0.5%）
   - 支持多GPU监控

2. **GPUMonitorDaemon**（[gpu_monitor_daemon.py](gpu_monitor_daemon.py)）
   - 独立守护进程
   - 持续记录GPU指标到JSON文件
   - 静默后台运行（无实时显示）
   - 每30秒显示简要状态

3. **GPUReportGenerator**（[gpu_monitor_daemon.py](gpu_monitor_daemon.py)）
   - 解析监控日志JSON文件
   - 生成详细的Markdown分析报告
   - 包含统计、时段分析、峰值检测等

### 监控指标

| 指标 | 说明 |
|------|------|
| `gpu_utilization` | GPU计算核心利用率（%）|
| `memory_used_mb` | 显存使用量（MB）|
| `memory_utilization` | 显存占用率（%）|
| `temperature` | GPU温度（°C）|
| `power_draw` | 功耗（W）|
| `relative_time` | 相对监控开始的时间（秒）|
| `datetime` | 绝对时间戳 |

---

## 📁 输出文件说明

### 1. JSON监控日志（`gpu_monitor_log.json`）

```json
{
  "metadata": {
    "start_time": "2025-01-21 10:30:00",
    "end_time": "2025-01-21 10:35:00",
    "duration_seconds": 300.5,
    "total_samples": 301,
    "sampling_interval": 1.0,
    "gpu_id": 0,
    "gpu_name": "NVIDIA GeForce RTX 3080 Ti"
  },
  "metrics": [
    {
      "gpu_utilization": 85.0,
      "memory_used_mb": 8192,
      "memory_utilization": 68.3,
      "temperature": 65,
      "power_draw": 280.5,
      "relative_time": 0.0,
      "datetime": "2025-01-21 10:30:00"
    },
    ...
  ]
}
```

### 2. Markdown分析报告（`gpu_monitor_report.md`）

包含完整的统计分析、时段分析、历史记录、峰值检测等内容。

---

## ⚠️ 注意事项

1. **并行运行**：监控进程和训练进程必须在独立的终端窗口中运行，确保互不干扰
2. **停止监控**：训练结束后，在监控终端按`Ctrl+C`停止监控，数据会自动保存
3. **采样间隔**：建议使用1.0秒采样间隔，过短可能增加CPU开销，过长可能错过关键数据
4. **磁盘空间**：长时间训练会产生较大的JSON文件，注意磁盘空间
5. **NVML权限**：确保有权限访问NVIDIA GPU（需要安装NVIDIA驱动）

---

## 🛠️ 故障排查

### 问题1：无法启动监控

**症状**：运行`gpu_monitor_daemon.py`报错

**解决方案**：
```powershell
# 检查pynvml是否安装
pip show nvidia-ml-py

# 如果未安装，执行安装
pip install nvidia-ml-py
```

### 问题2：报告生成失败

**症状**：`--analyze`命令报错

**解决方案**：
- 确认JSON文件存在且完整
- 检查JSON文件格式是否正确
- 确保监控进程已正常停止（按`Ctrl+C`）

### 问题3：GPU利用率显示为0%

**症状**：监控显示GPU利用率始终为0%

**解决方案**：
- 确认训练脚本正在运行
- 检查训练脚本是否正确使用GPU
- 确认监控的GPU ID与训练使用的GPU ID一致

---

## 📚 相关文档

- [GPU监控模块技术文档](GPU_MODULE_COMPLETE_REPORT.md)
- [GPU优化器说明](gpu_monitor/gpu_optimizer.py)
- [PyNVML监控器实现](gpu_monitor/pynvml_monitor.py)

---

## ✅ 总结

本GPU监控系统提供了完整的后台静默监控解决方案：

1. ✅ **独立进程**：监控与训练完全分离
2. ✅ **静默记录**：后台持续采集GPU状态数据，无实时显示干扰
3. ✅ **详细报告**：包含每轮、每时段的历史详情
4. ✅ **高性能**：基于pynvml，CPU开销极低
5. ✅ **易于使用**：简单的命令行接口

开始使用：
```powershell
# 终端1
python gpu_monitor_daemon.py

# 终端2
python run.py

# 训练结束后，终端1按Ctrl+C，然后：
python gpu_monitor_daemon.py --analyze gpu_monitor_log.json --report report.md
```
