# 模型自动调参代理 - 完整架构文档

## 📋 目录
1. [项目概述](#项目概述)
2. [工作流程](#工作流程)
3. [系统架构图](#系统架构图)
4. [核心模块说明](#核心模块说明)
5. [脚本详解](#脚本详解)
6. [配置系统](#配置系统)
7. [执行流程](#执行流程)

---

## 项目概述

**STSb Auto-Tune Agent** 是一个基于 GPT 驱动的自动超参数调优系统。它能够：

- ✅ 自动识别最优的模型超参数
- ✅ 单变量顺序调参（逐个参数优化）
- ✅ GPT 智能建议参数调整方向
- ✅ 自动生成详细的调参报告
- ✅ 完整的中文交互和日志记录

**适用场景：**
- Sentence Transformer 模型微调
- NLP 模型超参数优化
- 自动化机器学习工作流
- 模型性能基准测试

---

## 工作流程

### 高层工作流

```
[启动程序]
    ↓
[初始化配置]
    ├─ 加载 config.py 的默认参数
    ├─ 初始化 OpenAI 客户端
    └─ 创建输出目录
    ↓
[基础训练]
    ├─ 使用默认参数训练一轮
    ├─ 计算评估指标（准确率、F1 等）
    └─ 记录基础性能
    ↓
[迭代调参]
    ├─→ [选择参数 1]
    │    ├─ 生成当前参数建议（GPT）
    │    ├─ 应用新参数
    │    ├─ 多轮测试（逐步优化）
    │    └─ 保存最优结果
    │
    ├─→ [选择参数 2]
    │    └─ 重复上述流程
    │
    └─→ [选择参数 3]
         └─ 重复上述流程
    ↓
[生成最终报告]
    ├─ 最终结果摘要
    ├─ 每轮详细日志
    ├─ 最优参数配置
    └─ 优化建议
    ↓
[保存最优模型]
    ├─ 复制最优模型到 models/best_overall_model
    └─ 生成性能对比表
    ↓
[完成]
```

---

## 系统架构图

### 文件结构

```
model-tuning-agent/
│
├── 📄 run.py                          [主程序入口]
├── 📄 config.py                       [配置中心]
├── 📄 setup_api_key.py                [API 密钥设置]
├── 📄 openai_client.py                [OpenAI 通信]
│
├── 📁 core/                           [核心功能模块]
│   └── training.py                    [训练、评估、配置管理]
│
├── 📁 agents/                         [智能代理模块]
│   └── gpt_agent.py                   [GPT 交互、参数建议]
│
├── 📁 utils/                          [工具函数]
│   └── report_generator.py            [报告生成]
│
├── 📁 models/                         [模型存储]
│   ├── best_overall_model/            [最优模型]
│   └── stv3_agent_demo_YYYY.../       [历次调参快照]
│
├── 📁 docs/                           [文档]
│   ├── reports/                       [调参报告]
│   └── ...
│
├── 📄 .env                            [API 密钥]（不上传 Git）
├── 📄 .env.example                    [配置模板]
├── 📄 .gitignore                      [Git 忽略规则]
├── 📄 README.md                       [项目说明]
├── 📄 GUIDE.md                        [用户指南]
└── 📄 ARCHITECTURE.md                 [本文档]
```

### 模块交互图

```
┌─────────────────────────────────────────────────────────────┐
│                        run.py                               │
│                     [主程序控制器]                          │
│  - 初始化系统                                              │
│  - 控制训练循环                                            │
│  - 管理调参流程                                            │
│  - 生成最终报告                                            │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴─────────────────────────┐
    │                                  │
    ▼                                  ▼
┌──────────────────┐         ┌──────────────────────┐
│   core/          │         │  agents/             │
│ training.py      │         │  gpt_agent.py        │
├──────────────────┤         ├──────────────────────┤
│ • 训练模型       │◄────────┤ • GPT 建议参数      │
│ • 评估指标       │ 查询    │ • 参数分析          │
│ • 设置参数       │ 结果    │ • 调优策略          │
│ • 保存模型       │         │ • 与 OpenAI 通信   │
└──────────────────┘         └──────────────────────┘
    ▲
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌──────────────────┐         ┌──────────────────────┐
│   config.py      │         │  openai_client.py    │
├──────────────────┤         ├──────────────────────┤
│ • 参数定义       │         │ • API 初始化         │
│ • 默认值         │         │ • 请求管理           │
│ • 调参范围       │         │ • 错误处理           │
│ • 模型设置       │         │ • 响应解析           │
└──────────────────┘         └──────────────────────┘
    ▲
    │
    ▼
┌──────────────────┐
│  utils/          │
│report_generator  │
├──────────────────┤
│ • 生成 MD 报告   │
│ • 格式化输出     │
│ • 性能对比       │
│ • 结果汇总       │
└──────────────────┘
```

---

## 核心模块说明

### 1️⃣ config.py - 配置中心

**作用：** 集中管理所有超参数和配置

**关键类和变量：**

```python
DEFAULT_CONFIG = {
    # 数据配置
    'STSB_TRAIN_SPLIT': 'train[:5000]',
    'STSB_VAL_SPLIT': 'validation[:500]',
    
    # 训练配置
    'NUM_TRAIN_EPOCHS': 1,
    'TRAIN_BATCH_SIZE': 8,
    'EVAL_BATCH_SIZE': 16,
    'LEARNING_RATE': 2e-5,
    
    # 模型配置
    'MODEL_NAME': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'SIMILARITY_FUNCTION': 'cosine',
    
    # ... 共 23 个可调参数
}

TUNABLE_KEYS = [
    'NUM_TRAIN_EPOCHS',
    'TRAIN_BATCH_SIZE',
    'LEARNING_RATE',
    # ... 需要调优的参数列表
]

class AGENT_SETTINGS:
    MAX_PRIORITY_PARAMS = 3          # 优先调优的参数个数
    ROUNDS_PER_PARAM = 3             # 每个参数测试轮数
    GPT_MODEL = 'gpt-3.5-turbo'      # 使用的 GPT 模型
```

**特点：**
- ✅ 所有参数都有默认值
- ✅ 分组管理（数据、训练、模型）
- ✅ 易于扩展新参数
- ✅ 支持批量修改

---

### 2️⃣ core/training.py - 训练和评估

**作用：** 执行模型训练、评估、参数应用

**主要函数：**

#### `make_default_config(config_dict=None)`
```python
def make_default_config(config_dict=None):
    """
    创建训练配置
    
    Args:
        config_dict: 参数字典，如果为 None 使用默认值
        
    Returns:
        training_args: 训练参数对象
        
    流程:
    1. 验证输入参数
    2. 填充缺失的参数（使用默认值）
    3. 创建 TrainingArguments 对象
    4. 返回可用于 Trainer 的配置
    """
```

**工作流程：**
```
输入参数字典
    ↓
验证每个参数的类型和范围
    ↓
与默认值合并（使用输入值覆盖默认值）
    ↓
应用模型特定配置（warmup steps, weight decay 等）
    ↓
返回训练配置对象
```

#### `train_one_round(current_config, round_name)`
```python
def train_one_round(current_config, round_name):
    """
    执行一轮完整的训练和评估
    
    Args:
        current_config: 当前参数配置
        round_name: 轮次名称（用于日志和模型保存）
        
    Returns:
        {
            'eval_accuracy': float,        # 准确率 (0-1)
            'eval_f1': float,              # F1 分数
            'eval_pearson': float,         # 皮尔逊相关系数
            'eval_loss': float,            # 损失值
            'model_dir': str,              # 保存的模型路径
            'training_time': float,        # 训练耗时（秒）
        }
        
    流程:
    1. 加载 STS-B 数据集
    2. 初始化 Sentence Transformer 模型
    3. 应用当前参数配置
    4. 执行训练循环
    5. 在验证集上评估
    6. 保存模型
    7. 返回性能指标
    """
```

**数据处理流程：**
```
数据集加载 (STS-B)
    ├─ 训练集: 5000 样本
    ├─ 验证集: 500 样本
    └─ 每个样本: (句子1, 句子2, 相似度标签)
    ↓
数据预处理
    ├─ 分词
    ├─ 句子编码
    └─ 批处理
    ↓
模型前向传播
    ├─ 输入：句子对
    ├─ 输出：嵌入向量
    └─ 计算相似度
    ↓
损失计算
    ├─ MSE Loss: (预测相似度 - 真实相似度)²
    └─ 反向传播更新参数
    ↓
评估指标
    ├─ Accuracy: 预测排序与真实排序的一致性
    ├─ F1: 精确率和召回率的调和平均
    ├─ Pearson: 相关系数
    └─ Loss: 平均损失
```

#### `set_global_seed(seed=42)`
```python
def set_global_seed(seed=42):
    """
    设置全局随机种子，确保可复现性
    
    Args:
        seed: 种子值（默认 42）
        
    设置的内容:
    - Python random 模块
    - NumPy 随机数生成器
    - PyTorch 随机数生成器
    - CUDA 随机数生成器（如果使用 GPU）
    """
```

---

### 3️⃣ agents/gpt_agent.py - GPT 智能代理

**作用：** 与 OpenAI API 交互，获取参数优化建议

**主要函数：**

#### `ask_gpt_for_initial_plan(config, tunable_keys)`
```python
def ask_gpt_for_initial_plan(config, tunable_keys):
    """
    让 GPT 分析当前配置，给出初始调优计划
    
    Args:
        config: 当前参数配置
        tunable_keys: 可调参数列表
        
    Returns:
        {
            'analysis': str,           # 对当前配置的分析
            'priority_params': list,   # 优先调优的参数列表
            'strategy': str,           # 调优策略
            'reasoning': str,          # 推理过程
        }
        
    GPT Prompt 示例:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    作为 NLP 模型调优专家，分析以下参数配置：
    
    当前配置:
    - NUM_TRAIN_EPOCHS: 1
    - TRAIN_BATCH_SIZE: 8
    - LEARNING_RATE: 2e-5
    
    数据集: STS-B (5000 训练样本)
    模型: Sentence Transformer
    
    请识别最值得优化的 3 个参数，理由是什么？
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
```

**GPT 分析流程：**
```
输入当前配置 → GPT 分析
    ↓
考虑因素:
    ├─ 数据集大小
    ├─ 模型复杂度
    ├─ 计算资源
    ├─ 参数相关性
    └─ 调优历史
    ↓
输出建议:
    ├─ 优先级排序
    ├─ 调整方向（增加/减少）
    ├─ 推荐值范围
    └─ 预期收益
```

#### `ask_gpt_for_new_config(param_name, history, current_score)`
```python
def ask_gpt_for_new_config(param_name, history, current_score):
    """
    根据调参历史，让 GPT 建议下一个参数值
    
    Args:
        param_name: 当前调优的参数名
        history: 调参历史 [{value, score}, ...]
        current_score: 当前最优得分
        
    Returns:
        {
            'next_value': float/int,   # 建议的下一个参数值
            'reasoning': str,          # 建议理由
            'expected_score': float,   # 预期分数（仅供参考）
        }
        
    调参历史示例:
    [
        {'value': 1e-5, 'score': 0.82},    # 第 1 次尝试
        {'value': 2e-5, 'score': 0.85},    # 第 2 次尝试 ✓ 最优
        {'value': 5e-5, 'score': 0.83},    # 第 3 次尝试
    ]
    
    GPT 逻辑:
    1. 识别当前最优值 (2e-5, 得分 0.85)
    2. 分析趋势:
       - 从 1e-5 到 2e-5: 性能提升 ↑
       - 从 2e-5 到 5e-5: 性能下降 ↓
    3. 结论: 最优值在 1e-5 和 2e-5 之间，或略高于 2e-5
    4. 建议下一个值
    """
```

**参数值优化策略：**
```
初始值: 默认值
    ↓
    
第 1 轮: 测试较小的值
示例: LEARNING_RATE 从 2e-5 → 1e-5
    ↓
评估性能 → 如果提升，继续减小
         → 如果下降，切换方向
    ↓
第 2 轮: 根据第 1 轮结果调整
示例: 如果 1e-5 更好，继续尝试 5e-6
      或在 1e-5 和 2e-5 之间寻找最优点
    ↓
第 3 轮: 微调或寻找精确最优值
    ↓
保存该参数的最优值，进入下一个参数
```

---

### 4️⃣ utils/report_generator.py - 报告生成

**作用：** 生成结构化的 Markdown 调参报告

**主要函数：**

#### `generate_run_report(param_history, round_results, final_config)`
```python
def generate_run_report(param_history, round_results, final_config):
    """
    生成完整的调参运行报告
    
    Args:
        param_history: 调参历史 {param_name: [{value, score}, ...]}
        round_results: 每轮结果 [round1, round2, ...]
        final_config: 最终配置
        
    Returns:
        report_path: 生成的报告文件路径
        
    报告结构:
    ┌─────────────────────────────┐
    │   【最终结果摘要】          │ ← 重点内容，置顶
    │ • 最优轮次                  │
    │ • 最优得分                  │
    │ • 优化参数                  │
    │ • 性能提升                  │
    └─────────────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │   【详细调参日志】          │ ← 历史过程
    │ 第 1 轮 (参数1)             │
    │  - 测试值 1                 │
    │  - 测试值 2                 │
    │  - 最优值 & 得分           │
    │                             │
    │ 第 2 轮 (参数2)             │
    │  ...                        │
    └─────────────────────────────┘
             ↓
    ┌─────────────────────────────┐
    │   【优化建议】              │
    │ • 后续改进方向              │
    │ • 可尝试的技术              │
    │ • 资源投入优先级            │
    └─────────────────────────────┘
    """
```

**报告文件示例：**
```
# STSb 自动调参报告
生成时间: 2025-12-10 17:45:03

## 最终结果摘要

### 最优轮次
第 7 轮 (参数: NUM_TRAIN_EPOCHS)

### 最优得分
- Accuracy: 0.8742
- F1: 0.8651
- Pearson: 0.8523

### 优化前后对比
| 指标 | 基础配置 | 最优配置 | 提升 |
|------|---------|---------|------|
| Accuracy | 0.8234 | 0.8742 | +6.18% |
| F1 | 0.8102 | 0.8651 | +6.76% |

### 优化参数
- NUM_TRAIN_EPOCHS: 1 → 3 (+200%)
- LEARNING_RATE: 2e-5 → 1e-5 (-50%)

## 详细调参日志

### 第 1 轮: NUM_TRAIN_EPOCHS
尝试值: [1, 2, 3]
| 值 | Accuracy | F1 | Pearson | 耗时 |
|---|----------|----|---------|----|
| 1 | 0.8234 | 0.8102 | 0.7891 | 45s |
| 2 | 0.8456 | 0.8324 | 0.8103 | 89s |
| 3 | 0.8652 | 0.8531 | 0.8321 | 134s | ← 最优

### 第 2 轮: LEARNING_RATE
...

## 优化建议
1. 考虑增加 warmup steps
2. 可尝试更复杂的学习率调度
3. ...
```

---

### 5️⃣ openai_client.py - OpenAI 通信

**作用：** 管理与 OpenAI API 的连接和通信

**主要类：**

```python
class OpenAIClient:
    """OpenAI API 客户端"""
    
    def __init__(self, api_key=None):
        """
        初始化客户端
        
        Args:
            api_key: OpenAI API Key
                   如果为 None，从环境变量 OPENAI_API_KEY 读取
                   
        过程:
        1. 从 .env 文件或环境变量加载 API Key
        2. 验证 Key 格式
        3. 初始化 OpenAI 客户端对象
        4. 设置超时和重试参数
        """
    
    def call_gpt(self, prompt, model='gpt-3.5-turbo', temperature=0.7):
        """
        调用 GPT 进行推理
        
        Args:
            prompt: 提示词
            model: 使用的模型
            temperature: 生成多样性控制
            
        Returns:
            response_text: GPT 的响应文本
            
        请求流程:
        1. 构建请求消息
        2. 通过 API 发送请求
        3. 等待响应
        4. 解析响应内容
        5. 处理错误（如 token 超限）
        6. 返回文本
        """
    
    def parse_json_response(self, response_text):
        """
        解析 GPT 的 JSON 格式响应
        
        Args:
            response_text: 原始响应文本
            
        Returns:
            parsed_data: 解析后的 Python 字典
            
        注意:
        - GPT 可能在 JSON 前后有额外文本
        - 自动提取 {...} 结构
        - 处理转义字符
        - 验证必需字段存在
        """
```

---

## 脚本详解

### 1️⃣ run.py - 主程序入口

**位置：** `x:\Intellipro\model-tuning-agent\run.py`

**行数：** 308 行

**主要组件：**

#### 导入和初始化（第 1-40 行）

```python
import os
import sys
import json
import time
from pathlib import Path

# 项目模块
from config import DEFAULT_CONFIG, AGENT_SETTINGS, TUNABLE_KEYS
from core.training import make_default_config, train_one_round, set_global_seed
from agents.gpt_agent import ask_gpt_for_initial_plan, ask_gpt_for_new_config
from utils.report_generator import generate_run_report
from openai_client import OpenAIClient

# 初始化
set_global_seed(42)  # 确保可复现性
client = OpenAIClient()  # 初始化 OpenAI 客户端
```

#### 主函数：run_agent()（第 100-280 行）

**功能：** 执行完整的自动调参流程

**执行流程：**

```python
def run_agent():
    """主程序"""
    
    # 1. 创建输出目录
    output_dir = Path('models') / 'stv3_agent_demo_YYYYMMDD_HHMMSS'
    report_dir = Path('docs/reports')
    
    # 2. 基础训练 (使用默认参数)
    print("执行基础训练...")
    base_config = make_default_config()
    base_results = train_one_round(base_config, 'baseline')
    base_score = base_results['eval_accuracy']
    
    # 3. 获取 GPT 初始建议
    print("咨询 GPT...")
    initial_plan = ask_gpt_for_initial_plan(DEFAULT_CONFIG, TUNABLE_KEYS)
    priority_params = initial_plan['priority_params']  # 优先调优的参数
    
    # 4. 迭代调参
    param_history = {}  # 记录每个参数的调参历史
    round_results = []  # 记录每一轮的结果
    
    for param_idx, param_name in enumerate(priority_params[:AGENT_SETTINGS.MAX_PRIORITY_PARAMS]):
        print(f"\n【第 {param_idx+1} 轮】调优参数: {param_name}")
        
        param_history[param_name] = []  # 初始化该参数的历史
        best_param_value = DEFAULT_CONFIG[param_name]  # 该参数的最优值
        best_param_score = base_score  # 该参数的最优得分
        
        # 对该参数进行多轮测试
        for round_in_param in range(AGENT_SETTINGS.ROUNDS_PER_PARAM):
            try:
                # A. 获取 GPT 建议的参数值
                suggestion = ask_gpt_for_new_config(
                    param_name, 
                    param_history[param_name],
                    best_param_score
                )
                next_value = suggestion['next_value']
                
                # B. 生成测试配置
                test_config = DEFAULT_CONFIG.copy()
                test_config[param_name] = next_value
                test_config_obj = make_default_config(test_config)
                
                # C. 执行训练
                round_name = f'{param_name}_r{round_in_param+1}'
                print(f"  测试: {param_name}={next_value}")
                
                results = train_one_round(test_config_obj, round_name)
                score = results['eval_accuracy']
                
                # D. 记录结果
                param_history[param_name].append({
                    'value': next_value,
                    'score': score
                })
                round_results.append(results)
                
                # E. 更新该参数的最优值
                if score > best_param_score:
                    best_param_score = score
                    best_param_value = next_value
                    print(f"  ✓ 新最优! 得分: {score:.4f}")
                else:
                    print(f"  得分: {score:.4f}")
                    
            except Exception as e:
                # 故障处理：该参数超时或出错
                print(f"  ⚠ 该轮失败: {str(e)[:50]}")
                continue  # 自动跳过，进入下一轮
        
        # 更新全局最优配置
        DEFAULT_CONFIG[param_name] = best_param_value
        print(f"  该参数最优值: {param_name}={best_param_value}")
    
    # 5. 生成报告
    print("\n生成报告...")
    report_path = generate_run_report(param_history, round_results, DEFAULT_CONFIG)
    print(f"报告已保存: {report_path}")
    
    # 6. 保存最优模型
    print("\n保存最优模型...")
    best_model_dir = Path('models/best_overall_model')
    # 复制最优模型文件
    
    print("\n调参完成！")

if __name__ == '__main__':
    run_agent()
```

**关键特性：**

1. **故障自动跳过**
   ```python
   try:
       results = train_one_round(config, name)
   except:
       continue  # 自动跳过失败的轮次
   ```

2. **GPT 驱动的参数建议**
   ```python
   # 每次都咨询 GPT，而不是固定参数值
   suggestion = ask_gpt_for_new_config(param_name, history, best_score)
   ```

3. **灵活的输出结构**
   ```python
   models/
   ├── best_overall_model/  # 最优模型
   └── stv3_agent_demo_DATETIME_rN/  # 每轮快照
   ```

---

### 2️⃣ setup_api_key.py - API 密钥设置

**位置：** `x:\Intellipro\model-tuning-agent\setup_api_key.py`

**行数：** 128 行

**工作流程：**

```python
def setup_api_key(api_key=None):
    """
    设置 OpenAI API Key
    
    流程:
    1. 如果没有提供 Key，提示用户输入（隐藏显示）
    2. 验证 Key 格式（应以 'sk-' 开头）
    3. 读取或创建 .env 文件
    4. 写入或更新 OPENAI_API_KEY
    5. 返回成功标志
    
    安全特性:
    - Key 从不存储在脚本中
    - Key 从不显示在屏幕上
    - Key 只保存在本地 .env 文件
    - .env 被 .gitignore 保护
    """
    
    # 关键步骤
    if not api_key:
        # 提示用户输入，隐藏输入内容
        api_key = getpass.getpass("API Key (不会显示在屏幕上): ")
    
    # 验证格式
    if not api_key.startswith("sk-"):
        print("警告: API Key 格式似乎不对")
        return False
    
    # 写入 .env
    env_file = Path(__file__).parent / ".env"
    with open(env_file, 'w') as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print("✓ API Key 已保存到 .env")
    return True

def main():
    """主入口"""
    print("OpenAI API Key 设置")
    print("-" * 40)
    
    success = setup_api_key()  # 交互式输入
    
    if success:
        print("\n✓ 设置完成")
        print("现在可以运行: python run.py")
    else:
        print("\n✗ 设置失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**重要特性：**
- ✅ 交互式输入（运行时提示）
- ✅ 脚本中不存储敏感信息
- ✅ 密钥隐藏输入（getpass）
- ✅ 自动创建/更新 .env

---

### 3️⃣ config.py - 配置中心

**位置：** `x:\Intellipro\model-tuning-agent\config.py`

**行数：** ~200 行

**核心部分：**

```python
# 默认配置字典（23 个参数）
DEFAULT_CONFIG = {
    # ━━━━━ 数据配置 ━━━━━
    'STSB_DATASET_NAME': 'sentence-transformers/stsb',
    'STSB_TRAIN_SPLIT': 'train[:5000]',
    'STSB_VAL_SPLIT': 'validation[:500]',
    
    # ━━━━━ 训练配置 ━━━━━
    'NUM_TRAIN_EPOCHS': 1,
    'TRAIN_BATCH_SIZE': 8,
    'EVAL_BATCH_SIZE': 16,
    'LEARNING_RATE': 2e-5,
    'WEIGHT_DECAY': 0.01,
    'WARMUP_RATIO': 0.1,
    'GRADIENT_ACCUMULATION_STEPS': 1,
    'MAX_GRAD_NORM': 1.0,
    
    # ━━━━━ 评估配置 ━━━━━
    'EVAL_STRATEGY': 'epoch',
    'SAVE_STRATEGY': 'epoch',
    'LOGGING_STEPS': 100,
    
    # ━━━━━ 模型配置 ━━━━━
    'MODEL_NAME': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'SIMILARITY_FUNCTION': 'cosine',
    'NUM_LABELS': 1,  # STS-B 回归任务
    
    # ━━━━━ 其他配置 ━━━━━
    'SEED': 42,
    'FP16': False,  # 使用 32 位浮点（更稳定）
}

# 可调参数列表
TUNABLE_KEYS = [
    'NUM_TRAIN_EPOCHS',
    'TRAIN_BATCH_SIZE',
    'LEARNING_RATE',
    'WEIGHT_DECAY',
    'WARMUP_RATIO',
    'GRADIENT_ACCUMULATION_STEPS',
    # ... 共 23 个
]

# 代理设置
class AGENT_SETTINGS:
    # 一次运行中最多优化多少个参数
    MAX_PRIORITY_PARAMS = 3
    
    # 每个参数测试多少轮
    ROUNDS_PER_PARAM = 3
    
    # 使用的 GPT 模型
    GPT_MODEL = 'gpt-3.5-turbo'
    
    # 轮次超时时间（秒）
    ROUND_TIMEOUT_SECONDS = 1800  # 30 分钟

# 模型设置
class MODEL_SETTINGS:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = False  # 自动混合精度
```

**扩展新参数的方法：**

1. 在 `DEFAULT_CONFIG` 中添加键值对
2. 在 `TUNABLE_KEYS` 中添加参数名
3. 在 `core/training.py` 的 `make_default_config()` 中添加处理逻辑

---

## 配置系统

### 参数优先级和调优范围

```
参数类别        | 默认值      | 调优范围        | 优先级 | 原因
─────────────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS | 1         | 1, 2, 3, 5     | ⭐⭐⭐ | 最影响性能
LEARNING_RATE    | 2e-5      | 1e-5 ~ 5e-5    | ⭐⭐⭐ | 关键超参数
TRAIN_BATCH_SIZE | 8         | 8, 16, 32      | ⭐⭐  | 次关键
WARMUP_RATIO     | 0.1       | 0.05 ~ 0.2     | ⭐⭐  | 训练稳定性
WEIGHT_DECAY     | 0.01      | 0.001 ~ 0.1    | ⭐    | 正则化
```

### 如何修改配置

**方法 1：修改源文件**
```python
# config.py
DEFAULT_CONFIG['NUM_TRAIN_EPOCHS'] = 2
```

**方法 2：运行时传入**
```python
# run.py
test_config = DEFAULT_CONFIG.copy()
test_config['LEARNING_RATE'] = 1e-5
config_obj = make_default_config(test_config)
```

**方法 3：代理自动调整**
```python
# gpt_agent.py 返回的建议会自动应用
suggestion = ask_gpt_for_new_config(...)
# 返回: {'next_value': 3, 'reasoning': '...'}
```

---

## 执行流程

### 完整的时间线

```
[T0] 程序启动
 └─ setup_api_key.py: 检查 .env 中是否有 API Key
    ├─ 有 ✓ → 继续
    └─ 无 ✗ → 提示用户输入

[T1] run.py 启动
 └─ 导入配置和模块
    └─ 初始化 OpenAI 客户端

[T2] 基础训练
 └─ 用默认参数训练 1 轮
    ├─ 加载数据集 (5000 样本)
    ├─ 初始化模型
    ├─ 执行训练循环 (NUM_TRAIN_EPOCHS=1)
    ├─ 在验证集上评估
    └─ 记录基础得分 (e.g., 0.8234)

[T3-T5] 第 1 轮调参 (参数 1: NUM_TRAIN_EPOCHS)
 └─ GPT 分析: "应该增加 epochs 提高性能"
    ├─ 迭代 1: 测试 epochs=2
    │  ├─ 得分 0.8456 ✓ 更优
    │  └─ 保存最优值
    ├─ 迭代 2: 测试 epochs=3
    │  ├─ 得分 0.8652 ✓ 更优
    │  └─ 保存最优值
    └─ 迭代 3: 测试 epochs=4
       ├─ 得分 0.8631 ✗ 过度训练
       └─ 保留之前的最优值 (epochs=3)

[T6-T8] 第 2 轮调参 (参数 2: LEARNING_RATE)
 └─ GPT 分析: "当前 epochs=3，可以降低学习率"
    ├─ 迭代 1: 测试 lr=1e-5
    ├─ 迭代 2: 测试 lr=5e-6
    └─ 迭代 3: 测试 lr=3e-5
       └─ 得出最优 lr

[T9-T11] 第 3 轮调参 (参数 3: TRAIN_BATCH_SIZE)
 └─ 类似的迭代过程...

[T12] 生成报告
 └─ 汇总所有调参记录
    ├─ 最终结果摘要
    ├─ 详细日志
    └─ 优化建议

[T13] 保存最优模型
 └─ 复制最优轮次的模型到 models/best_overall_model

[结束] 输出最终统计
 └─ 总耗时、优化幅度、推荐配置等
```

---

## 关键设计决策

### 1️⃣ 为什么使用单变量顺序调参？

```
优点:
✅ 解释性强：每个参数的影响清晰可见
✅ 易于调试：出现问题时容易定位原因
✅ 计算量可控：每轮只变化一个参数
✅ 与 GPT 兼容：GPT 可以基于历史做出更好建议

缺点:
❌ 无法发现参数间的相互作用
❌ 比网格搜索慢（但比随机搜索快）

选择原因: 
→ 对于小规模调参任务，性价比最优
→ 容易与 LLM 集成
```

### 2️⃣ 为什么选择 gpt-3.5-turbo？

```
对比分析:
模型              成本    速度    质量   推荐度
─────────────────────────────────────────────
gpt-3.5-turbo     $    很快    ★★★   ✓ 选中
gpt-4             $$$  较慢    ★★★★  （贵）
claude-3-sonnet   $$   快      ★★★   （其他）
local LLaMA       免费  取决于  ★★    (需自建)

选择原因:
→ 成本最低，适合频繁调用
→ 速度快，用户体验好
→ 质量足够，能理解参数调优逻辑
→ API 稳定性有保障
```

### 3️⃣ 故障处理机制

```
故障检测:
├─ GPU 内存不足
├─ 模型加载失败  
├─ 数据集下载超时
├─ 训练 Loss 爆炸
└─ 评估出错

处理方案:
├─ Try-Except 捕获异常
├─ 打印错误信息
├─ 自动跳过该轮 (continue)
├─ 保留之前的最优值
└─ 继续下一轮参数调优

优点:
✓ 程序稳定性高
✓ 不会因单轮失败而中断整个流程
✓ 用户可以查看错误日志诊断问题
```

---

## 常见问题排除

### Q1: 为什么训练很慢？

**答：** 检查以下几点：
1. GPU 是否被正确使用？ → 检查 `DEVICE` 设置
2. 数据集是否太大？ → 修改 `STSB_TRAIN_SPLIT`
3. `NUM_TRAIN_EPOCHS` 是否太高？ → 从 1 开始调参
4. Batch Size 是否太小？ → 增加 `TRAIN_BATCH_SIZE`

### Q2: OpenAI API 返回错误？

**答：** 可能的原因：
1. API Key 过期或无效 → 更新 .env
2. 超过使用额度 → 检查 OpenAI 账户
3. 网络问题 → 检查代理设置
4. Prompt token 过多 → 简化 prompt 内容

### Q3: 模型性能下降而不是提升？

**答：** 这是正常的，原因可能是：
1. 某个参数值不适合当前数据
2. 需要更多的超参数调整组合
3. 数据集可能需要预处理改进
→ 继续运行，GPT 会识别出不好的参数值

### Q4: 如何自定义调参参数？

**答：** 编辑 `config.py`：
```python
# 添加新参数到 DEFAULT_CONFIG
DEFAULT_CONFIG['NEW_PARAM'] = value

# 添加到可调参数列表
TUNABLE_KEYS.append('NEW_PARAM')

# 在 core/training.py 中处理该参数
```

---

## 总结

这个系统通过以下创新设计实现了自动超参数调优：

1. **GPT 驱动的决策** - 智能分析历史数据，给出最优建议
2. **模块化架构** - 清晰的职责分工，易于扩展
3. **鲁棒的错误处理** - 单轮失败不影响全局
4. **详细的日志记录** - 完全可追溯的调参过程
5. **灵活的配置系统** - 支持快速修改和实验

**适用场景：** 自动化 NLP 模型调优、快速基准测试、生产环境超参数优化

**成本效益：** 用 $0.5-2 的 API 成本，节省数小时的手工调参时间

