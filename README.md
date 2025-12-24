# Model Tuning Agent

自动调参的 STSb 模型训练系统。

## 干啥的

跑 Sentence Transformer 模型，自动调参数。支持 OpenRouter 免费模型。

## 快速开始

```bash
# 装依赖
pip install torch sentence-transformers transformers datasets openai

# 配置 API key
python setup_openrouter_api_key.py

# 跑起来
python run.py
```

## 主要文件

- `run.py` - 主入口，跑这个就行
- `config.py` - 改参数在这改
- `core/training.py` - 训练逻辑
- `agents/gpt_agent.py` - GPT 调参逻辑
- `openrouter_client.py` - OpenRouter API 客户端

## 配置

在 `config.py` 改：
- 数据集大小（默认 train[:200]）
- 训练轮数（默认 1 轮）
- batch size、学习率等
- 选哪个 GPT 模型

## 测试脚本

快速测试：
```bash
python scripts/run_quick_model_tests.py  # 小数据集快速验证
python scripts/run_deep_model_tests.py   # 大数据集深度测试
```

## 报告

训练完自动生成报告在 `docs/reports/`，最新的在主目录 `LATEST_REPORT.md`。

## 结构

```
├── run.py                  # 主入口
├── config.py              # 配置文件
├── openrouter_client.py   # OpenRouter 客户端
├── agents/                # GPT agent
├── core/                  # 训练核心
├── utils/                 # 工具类
├── models/                # 训练输出的模型
├── scripts/               # 测试脚本
└── docs/reports/          # 自动生成的报告
```

## 依赖

- Python 3.11+
- PyTorch
- sentence-transformers
- transformers
- datasets
- openai

就这样。
