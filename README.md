# Model Tuning Agent

基于 GPT 的 Sentence Transformer 自动调参系统。

## 项目简介

自动化训练和优化 Sentence Transformer 模型，使用 GPT 智能调整超参数。支持 OpenRouter 免费模型接口。

## 使用方法

### 安装依赖

```bash
pip install torch sentence-transformers transformers datasets openai
```

### 配置 API

```bash
python setup_openrouter_api_key.py
```

### 运行训练

```bash
python run.py
```

## 核心文件

- `run.py` - 主程序入口
- `config.py` - 配置文件（训练参数、数据集设置）
- `core/training.py` - 训练核心逻辑
- `agents/gpt_agent.py` - GPT 调参策略
- `openrouter_client.py` - OpenRouter API 客户端

## 参数配置

编辑 `config.py` 调整：
- 训练数据量（默认 `train[:200]`）
- 训练轮数（默认 1 epoch）
- Batch size、学习率等超参数
- GPT 模型选择

## 测试工具

```bash
python scripts/run_quick_model_tests.py  # 快速验证（小数据集）
python scripts/run_deep_model_tests.py   # 深度测试（大数据集）
```

## 查看结果

- 训练报告：`docs/reports/`
- 最新报告：`LATEST_REPORT.md`
- 模型输出：`models/`

## 项目结构

```
├── run.py                  # 主程序
├── config.py              # 配置文件
├── openrouter_client.py   # API 客户端
├── agents/                # GPT agent 模块
├── core/                  # 训练核心代码
├── utils/                 # 工具函数
├── models/                # 训练输出
├── scripts/               # 测试脚本
└── docs/                  # 文档与报告
    └── reports/           # 训练报告
```

## 技术栈

- Python 3.11+
- PyTorch
- sentence-transformers
- transformers
- datasets
- openai

## 文档

- `docs/FUNCTIONS_DOC.md` - 函数文档
- `docs/reports/` - 历史训练报告
- `LATEST_REPORT.md` - 最新测试结果
