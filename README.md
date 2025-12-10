模型微调代理（Model Tuning Agent）

概述

本仓库实现了一个模块化的 Python 项目，用于通过 GPT 驱动的代理自动化调整模型超参数。项目结构清晰、职责分离，并提供单一入口用于运行微调与自动调参流程。

主要特性

- 模块化结构：`core/`、`agents/`、`utils/`。
- 集中配置：所有可调参数在 `config.py` 中管理。
- GPT 驱动的调参流程由 `run.py` 协同执行。
- 自动生成运行报告，保存在 `docs/reports/`。
- 提供 `setup_api_key.py` 用于快速配置 OpenAI API Key。

前置条件

- Python 3.11
- PyTorch
- sentence-transformers
- transformers
- datasets
- openai（或兼容的 GPT HTTP 客户端）

安装

建议使用虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果仓库中没有 `requirements.txt`，请根据 `PROGRAM_DESCRIPTION.md` 或代码中导入的包手动安装所需依赖。

快速开始

1. 配置 API Key（任选其一）：

```powershell
python setup_api_key.py
# 或在环境变量中设置 OPENAI_API_KEY，或在 .env 文件中放置 API Key
```

2. 编辑 `config.py`，根据机器资源和数据规模调整参数。

3. 运行代理：

```powershell
python run.py
```

项目结构

- `run.py` — 主入口，协调调参流程。
- `config.py` — 集中配置与可调参数列表。
- `core/` — 与训练相关的实现（数据加载、训练、评估）。
- `agents/` — 与 GPT 通信的封装与提示构建逻辑。
- `utils/` — 工具函数（OpenAI 客户端、报告生成等）。
- `docs/` — 文档与生成的运行报告。

推荐工作流程

- 将参数修改集中在 `config.py`，避免分散在多个脚本中。
- 调试与快速迭代时使用小数据集或减少训练轮数以加速验证。
- 每次运行后检查 `docs/reports/` 中生成的报告以评估效果。

维护与清理

仓库中保留了一些早期版本的旧文件以便向后兼容或用于参考。若希望将项目精简为标准化工程，可根据 `FILE_CLASSIFICATION.md` 中的清单删除不再需要的旧文件。建议在删除前使用版本控制或备份以便恢复。

支持与贡献

如需修改或扩展，建议通过提交 issue 或 PR 进行。请在 PR 中包含可复现的测试或小样例以便审查。

版权与许可证

仓库当前未包含许可证文件。若计划公开分发，请添加合适的 `LICENSE` 文件。
