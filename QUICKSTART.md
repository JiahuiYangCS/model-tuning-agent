快速试运行指南

前置要求

Python 3.11, PyTorch, sentence-transformers, transformers, datasets, openai

环境准备

创建虚拟环境并安装依赖：

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch sentence-transformers transformers datasets openai


配置 API Key

方式一（推荐）：使用脚本自动配置

powershell
python setup_api_key.py


方式二：手动设置环境变量

powershell
$env:OPENAI_API_KEY = "sk-your-key-here"


方式三：创建 .env 文件

在项目根目录创建 .env 文件，内容如下：

OPENAI_API_KEY=sk-your-key-here


快速试运行（5 分钟）

默认配置已为快速测试优化（仅用 200 个样本训练）。直接运行：

powershell
python run.py


脚本流程

1. 加载默认配置
2. 调用 GPT 获取初始调参建议
3. 按建议进行 3 轮参数微调（每轮训练 1 个 epoch）
4. 生成报告到 docs/reports/

预期耗时：5-15 分钟（取决于网络和 GPT 响应）

输出位置

模型检查点：models/
运行报告：docs/reports/agent_run_report_*.md

自定义快速测试

编辑 config.py，修改以下参数加快速度：

python
DEFAULT_CONFIG = {
    "STSB_TRAIN_SPLIT": "train[:100]",  # 改为 100 个样本（更快）
    "STSB_DEV_SPLIT": "validation[:50]",
    "NUM_TRAIN_EPOCHS": 1,               # 保持为 1（不要增加）
    "TRAIN_BATCH_SIZE": 8,               # 如果显存不足改为 4
    "EVAL_STEPS": 20,                    # 减少评估频率
}


常见问题

显存不足

改 config.py：TRAIN_BATCH_SIZE = 4, GRAD_ACC_STEPS = 2

GPT API 错误

检查 OPENAI_API_KEY 有效性，检查网络连接

模型加载失败

确保 sentence-transformers 已安装，首次运行会自动下载模型

验证成功标志

- 生成了 docs/reports/agent_run_report_*.md
- models/ 目录下有训练输出
- 无 Python 异常
