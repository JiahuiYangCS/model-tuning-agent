使用说明书

项目概述

模型微调代理（Model Tuning Agent）是一个自动化超参数调优工具。通过 GPT 驱动的代理自动建议和调整训练参数，逐参数进行单变量优化。

核心特性

- 自动超参建议：GPT 分析训练结果并提出参数调整建议
- 单变量调优：逐个参数迭代优化，保持其他参数不变
- 自动报告生成：每次运行自动生成中英文混合报告
- 配置集中管理：所有参数在 config.py 中，易于修改

文件结构

run.py                 主入口，协调整个流程
config.py              全局配置，所有参数在此
setup_api_key.py       API Key 配置脚本
core/training.py       训练相关函数
agents/gpt_agent.py    GPT 交互逻辑
utils/                 工具函数（OpenAI 客户端、报告生成等）
docs/reports/          生成的运行报告
models/                训练后的模型检查点

工作流程

1. 初始化阶段
   - 加载 DEFAULT_CONFIG 中的默认超参
   - 初始化 OpenAI 客户端

2. 获取初始建议
   - 调用 GPT 分析当前配置
   - GPT 返回 base_config（初始建议）和 priority_keys（优先调参列表）

3. 逐参数优化循环
   对每个 priority_key（最多 3 个）：
   - 执行 ROUNDS_PER_PARAM 轮（默认 3 轮）训练
   - 每轮训练后用 GPT 获取该参数的新建议
   - 将 GPT 建议应用到配置
   - 记录每轮的评估分数，跟踪参数最优值

4. 完成后处理
   - 将最优参数配置固定
   - 复制最佳模型到 best_overall_model/
   - 调用 GPT 生成最终总结报告

5. 报告生成
   - 自动生成 docs/reports/agent_run_report_<时间戳>.md
   - 包含每轮的指标、参数变化、GPT 建议

基本使用

第一次运行

bash
python setup_api_key.py     # 配置 API Key
python run.py               # 启动自动调参


标准调参（使用默认配置）

config.py 中的 DEFAULT_CONFIG 已包含合理的默认值。若无特殊需求，直接运行：

bash
python run.py


自定义参数调参

编辑 config.py 中的 DEFAULT_CONFIG，修改需要的超参：

python
DEFAULT_CONFIG = {
    ...
    "NUM_TRAIN_EPOCHS": 3,              # 增加训练轮数
    "TRAIN_BATCH_SIZE": 16,             # 增加批大小
    "LEARNING_RATE": 1e-4,              # 调整学习率
    ...
}

保存后运行：

bash
python run.py


配置说明

关键超参

NUM_TRAIN_EPOCHS        训练轮数，增加可提升效果但耗时更长
TRAIN_BATCH_SIZE        批大小，越大显存需求越大
LEARNING_RATE           学习率，影响收敛速度和最终效果
WARMUP_RATIO            预热比例，通常 0.05-0.1
EVAL_STEPS              评估间隔，影响收敛监控频率
SAVE_STEPS              保存间隔

可调参数列表

DEFAULT_CONFIG 中的以下参数可被 GPT 自动调整（在 TUNABLE_KEYS 中）：

BASE_MODEL, NUM_TRAIN_EPOCHS, TRAIN_BATCH_SIZE, LEARNING_RATE, WARMUP_RATIO, EVAL_STEPS, SAVE_STEPS, 等

受保护参数

不在 TUNABLE_KEYS 中的参数（如 OUTPUT_DIR_ROOT）不会被 GPT 修改，保证项目安全性。

运行模式

交互式模式（默认）

每轮训练后会询问用户是否继续下一轮。按 y 继续或 n 停止。

全自动模式（可配置）

修改 AGENT_SETTINGS 中的 INTERACTIVE_MODE = False（需在代码中修改）。

输出与报告

模型检查点

位置：models/ 目录下，按时间戳组织

最佳模型副本

完成后可选复制最佳模型到 best_overall_model/

生成报告

位置：docs/reports/agent_run_report_<时间戳>.md

内容：
- 每轮的训练配置、评估分数、参数变化
- GPT 的建议和说明（中英文混合）
- 最终总结

故障排除

依赖缺失

bash
pip install -r requirements.txt

或手动安装：

bash
pip install torch sentence-transformers transformers datasets openai


API Key 问题

确保 .env 文件在项目根目录，或环境变量 OPENAI_API_KEY 已设置

bash
$env:OPENAI_API_KEY    # PowerShell 检查


显存不足

降低批大小和样本数：

python
TRAIN_BATCH_SIZE = 4
STSB_TRAIN_SPLIT = "train[:100]"


模型下载慢

sentence-transformers 首次下载模型会较慢，建议在网络环境好时运行

扩展与维护

添加自定义数据集

编辑 core/training.py，在 train_one_round() 中加载自定义数据

修改 GPT 提示词

编辑 agents/gpt_agent.py 中的 prompt 部分

版本控制

config.py 推荐纳入版本控制，记录超参历史

models/ 和 docs/reports/ 建议在 .gitignore 中排除
