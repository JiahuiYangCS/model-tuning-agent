快速开始完全指南

第一步: 安装依赖

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch sentence-transformers transformers datasets openai


第二步: 配置 API Key

编辑 setup_api_key.py 文件，找到以下行:

python
API_KEY = ""


粘贴你的 OpenAI API Key:

python
API_KEY = "sk-proj-your-key-here"


然后运行:

powershell
python setup_api_key.py


出现成功提示表示 .env 文件已创建。

第三步: 快速测试（5 分钟）

直接运行（使用默认快速配置）:

powershell
python run.py


程序会自动完成:
- 从 GPT 获取初始建议
- 对 3 个参数各进行 3 轮调优
- 保存最佳模型
- 生成报告到 docs/reports/

查看报告:

powershell
Get-Content docs/reports/agent_run_report_*.md


第四步: 自定义参数调优

编辑 config.py 文件中的 DEFAULT_CONFIG 部分。

关键参数说明:

NUM_TRAIN_EPOCHS        训练轮数（增加可提升效果，也耗时更长）
TRAIN_BATCH_SIZE        批大小（显存充足可改为 16 或 32）
LEARNING_RATE           学习率（通常 1e-5 到 5e-4）
STSB_TRAIN_SPLIT        训练数据量（train[:200] 为 200 样本，可改为 train[:1000] 用 1000 样本）

示例: 用更多数据和更多轮次

python
DEFAULT_CONFIG = {
    "STSB_TRAIN_SPLIT": "train[:1000]",    # 改为 1000 样本
    "NUM_TRAIN_EPOCHS": 3,                  # 改为 3 轮
    "TRAIN_BATCH_SIZE": 16,                 # 改为 16 大小
    ...
}


然后重新运行:

powershell
python run.py


调参规则:

- 改参数后直接运行即可，无需其他操作
- 所有参数在 DEFAULT_CONFIG 中修改
- 修改后自动应用，不需重启环境

常见问题与解决

显存不足

出现 Out of Memory 错误:

python
TRAIN_BATCH_SIZE = 4                    # 改为更小值
GRAD_ACC_STEPS = 2                      # 增加梯度累积


模型加载失败

第一次运行会自动下载 sentence-transformers 模型，可能较慢。保证网络畅通。

GPT API 错误

确保 API Key 有效:

powershell
cat .env


确保网络可访问 OpenAI 服务。

程序卡在某一步

如果一直卡住，可以 Ctrl+C 停止，然后调整参数（减少样本数、减少轮次）再试。

工作流程说明

每次运行 python run.py，程序会:

1. 加载 config.py 中的所有参数
2. 调用 GPT 获取初始建议 (base_config 和要调的参数列表)
3. 对每个参数进行循环:
   - 执行一轮训练（使用当前参数值）
   - 评估模型效果
   - 问 GPT 该参数的新建议
   - 应用 GPT 建议，进入下一轮
4. 所有参数调完后停止
5. 生成报告并保存最佳模型

输出位置

models/                     训练过程中的模型检查点
docs/reports/              每次运行的报告（Markdown 格式）

标准使用流程

快速测试（首次）:

powershell
python setup_api_key.py
python run.py


查看报告:

powershell
ls docs/reports/


修改参数并再次测试:

编辑 config.py，然后:

powershell
python run.py


重复直到满意为止。

高级配置

所有可被 GPT 自动调整的参数列表在 config.py 的 TUNABLE_KEYS 中。
不在此列表中的参数不会被 GPT 修改。

要启用或禁用某些参数的自动调优，编辑 TUNABLE_KEYS:

python
TUNABLE_KEYS = [
    "NUM_TRAIN_EPOCHS",
    "TRAIN_BATCH_SIZE",
    "LEARNING_RATE",
    ...
]


删除某个参数名称就能禁用它的自动调优。

故障排除清单

问题: Import Error
解决: 检查虚拟环境是否激活，依赖是否完整安装

问题: API Key 错误
解决: 重新运行 python setup_api_key.py，确保 .env 文件存在且内容正确

问题: 训练太慢
解决: 减少 STSB_TRAIN_SPLIT 的样本数，改为 train[:100]

问题: 显存溢出
解决: 减少 TRAIN_BATCH_SIZE，改为 4 或 8

问题: 报告生成失败
解决: 检查 docs/ 目录权限，或手动创建 docs/reports/ 目录

快速命令参考

激活虚拟环境:
  .\.venv\Scripts\Activate.ps1

设置 API Key:
  python setup_api_key.py

运行调参:
  python run.py

查看报告:
  cat docs/reports/agent_run_report_<日期>.md

重置所有参数:
  编辑 config.py，重新设置 DEFAULT_CONFIG

删除旧模型:
  rm -r models/
