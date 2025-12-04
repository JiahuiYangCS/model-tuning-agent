# STSb Auto-Tune Agent v6（简易说明）

简体中文说明，帮助快速上手 `agent_main_v6.py` 演示程序。

**概述**
- 这是一个演示性自动微调 Agent，针对 SentenceTransformer 的 STS-B（STSb）任务，采用“控制变量 + 顺序单变量调参”流程：先由 LLM（GPT）给出初始 base_config 与若干 priority_keys，然后依次对每个优先参数做若干轮单变量调参、每轮训练后让 GPT 给出建议并更新配置，最后生成整体总结并（可选）复制最佳模型。

**重要文件**
- `agent_main_v6.py`: 程序入口，控制自动调参主循环与交互（交互式询问是否继续）。
- `config_and_train.py`: 默认配置、训练流程、评估与可选的 Quora 校验功能。
- `gpt_agent_v6.py`: 与 LLM（通过 `openai_client.py` 初始化的 `client`）交互，包含三个调用：初始计划、每轮建议、新建总体总结。
- `openai_client.py`: 封装 OpenAI 客户端的初始化（读取环境变量 `OPENAI_API_KEY`）。

**先决条件（Prerequisites）**
- Python 3.8+（建议使用虚拟环境）。
- 必要 Python 包（示例安装命令，Windows PowerShell）：

```
python -m pip install openai sentence-transformers datasets transformers matplotlib pandas
# 请根据你的平台单独安装合适版本的 torch（https://pytorch.org/get-started/locally/）。
```

- 在运行前请设置环境变量 `OPENAI_API_KEY`：

```
$env:OPENAI_API_KEY = "sk-xxx"
```

**快速开始（PowerShell）**

1. 在项目目录下（含 `agent_main_v6.py` 文件处）运行：

```
python .\agent_main_v6.py
```

2. 程序会打印初始 config，并调用 GPT 生成 `base_config` 与 `priority_keys`。随后进入每个参数的多轮单变量调参，每轮训练完成后会询问是否继续下一轮（交互式输入 `y/n`）。

3. 训练产生的模型输出保存在 `models/` 子目录下（`OUTPUT_DIR_ROOT`），程序结束时若检测到最佳轮次，会尝试复制该轮模型为 `best_overall_model`。

**运行注意事项**
- 该示例为了便于本地测试默认使用较小数据切片（见 `config_and_train.make_default_config()`），生产/大规模运行请修改相应配置（训练集、batch、epoch、device 等）。
- 如果未正确设置 `OPENAI_API_KEY`，LLM 调用会失败，程序会捕获异常并使用兜底策略继续运行，但会跳过基于 GPT 的建议步骤。
- 若使用 CUDA，请确保 `torch` 已安装对应的 CUDA 版本。

**常见改进方向**
- 将 `TUNABLE_KEYS` 扩展到更多超参或增加更细粒度的搜索策略。
- 自动化记录与可视化（如将 GPT 的建议和每轮结果写入 CSV/MLflow）。

如需我把 `requirements.txt`、启动脚本或 CI 工作流也一并加上，告诉我你的偏好，我可以继续添加。
