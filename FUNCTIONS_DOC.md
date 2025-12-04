# 函数与模块说明（逐函数解释）

下列文档为仓库中每个主要函数的作用说明、参数、返回值、使用注意与实现要点。文档以中文为主，便于快速阅读和维护。

----------

**文件**：`openai_client.py`
- 说明：该文件仅负责初始化 OpenAI 客户端实例 `client`，并从环境变量中读取 API Key（由 OpenAI 官方 Python 客户端库 `openai` 提供的 `OpenAI` 类）。
- 主要符号：
  - `client`：已初始化的 `OpenAI` 客户端，全局复用供其它模块（如 `gpt_agent_v6.py`）调用。

注意：运行前请设置环境变量 `OPENAI_API_KEY`（PowerShell 示例：`$env:OPENAI_API_KEY = "sk-xxx"`）。如果该变量缺失或无效，后续调用 LLM 接口会抛出异常。

----------

**文件**：`gpt_agent_v6.py`

- `build_agent_input(config_for_agent: Dict[str, Any], training_summary: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None, primary_key: Optional[str] = None) -> str`
  - 作用：把当前轮的配置（供 LLM 参考）、训练总结、可选历史与当前正在调的主键打包为 JSON 字符串，作为用户输入发送给 LLM。该函数用于统一构造 prompt 中的用户输入部分。
  - 参数：
    - `config_for_agent`：当前要给 LLM 的配置视图（通常由 `export_config_for_agent` 提供）。
    - `training_summary`：本轮训练后的 summary（包含 `main_score`、`output_dir` 等字段）。
    - `history`：可选，最近若干轮历史记录列表（用来提供上下文）。
    - `primary_key`：可选，表示正在做单变量调参的键名；若提供，LLM 会被要求只修改该键。
  - 返回：格式化后的 JSON 字符串（保证中文可读性，使用 `ensure_ascii=False`）。

- `ask_gpt_for_initial_plan(config_for_agent: Dict[str, Any], model: str = "gpt-5.1") -> Dict[str, Any]`
  - 作用：在自动微调流程开始前，调用 LLM 让其返回：
    1. `base_config`：相对当前 config 的一份“更稳妥”的初始配置建议（只包含要修改的键）；
    2. `priority_keys`：按重要性排序的 1~3 个最值得优先调参的键名。
  - 参数：
    - `config_for_agent`：给 LLM 的当前配置视图。
    - `model`：要调用的 LLM 模型名称（默认为 `gpt-5.1`）。
  - 返回：一个包含 `comment`（中文说明）、`base_config`（dict）、`priority_keys`（list）的字典。
  - 错误处理：若 LLM 返回不可解析的 JSON 或缺少必要字段，会抛出错误；调用方应捕获异常并采取兜底策略。

- `ask_gpt_for_new_config(config_for_agent: Dict[str, Any], training_summary: Dict[str, Any], model: str = "gpt-5.1", history: Optional[List[Dict[str, Any]]] = None, primary_key: Optional[str] = None) -> Dict[str, Any]`
  - 作用：每轮训练结束后调用 LLM，让其基于本轮结果（与历史）给出中文评价 `comment` 与新的超参建议 `new_config`。
  - 特殊行为：当传入 `primary_key` 时，LLM 被约束只能在 `new_config` 中修改该键对应的值（单变量模式）。
  - 返回：包含 `comment`（中文字符串）与 `new_config`（字典）的对象。
  - 注意：调用者需要验证 `new_config` 的内容以避免误修改其它键。

- `ask_gpt_for_overall_summary(history: List[Dict[str, Any]], best_round: int, best_score: float, best_config: Dict[str, Any], model: str = "gpt-5.1") -> str`
  - 作用：在整个自动调参流程结束后，让 LLM 给出一段中文的整体总结（包含趋势、质量评估、后续建议）。
  - 参数：
    - `history`：所有轮次（或最近切片）的历史条目列表。
    - `best_round`, `best_score`, `best_config`：用于提醒 LLM 哪一轮是最优并给出上下文。
  - 返回：一段简短的中文自然段文本，可直接打印。

实现要点（`gpt_agent_v6.py`）：
- 所有调用都严格要求 LLM 返回 JSON（或纯文本在总体总结中），代码会尝试解析并在解析失败时抛出异常以便上层处理。请注意 LLM 输出格式可能不稳定，调用方应做好兜底逻辑。

----------

**文件**：`config_and_train.py`

- `make_default_config() -> Dict[str, Any]`
  - 作用：返回一份演示用的默认配置（缩小版），包含可开关项、模型名、数据切片、训练超参及保存/记录选项。
  - 返回：配置字典，结构清晰易改，便于交互式调参。

- `TUNABLE_KEYS`（全局变量）
  - 作用：列出允许被 LLM 修改（或 Agent 调整）的键名列表。`export_config_for_agent` 会基于此从当前 config 中抽取子集供 LLM 查看。

- `export_config_for_agent(config: Dict[str, Any]) -> Dict[str, Any]`
  - 作用：根据 `TUNABLE_KEYS` 从完整 `config` 中抽取一份子集，作为发给 LLM 的安全视图，避免泄露或传入非允许修改的字段。
  - 返回：只包含 `TUNABLE_KEYS` 中键的字典（若存在于 `config` 中）。

- `set_global_seed(seed: int = 42)`
  - 作用：设置 Python、NumPy、Torch 随机种子，包含 GPU 的种子设置，保证可复现性。

- `train_one_round(config: Dict[str, Any], round_id: int = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]`
  - 作用：基于当前 `config` 进行一次完整微调训练流程并评估，返回 `summary` 与 `metrics`。
  - 主要流程：
    1. 设置随机种子与设备（`cuda`/`cpu`）；
    2. 构造输出目录并打印运行信息；
    3. 加载 `sentence-transformers/stsb` 的训练与开发切片；
    4. 初始化 `SentenceTransformer`、损失（`CoSENTLoss`）与 `EmbeddingSimilarityEvaluator`；
    5. 构造 `SentenceTransformerTrainingArguments`（由 `config` 驱动），初始化 `SentenceTransformerTrainer` 并调用 `.train()`；
    6. 保存模型并用 evaluator 计算主评估分数 `main_score`（兼容 dict/float 返回值）；
    7. 返回 `summary`（含 `main_score`、`output_dir` 等）与 `metrics`。
  - 返回：(`summary`, `metrics`) 两个字典。
  - 注意事项：
    - 该函数会真正触发训练，可能消耗显存、时间与外部资源；默认 `make_default_config()` 使用了较小数据切片，便于本地测试。

- `run_quora_test_if_enabled(config: Dict[str, Any], model: SentenceTransformer)`
  - 作用：当 `ENABLE_QUORA_TEST=True` 时，加载 Quora 数据集的指定切片，利用 `model` 对问对进行嵌入并统计重复/非重复对的平均相似度，同时绘制直方图以供人工检查。
  - 注意：该函数使用 `matplotlib`、`pandas`，并会执行模型推理，可能较慢。

----------

**文件**：`agent_main_v6.py`

- `apply_new_config(base_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]`
  - 作用：把 LLM 返回的 `new_config` 合并回当前运行时配置 `base_config`，但只覆盖 `base_config` 中已存在的键（安全合并，不新增未知键）。
  - 返回：合并后的配置副本（使用 `deepcopy` 实现），以避免原对象被意外修改。

- `run_agent_v6() -> None`
  - 作用：该脚本的主函数，负责整个自动调参 Agent 的控制流，包括：
    1. 载入默认配置；
    2. 调用 `ask_gpt_for_initial_plan` 获取 `base_config` 与 `priority_keys`；
    3. 对最多前三个 `priority_keys` 逐个执行 “控制变量 + 单变量多轮调参” 循环（默认 `ROUNDS_PER_PARAM=3`）：
       - 每轮调用 `train_one_round` 执行训练；记录 `main_score` 并更新参数/全局最佳；
       - 每轮结束后调用 `ask_gpt_for_new_config`（带 `primary_key`）请求仅对当前 key 的建议；
       - 将 LLM 建议（若包含该 key）应用到当前配置；
       - 在每轮之间询问用户是否继续（交互式 `input`）。
    4. 每个参数所有内部轮次结束后，将该参数的「内部最佳取值」固定写回 `current_config`；
    5. 全部参数调完后，尝试复制最佳模型到 `best_overall_model` 目录并调用 `ask_gpt_for_overall_summary` 打印总体结论。
  - 重要变量：`best_score`, `best_round`, `best_config`, `history_for_agent`（用于传给 LLM 的历史记录）。
  - 运行方式：脚本结尾处默认会调用 `run_agent_v6()`，可直接 `python agent_main_v6.py` 运行；在交互式 Notebook 也可直接导入并执行。
  - 注意：脚本是交互式的，会在每个参数内部轮数之间向用户确认是否继续；若想全自动运行，可修改或移除 `input()` 部分。

----------

附加说明与维护建议：
- 对 LLM 返回内容要做严格校验：当前实现主要校验 JSON 结构，但没有深度校验数值范围。建议在 `apply_new_config` 或调用处再加入类型/范围校验（如 `LEARNING_RATE` 范围、`TRAIN_BATCH_SIZE` 为整数等）。
- 若要把流程做成完全无人值守，请在 `agent_main_v6.py` 中增加一个 `--non-interactive` 参数并实现超时或默认继续策略。
- 建议把 `TUNABLE_KEYS` 和默认配置抽象到一个 YAML/JSON 文件，便于外部系统或 UI 调整。

如果你希望我把这些校验或 CLI 参数化的改动实现为代码，我可以继续按需修改文件并运行简单静态检查。
