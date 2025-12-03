from typing import Dict, Any, List, Optional
import json

from openai_client import client
from config_and_train import TUNABLE_KEYS


from typing import Dict, Any, List, Optional


def build_agent_input(
    config_for_agent: Dict[str, Any],
    training_summary: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
    primary_key: Optional[str] = None,
) -> str:
    """
    把当前轮的配置 + 训练总结 +（可选）历史多轮信息 + 当前单变量 key 打包成 JSON，发给大模型。
    history 允许为 None，这样可以兼容没有记忆的调用场景。
    """
    payload: Dict[str, Any] = {
        "config": config_for_agent,
        "training_summary": training_summary,
    }
    if history is not None:
        # 为了避免 prompt 过长，只保留最近若干轮历史
        payload["history"] = history[-10:]
    if primary_key is not None:
        payload["primary_key"] = primary_key
    return json.dumps(payload, ensure_ascii=False, indent=2)


def ask_gpt_for_initial_plan(
    config_for_agent: Dict[str, Any],
    model: str = "gpt-5.1",
) -> Dict[str, Any]:
    """
    第 0 步：在真正开始多轮训练前，先让大模型：
    1）基于当前默认 config 给出一份“整体比较稳妥”的 base_config；
    2）在 TUNABLE_KEYS 里选出 1~3 个最值得优先单独调参的 priority_keys（按优先级排序）。

    注意：这里依赖 TUNABLE_KEYS 全局变量（在上一单元格中定义），
    里面包含了所有允许被大模型修改的键（数量可能很多）。
    """
    system_prompt = """
你是一名精通 SentenceTransformer / embedding 微调的中文深度学习工程师。

现在要启动一个“控制变量 + 顺序单变量多轮调参”的自动微调流程，特点是：
- 先从若干可调超参数中挑出最值得关注的前三个（priority_keys，按重要性排序）；
- 对 priority_keys[0] 先做若干轮单变量调参，其它超参数暂时视为固定背景；
- 第一轮参数确定后，再在其基础上，对 priority_keys[1] 做单变量调参；
- 依次类推，最多考虑前三个。

你会收到一个 JSON，其中包含：
- config：当前默认配置（仅包含 TUNABLE_KEYS 子集中的键）
- tunable_keys：允许调参的键名称列表（TUNABLE_KEYS，数量可能 10~20 个以上）

请你：
1. 基于当前 config 给出一份“整体比较稳妥”的 base_config。
   - 只需要在你觉得有必要的键上稍微调整数值；
   - 未出现的键会沿用原 config 的值。
2. 从 tunable_keys 中，按照“对 STSb embedding 微调效果的影响力从大到小”的顺序，选出 1~3 个 priority_keys。
   - priority_keys 必须是 tunable_keys 的子集；
   - 按重要性排序，长度不要超过 3。

你的输出必须是一个合法 JSON，对象格式为：
{
  "comment": "<几句话的中文解释，说明你为什么这样选择>",
  "base_config": {
    "...": "..."
  },
  "priority_keys": ["<key1>", "<key2>", "<key3>"]
}

要求：
- priority_keys 中的每个 key 必须来源于 tunable_keys；
- priority_keys 可以少于 3 个（比如 2 个），但不要超过 3 个；
- base_config 只需要包含你希望修改的键；未出现的键会沿用原 config 的值；
- 不要输出其他无关内容，不要加 Markdown 代码块标记。
    """.strip()

    payload: Dict[str, Any] = {
        "config": config_for_agent,
        "tunable_keys": TUNABLE_KEYS,
    }
    user_input = json.dumps(payload, ensure_ascii=False, indent=2)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
    )
    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回的内容不是合法 JSON，content=\n{content}") from e

    if "base_config" not in data or "priority_keys" not in data:
        raise ValueError(f"LLM 返回 JSON 中缺少 base_config/priority_keys 字段：{data}")
    return data


def ask_gpt_for_new_config(
    config_for_agent: Dict[str, Any],
    training_summary: Dict[str, Any],
    model: str = "gpt-5.1",
    history: Optional[List[Dict[str, Any]]] = None,
    primary_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    调用 GPT，读取当前这一轮的 config + 训练结果（以及最近若干轮 history），
    让它输出一段中文评价 comment 和调整后的 new_config。

    - 普通模式（primary_key=None）下：允许在 TUNABLE_KEYS 里适度调整多个键。
    - 单变量模式（primary_key 不为 None）下：只允许调整 primary_key 对应的这一项，
      其它键必须保持与 config 完全一致（不要写入 new_config 中）。
    """
    system_prompt = """
你是一名精通 SentenceTransformer / embedding 微调的中文深度学习工程师，负责帮人类做「自动调参」。

你会收到一个 JSON，其中包含：
- config：本轮训练前使用的关键超参（子集，仅 TUNABLE_KEYS 中的键）
- training_summary：本轮训练结束后的主评估分数等信息
- history（可选）：最近若干轮的历史记录
- primary_key（可选）：如果存在，表示当前阶段是“单变量调参”，只允许优化这一项。

请你：
1. 用中文简要评价本轮训练结果（例如是否优于上一轮、是否有提升、是否可能过拟合等）。
2. 给出一组新的超参数 new_config。

具体规则：
- 如果 JSON 中提供了 primary_key：
  - new_config 只能改变 primary_key 这一项的数值；
  - 不要修改其它任意键，更不要新增键名；
  - 最好给出相对当前值合理的小幅度调整，而不是剧烈跳变。
- 如果 JSON 中没有 primary_key：
  - 你可以在 TUNABLE_KEYS 内适度调整多个键，但仍建议只动少数几个关键超参。

额外提示：
- 学习率建议在 [1e-5, 5e-4] 范围内微调，不要一下子跳太大。
- epoch 数建议从 1 开始，逐步增加，观察是否持续提升。
- batch_size 受显存限制，不要轻易设得过大。

你的输出必须是一个合法 JSON，对象格式为：
{
  "comment": "<几句话的中文评价>",
  "new_config": {
    "SOME_KEY": "...",
    ...
  }
}
不要输出其他无关内容，不要加 Markdown 代码块标记。
    """.strip()

    user_input = build_agent_input(
        config_for_agent,
        training_summary,
        history=history,
        primary_key=primary_key,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
    )

    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回的内容不是合法 JSON，content=\n{content}") from e

    if "comment" not in data or "new_config" not in data:
        raise ValueError(f"LLM 返回 JSON 中缺少 comment/new_config 字段：{data}")
    return data


def ask_gpt_for_overall_summary(
    history: List[Dict[str, Any]],
    best_round: int,
    best_score: float,
    best_config: Dict[str, Any],
    model: str = "gpt-5.1",
) -> str:
    """
    在所有轮次结束后，请 GPT 用中文给一个“整体总结性评价”。
    返回值是一段简短中文文本，用于打印在控制台。
    """
    system_prompt = """
你是一名中文母语的深度学习工程师，擅长阅读多轮实验日志并给出简明扼要的总结。

你会收到一个 JSON，其中包含：
- history：一个列表，每一项包含：
  - round_id：轮次
  - tuned_key：本轮主要调的 key（如果有）
  - inner_round_index：在该 key 下的第几轮（1~3），如果有
  - config_for_agent：本轮的超参（子集）
  - main_score：本轮主评估分数
  - metrics：其它评估指标（如果有）
- best_round：历史上主评估分数最高的轮次
- best_score：该轮的主评估分数
- best_config：该轮使用的超参（子集）

请你：
1. 用中文概括这几轮调参的大致变化趋势（例如学习率、epoch、batch_size 有哪些变化等）。
2. 用 1~2 句话评价这次自动微调的整体质量（例如“提升明显”“略有提升”“基本没有变化”等）。
3. 给出 1~2 条下一步可以尝试的方向建议（例如继续增大数据量、更换底模、调整损失函数等）。

请直接输出一小段中文自然段文字，不要再输出 JSON 或代码块。
    """.strip()

    payload: Dict[str, Any] = {
        "history": history,
        "best_round": best_round,
        "best_score": best_score,
        "best_config": best_config,
    }

    user_input = json.dumps(payload, ensure_ascii=False, indent=2)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
    )
    content = completion.choices[0].message.content.strip()
    return content