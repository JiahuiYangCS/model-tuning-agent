from copy import deepcopy
import os
import shutil
import json

from typing import Dict, Any, List, Optional

from config_and_train import make_default_config, export_config_for_agent, train_one_round, TUNABLE_KEYS
from gpt_agent_v6 import ask_gpt_for_initial_plan, ask_gpt_for_new_config, ask_gpt_for_overall_summary



def apply_new_config(base_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """把 GPT 返回的 new_config 应用到当前 config（只覆盖已有键）。"""
    cfg = deepcopy(base_config)
    for k, v in new_config.items():
        if k in cfg:
            cfg[k] = v
    return cfg


def generate_run_report(
    history: List[Dict[str, Any]],
    best_round: Optional[int],
    best_score: float,
    best_config: Optional[Dict[str, Any]],
    priority_keys: List[str],
    base_cfg: Dict[str, Any],
) -> str:
    """
    生成一份简单的中英双语报告，总结 agent 在本次运行中每一步做了什么，并把报告写到 `docs/reports/`。
    返回报告文件的绝对路径。
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.getcwd(), "docs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    filename = f"agent_run_report_{ts}.md"
    report_path = os.path.join(report_dir, filename)

    lines: List[str] = []
    lines.append(f"# Agent 运行报告 / Agent Run Report ({ts})\n")
    lines.append("## 概要 / Summary\n")
    lines.append(f"- 优先调参列表 Priority keys: {priority_keys}\n")
    lines.append(f"- 应用的 base_config (只列出被修改或建议的键) / base_config applied: {json.dumps(base_cfg, ensure_ascii=False)}\n")
    lines.append(f"- 历史最佳轮次 Best round: {best_round}, Best score: {best_score:.4f}\n")

    lines.append("## 逐轮记录 / Per-round log (simple CN/EN explanations)\n")
    if not history:
        lines.append("无历史记录 / No history recorded.\n")
    else:
        for h in history:
            rid = h.get("round_id")
            key = h.get("tuned_key")
            inner = h.get("inner_round_index")
            cfg = h.get("config_for_agent")
            score = h.get("main_score")

            lines.append(f"### 轮次 Round {rid} — 调参键 tuned_key: {key} (inner {inner})\n")
            lines.append(f"- 本轮使用的配置 / Config used: {json.dumps(cfg, ensure_ascii=False)}\n")
            lines.append(f"- 本轮主评估分数 / Main score: {score}\n")
            # 简单易懂的中英说明
            lines.append(f"- 简要说明（中文）：本轮对 `{key}` 进行了单变量调参，记录了当前取值与评估分数，用于比较是否优于之前的取值。\n")
            lines.append(f"- Brief (EN): This round tuned the single key `{key}` and recorded its value and evaluation score to compare with previous values.\n")
            lines.append("\n")

    lines.append("## 结论与下一步建议 / Conclusions & Next Steps\n")
    lines.append("- 结论（中文）：请查看 above 的每轮评分，选择评分最高的配置作为最终使用或进一步验证。\n")
    lines.append("- Conclusion (EN): Inspect per-round scores above and pick the best-scoring configuration for final use or further validation.\n")
    lines.append("- 建议 / Suggestion: 可将 best_config 用于后续更长训练，或扩大数据/修改底模以进一步提升。\n")

    # 写文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 打印报告路径并返回
    print(f"\n📄 运行报告已生成：{report_path}")
    return report_path


# =============== 主循环 Agent Demo v6 ===============
# 特点：
# 1）保留 v5 中所有 TUNABLE_KEYS，不做删减；
# 2）由大模型先给出 base_config + priority_keys（最多 3 个关键参数）；
# 3）对每个 priority_key 采用「控制变量法」做 3 轮单变量调参；
# 4）每个参数内部结束后，固定该参数的最佳取值，依次调下一个参数；
# 5）整个过程仍然是一个交互式 Agent，会多次与 GPT 对话，并多次询问用户是否继续。


def run_agent_v6() -> None:
    # 建议首次测试时仅调几个参数，每个参数 3 轮，整体跑通流程即可
    current_config = make_default_config()

    print("==============================================")
    print("🔧 STSb Auto-Tune Agent v6（控制变量 + 单变量顺序调参）")
    print("==============================================")
    print("👉 默认初始超参（export_config_for_agent）：")
    print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

    # 全局最佳统计
    best_score: float = -1e9
    best_round: Optional[int] = None
    best_config: Optional[Dict[str, Any]] = None
    best_output_dir: Optional[str] = None  # 记录最佳轮次对应的输出目录

    # 历史记录，提供给 GPT + 最终总结
    history_for_agent: List[Dict[str, Any]] = []

    # ========= 第 0 步：让 GPT 选出 base_config + priority_keys =========
    # prepare defaults so they exist even if GPT call fails
    base_cfg: Dict[str, Any] = {}
    valid_priority_keys: List[str] = []

    try:
        print("\n===== 第 0 步：调用 GPT 生成 base_config + priority_keys（最多 3 个） =====")
        init_plan = ask_gpt_for_initial_plan(
            export_config_for_agent(current_config),
            model="gpt-5.1",
        )
        base_cfg = init_plan.get("base_config") or {}
        priority_keys = init_plan.get("priority_keys") or []
        comment = init_plan.get("comment", "")

        # 过滤优先级 key（必须在 TUNABLE_KEYS 中）
        valid_priority_keys: List[str] = []
        for k in priority_keys:
            if isinstance(k, str) and k in TUNABLE_KEYS and k not in valid_priority_keys:
                valid_priority_keys.append(k)

        if not valid_priority_keys:
            # 如果大模型没有给出有效结果，就从 TUNABLE_KEYS 里简单选前 3 个兜底
            valid_priority_keys = TUNABLE_KEYS[:3]

        print("\n===== GPT 对初始策略的说明 =====")
        print(comment)
        print("\n👉 大模型建议优先调参顺序 priority_keys：", valid_priority_keys)

        # 应用 base_config 到当前 config（只覆盖已有键）
        if isinstance(base_cfg, dict):
            filtered_base_cfg = {k: v for k, v in base_cfg.items() if k in current_config}
            if filtered_base_cfg:
                current_config = apply_new_config(current_config, filtered_base_cfg)

        print("\n✅ 应用 base_config 后的初始超参：")
        print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

    except Exception as e:
        print("\n⚠️ 初始计划调用失败，将使用 make_default_config() 且优先顺序为 TUNABLE_KEYS 的前 3 个。错误信息：", repr(e))
        valid_priority_keys = TUNABLE_KEYS[:3]

    # 最多只调前 3 个参数（如果 GPT 少给，就按实际个数）
    MAX_PARAMS = min(3, len(valid_priority_keys))
    ROUNDS_PER_PARAM = 3
    global_round_id: int = 0

    # ========= 依次对 priority_keys 做控制变量单变量调参 =========
    for param_index in range(MAX_PARAMS):
        key = valid_priority_keys[param_index]
        print("\n====================================================")
        print(f"=== 开始针对第 {param_index + 1} 个重点参数：{key} 做控制变量三轮调参 ===")
        print("（其它超参数在本阶段视为固定背景，仅微调这一项）")

        param_best_score: float = -1e9
        param_best_round: Optional[int] = None
        param_best_value = current_config.get(key, None)

        for inner_round in range(1, ROUNDS_PER_PARAM + 1):
            global_round_id += 1
            print("\n----------------------------------------------------")
            print(f"参数 {key} 第 {inner_round}/{ROUNDS_PER_PARAM} 轮（全局轮次 #{global_round_id}）")
            print("当前关键超参（export_config_for_agent）：")
            print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

            # 1. 本轮训练
            training_summary, full_metrics = train_one_round(current_config, round_id=global_round_id)
            main_score = float(training_summary["main_score"])
            print(f"🔹 本轮 main_score = {main_score:.4f}")

            # 记录本轮历史
            history_item: Dict[str, Any] = {
                "round_id": global_round_id,
                "tuned_key": key,
                "inner_round_index": inner_round,
                "config_for_agent": export_config_for_agent(current_config),
                "main_score": main_score,
                "metrics": full_metrics,
            }
            history_for_agent.append(history_item)

            # 更新该参数内部的最佳记录
            cur_value = current_config.get(key, None)
            if main_score > param_best_score:
                param_best_score = main_score
                param_best_round = global_round_id
                param_best_value = cur_value

            # 更新全局最佳记录
            if main_score > best_score:
                best_score = main_score
                best_round = global_round_id
                best_config = export_config_for_agent(current_config)
                best_output_dir = training_summary.get("output_dir", None)
                print(f"🏆 全局最佳轮次更新为 #{best_round}, best_score={best_score:.4f}")

            # 2. 让 GPT 基于当前结果，只调这一项 key
            try:
                suggestion = ask_gpt_for_new_config(
                    export_config_for_agent(current_config),
                    training_summary,
                    model="gpt-5.1",
                    history=history_for_agent,
                    primary_key=key,
                )
            except Exception as e:
                print("\n⚠️ 调用 GPT 获取新配置失败，将提前结束该参数的调参。错误信息：", repr(e))
                break

            print("\n===== GPT 对本轮的中文评价 comment =====")
            print(suggestion["comment"])

            new_cfg_from_agent = suggestion.get("new_config") or {}

            # 只允许使用该 key 的建议，控制变量法
            if key in new_cfg_from_agent:
                new_val = new_cfg_from_agent[key]
                print(f"\n👉 GPT 建议新的 {key} = {new_val!r}")
                current_config = apply_new_config(current_config, {key: new_val})
            else:
                print(f"\n⚠️ GPT 返回的 new_config 中没有 {key}，本轮结束后不更新该参数。")

            # 人工确认是否继续该参数下一个 inner round
            if inner_round < ROUNDS_PER_PARAM:
                ans = input(f"\n是否继续针对 {key} 进行下一轮单变量调参？(y/n)：").strip().lower()
                if ans not in ("y", "yes", "1", "是", "好"):
                    print("🛑 用户选择提前结束该参数的后续轮数。")
                    break

        # -------- 一个参数内部的 3 轮结束后，固定最佳取值 --------
        if param_best_value is not None:
            print("\n✅ 参数 {key} 的三轮调参已完成。".replace("{key}", key))
            print(f"   该参数内部最佳轮次: 全局 #{param_best_round}, best_score={param_best_score:.4f}, 最佳 {key}={param_best_value!r}")
            # 固定为该参数的最佳取值，作为接下来参数调参的基础
            current_config = apply_new_config(current_config, {key: param_best_value})
            print("   已将该最佳取值写回当前全局配置。")
        else:
            print(f"\n⚠️ 参数 {key} 没有得到有效的最佳值记录，将保留当前配置中的取值。")

        # 给用户一个阶段性汇报
        print("\n📌 当前为止的历史全局最佳：")
        print(f"   best_round = {best_round}, best_score = {best_score:.4f}")
        if best_config is not None:
            print("   对应超参：")
            print(json.dumps(best_config, ensure_ascii=False, indent=2))

        # 询问是否进入下一个参数
        if param_index < MAX_PARAMS - 1:
            next_key = valid_priority_keys[param_index + 1]
            ans = input(f"\n是否继续调下一个参数（{next_key}）？(y/n)：").strip().lower()
            if ans not in ("y", "yes", "1", "是", "好"):
                print("🛑 用户选择提前结束，不再调后续参数。")
                break

    # ========= 全部调参流程结束 =========
    print("\n============================")
    print("Auto-Tuning Agent v6 结束")
    print(f"历史最佳轮次: {best_round}, 历史最佳主评估分数: {best_score:.4f}")
    if best_config is not None:
        print("历史最佳轮次使用的超参（子集）：")
        for k, v in sorted(best_config.items()):
            print(f"{k}: {v}")

    # 如果记录到了最佳模型对应的输出目录，可以尝试自动复制一份“总最佳模型”
    if best_output_dir is not None:
        try:
            parent_dir = os.path.dirname(best_output_dir.rstrip("/\\"))
            best_overall_dir = os.path.join(parent_dir, "best_overall_model")
            print(f"\n📦 正在将最佳轮次模型从\n  {best_output_dir}\n复制到\n  {best_overall_dir}")
            shutil.copytree(best_output_dir, best_overall_dir, dirs_exist_ok=True)
            print("✅ 最佳模型权重已复制完成。")
        except Exception as e:
            print("\n⚠️ 复制最佳模型权重时出错（不影响训练结果），错误信息：", repr(e))
    # 生成一份简单中英双语运行报告（写入 docs/reports/）
    try:
        try:
            report_path = generate_run_report(
                history_for_agent,
                best_round,
                float(best_score),
                best_config,
                valid_priority_keys,
                base_cfg,
            )
        except Exception as e:
            print("\n⚠️ 生成运行报告时出错（不影响训练结果），错误信息：", repr(e))
    except Exception:
        pass

    # 最后：生成一次整体总结
    try:
        overall_comment = ask_gpt_for_overall_summary(
            history_for_agent,
            best_round if best_round is not None else -1,
            float(best_score),
            best_config if best_config is not None else {},
            model="gpt-5.1",
        )
        print("\n===== 本次多轮自动调参的整体总结（GPT） =====")
        print(overall_comment)
    except Exception as e:
        print("\n⚠️ 生成整体总结时出错（不影响训练结果），错误信息：", repr(e))


# 直接在 Notebook 中运行这一行即可启动 v6 Agent 流程
# 注意：已移除顶层自动执行调用，确保安全导入模块用于测试或其他用途。
if __name__ == "__main__":
    # 直接从命令行运行： python agent_main_v6.py
    run_agent_v6()
