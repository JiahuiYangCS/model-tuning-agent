#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STSb Auto-Tune Agent v6 主入口脚本 / Main Entry Point

⚡ 一键运行自动调参 Agent

用法 / Usage:
    python run.py

💡 要改配置，编辑 config.py
💡 To modify config, edit config.py
"""

import json
import os
import shutil
from copy import deepcopy
from typing import Dict, Any, List, Optional

# 导入核心模块 / Import core modules
from config import DEFAULT_CONFIG, TUNABLE_KEYS, AGENT_SETTINGS
from core.training import (
    make_default_config,
    export_config_for_agent,
)
from agents.gpt_agent import (
    ask_gpt_for_initial_plan,
    ask_gpt_for_new_config,
    ask_gpt_for_overall_summary,
)
from utils.report_generator import generate_run_report


def apply_new_config(base_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 GPT 返回的 new_config 应用到当前 config（只覆盖已有键）
    Apply new config from GPT (only override existing keys)
    """
    cfg = deepcopy(base_config)
    for k, v in new_config.items():
        if k in cfg:
            cfg[k] = v
    return cfg


def run_agent() -> None:
    """
    主函数：协调整个自动调参流程
    Main orchestration function
    """
    print("=" * 70)
    print("🚀 STSb Auto-Tune Agent v6（控制变量 + 单变量顺序调参）")
    print("=" * 70)
    print()

    # 初始化配置 / Initialize configuration
    from core.training import train_one_round

    current_config = make_default_config()

    print("👉 默认初始超参（仅显示可调参部分）:")
    print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))
    print()

    # 全局最佳统计 / Global best tracking
    best_score: float = -1e9
    best_round: Optional[int] = None
    best_config: Optional[Dict[str, Any]] = None
    best_output_dir: Optional[str] = None

    # 历史记录 / History tracking
    history_for_agent: List[Dict[str, Any]] = []

    # ========= 第 0 步：让 GPT 选出 base_config + priority_keys =========
    base_cfg: Dict[str, Any] = {}
    valid_priority_keys: List[str] = []

    try:
        print("\n===== 步骤 0 / Step 0: 调用 GPT 生成 base_config + priority_keys =====")
        init_plan = ask_gpt_for_initial_plan(
            export_config_for_agent(current_config),
            model=AGENT_SETTINGS.GPT_MODEL,
        )
        base_cfg = init_plan.get("base_config") or {}
        priority_keys = init_plan.get("priority_keys") or []
        comment = init_plan.get("comment", "")

        # 过滤优先级 key / Filter priority keys
        valid_priority_keys: List[str] = []
        for k in priority_keys:
            if isinstance(k, str) and k in TUNABLE_KEYS and k not in valid_priority_keys:
                valid_priority_keys.append(k)

        if not valid_priority_keys:
            # 兜底策略 / Fallback
            valid_priority_keys = TUNABLE_KEYS[:AGENT_SETTINGS.MAX_PRIORITY_PARAMS]

        print("\n===== GPT 对初始策略的说明 / GPT Initial Strategy =====")
        print(comment)
        print("\n👉 建议优先调参顺序 / Priority keys:", valid_priority_keys)

        # 应用 base_config / Apply base config
        if isinstance(base_cfg, dict):
            filtered_base_cfg = {k: v for k, v in base_cfg.items() if k in current_config}
            if filtered_base_cfg:
                current_config = apply_new_config(current_config, filtered_base_cfg)

        print("\n✅ 应用 base_config 后的初始超参 / Initial config after applying base_config:")
        print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\n⚠️  初始计划调用失败，使用默认策略 / Initial plan failed, using default strategy")
        print(f"   错误 / Error: {repr(e)}")
        valid_priority_keys = TUNABLE_KEYS[:AGENT_SETTINGS.MAX_PRIORITY_PARAMS]

    # 最多只调前 N 个参数 / Max parameters to tune
    MAX_PARAMS = min(AGENT_SETTINGS.MAX_PRIORITY_PARAMS, len(valid_priority_keys))
    ROUNDS_PER_PARAM = AGENT_SETTINGS.ROUNDS_PER_PARAM
    global_round_id: int = 0

    # ========= 依次对 priority_keys 做控制变量单变量调参 =========
    # ========= Loop over each priority key for single-variable tuning =========
    for param_index in range(MAX_PARAMS):
        key = valid_priority_keys[param_index]
        print("\n" + "=" * 70)
        print(f"=== 参数 {param_index + 1}/{MAX_PARAMS} / Parameter {param_index + 1}/{MAX_PARAMS}: {key} ===")
        print("=" * 70)

        param_best_score: float = -1e9
        param_best_round: Optional[int] = None
        param_best_value = current_config.get(key, None)

        for inner_round in range(1, ROUNDS_PER_PARAM + 1):
            global_round_id += 1
            print("\n" + "-" * 70)
            print(f"轮次 / Round {global_round_id} - {key} 第 {inner_round}/{ROUNDS_PER_PARAM} 轮")
            print("-" * 70)
            print("当前关键超参 / Current config:")
            print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

            # 训练 / Training
            training_summary, full_metrics = train_one_round(current_config, round_id=global_round_id)
            main_score = float(training_summary["main_score"])
            print(f"\n🔹 本轮分数 / Score: {main_score:.4f}")

            # 记录历史 / Record history
            history_item: Dict[str, Any] = {
                "round_id": global_round_id,
                "tuned_key": key,
                "inner_round_index": inner_round,
                "config_for_agent": export_config_for_agent(current_config),
                "main_score": main_score,
                "metrics": full_metrics,
            }
            history_for_agent.append(history_item)

            # 更新该参数内部的最佳记录 / Update parameter's best score
            cur_value = current_config.get(key, None)
            if main_score > param_best_score:
                param_best_score = main_score
                param_best_round = global_round_id
                param_best_value = cur_value

            # 更新全局最佳记录 / Update global best score
            if main_score > best_score:
                best_score = main_score
                best_round = global_round_id
                best_config = export_config_for_agent(current_config)
                best_output_dir = training_summary.get("output_dir", None)
                print(f"🏆 新的全局最优 / New global best! Round #{best_round}, score={best_score:.4f}")

            # 调用 GPT 获取建议 / Call GPT for suggestions
            try:
                suggestion = ask_gpt_for_new_config(
                    export_config_for_agent(current_config),
                    training_summary,
                    model=AGENT_SETTINGS.GPT_MODEL,
                    history=history_for_agent,
                    primary_key=key,
                )
            except Exception as e:
                print(f"\n⚠️  GPT 调用失败，本参数调参提前结束 / GPT call failed, ending this parameter's tuning")
                print(f"   错误 / Error: {repr(e)}")
                break

            print("\n===== GPT 的评价 / GPT Comment =====")
            print(suggestion["comment"])

            new_cfg_from_agent = suggestion.get("new_config") or {}

            # 应用建议 / Apply suggestion
            if key in new_cfg_from_agent:
                new_val = new_cfg_from_agent[key]
                print(f"\n👉 GPT 建议新值 / Suggested new value: {key} = {new_val!r}")
                current_config = apply_new_config(current_config, {key: new_val})
            else:
                print(f"\n⚠️  GPT 未返回 {key}，保持当前值 / GPT didn't suggest {key}, keeping current value")

            # 用户确认 / User confirmation
            if AGENT_SETTINGS.INTERACTIVE_MODE and inner_round < ROUNDS_PER_PARAM:
                ans = input(f"\n继续下一轮吗？/ Continue next round? (y/n): ").strip().lower()
                if ans not in ("y", "yes", "1", "是", "好"):
                    print("🛑 用户选择结束该参数的调参 / User chose to end this parameter's tuning")
                    break

        # 一个参数的调参结束 / Parameter tuning complete
        if param_best_value is not None:
            print(f"\n✅ 参数 {key} 调参完成 / Parameter {key} tuning complete")
            print(f"   该参数最佳: 轮次 #{param_best_round}, 分数 {param_best_score:.4f}, {key}={param_best_value!r}")
            current_config = apply_new_config(current_config, {key: param_best_value})
            print(f"   已固定 {key} 的最佳值，转向下一参数")
        else:
            print(f"\n⚠️  参数 {key} 无有效记录 / No valid records for {key}, keeping current value")

        # 阶段性总结 / Progress report
        print("\n📌 当前全局最佳 / Current Global Best:")
        print(f"   轮次 #{best_round}, 分数 {best_score:.4f}")
        if best_config is not None:
            print("   对应配置 / Corresponding config:")
            print(json.dumps(best_config, ensure_ascii=False, indent=2))

        # 询问是否继续下一参数 / Ask user about next parameter
        if AGENT_SETTINGS.INTERACTIVE_MODE and param_index < MAX_PARAMS - 1:
            next_key = valid_priority_keys[param_index + 1]
            ans = input(f"\n继续调下一个参数（{next_key}）吗？/ Continue with {next_key}? (y/n): ").strip().lower()
            if ans not in ("y", "yes", "1", "是", "好"):
                print("🛑 用户选择结束调参 / User chose to end tuning")
                break

    # ========= 全部调参流程结束 =========
    # ========= Post-processing =========
    print("\n" + "=" * 70)
    print("✨ 调参完成 / Auto-Tuning Complete")
    print("=" * 70)
    print(f"\n📊 最终结果 / Final Results:")
    print(f"   最佳轮次 / Best round: {best_round}")
    print(f"   最佳分数 / Best score: {best_score:.4f}")
    if best_config is not None:
        print("   最佳配置 / Best config:")
        for k, v in sorted(best_config.items()):
            print(f"      {k}: {v}")

    # 复制最佳模型 / Copy best model
    if best_output_dir is not None:
        try:
            parent_dir = os.path.dirname(best_output_dir.rstrip("/\\"))
            best_overall_dir = os.path.join(parent_dir, "best_overall_model")
            print(f"\n📦 复制最佳模型 / Copying best model...")
            print(f"   从 / From: {best_output_dir}")
            print(f"   到 / To: {best_overall_dir}")
            shutil.copytree(best_output_dir, best_overall_dir, dirs_exist_ok=True)
            print("✅ 模型复制完成 / Best model copied")
        except Exception as e:
            print(f"\n⚠️  复制最佳模型失败（不影响结果）/ Copy failed (doesn't affect results): {repr(e)}")

    # 生成报告 / Generate report
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
        print(f"\n⚠️  生成报告失败（不影响结果）/ Report generation failed: {repr(e)}")

    # 生成整体总结 / Generate overall summary
    try:
        print("\n" + "=" * 70)
        print("📝 GPT 整体总结 / Overall Summary from GPT")
        print("=" * 70)
        overall_comment = ask_gpt_for_overall_summary(
            history_for_agent,
            best_round if best_round is not None else -1,
            float(best_score),
            best_config if best_config is not None else {},
            model=AGENT_SETTINGS.GPT_MODEL,
        )
        print(overall_comment)
    except Exception as e:
        print(f"\n⚠️  生成总结失败（不影响结果）/ Summary generation failed: {repr(e)}")

    print("\n" + "=" * 70)
    print("✨ 所有流程已完成 / All done!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断 / User interrupted")
    except Exception as e:
        print(f"\n\n❌ 程序出错 / Fatal error: {repr(e)}")
        import traceback
        traceback.print_exc()
