from copy import deepcopy
import os
import shutil
import json

from typing import Dict, Any, List, Optional

from config_and_train import make_default_config, export_config_for_agent, train_one_round, TUNABLE_KEYS
from gpt_agent_v6 import ask_gpt_for_initial_plan, ask_gpt_for_new_config, ask_gpt_for_overall_summary



def apply_new_config(base_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply GPT-suggested new_config to current config (only overwrite existing keys)."""
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
    """Generate bilingual report and save to docs/reports/"""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.getcwd(), "docs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    filename = f"agent_run_report_{ts}.md"
    report_path = os.path.join(report_dir, filename)

    lines: List[str] = []
    lines.append(f"# Agent Run Report ({ts})\n")
    lines.append("## Summary\n")
    lines.append(f"- Priority keys: {priority_keys}\n")
    lines.append(f"- base_config applied: {json.dumps(base_cfg, ensure_ascii=False)}\n")
    lines.append(f"- Best round: {best_round}, Best score: {best_score:.4f}\n")

    lines.append("## Per-round log\n")
    if not history:
        lines.append("No history recorded.\n")
    else:
        for h in history:
            rid = h.get("round_id")
            key = h.get("tuned_key")
            inner = h.get("inner_round_index")
            cfg = h.get("config_for_agent")
            score = h.get("main_score")

            lines.append(f"### Round {rid} - tuned_key: {key} (inner {inner})\n")
            lines.append(f"- Config used: {json.dumps(cfg, ensure_ascii=False)}\n")
            lines.append(f"- Main score: {score}\n")
            lines.append("\n")

    lines.append("## Conclusions & Next Steps\n")
    lines.append("- Pick the best-scoring configuration for final use or further validation.\n")
    lines.append("- Suggestions: Use best_config for longer training or expand data/model for further improvements.\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[REPORT] Generated at: {report_path}")
    return report_path


def run_agent_v6() -> None:
    """Auto-Tuning Agent v6: Control-variable + Single-variable Sequential Tuning"""
    current_config = make_default_config()

    print("==============================================")
    print("[TOOL] STSb Auto-Tune Agent v6")
    print("==============================================")
    print("[INFO] Default initial parameters:")
    print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

    best_score: float = -1e9
    best_round: Optional[int] = None
    best_config: Optional[Dict[str, Any]] = None
    best_output_dir: Optional[str] = None

    history_for_agent: List[Dict[str, Any]] = []

    base_cfg: Dict[str, Any] = {}
    valid_priority_keys: List[str] = []

    try:
        print("\n===== Step 0: Call GPT for initial plan (base_config + priority_keys) =====")
        init_plan = ask_gpt_for_initial_plan(
            export_config_for_agent(current_config),
            model="gpt-3.5-turbo",
        )
        base_cfg = init_plan.get("base_config") or {}
        priority_keys = init_plan.get("priority_keys") or []
        comment = init_plan.get("comment", "")

        valid_priority_keys: List[str] = []
        for k in priority_keys:
            if isinstance(k, str) and k in TUNABLE_KEYS and k not in valid_priority_keys:
                valid_priority_keys.append(k)

        if not valid_priority_keys:
            valid_priority_keys = TUNABLE_KEYS[:3]

        print("\n===== GPT Initial Strategy =====")
        print(comment)
        print("\n[GPT] Recommended priority_keys:", valid_priority_keys)

        if isinstance(base_cfg, dict):
            filtered_base_cfg = {k: v for k, v in base_cfg.items() if k in current_config}
            if filtered_base_cfg:
                current_config = apply_new_config(current_config, filtered_base_cfg)

        print("\n[SUCCESS] Initial config after applying base_config:")
        print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

    except Exception as e:
        print("\n[WARN] Initial plan call failed, using default config. Error:", repr(e))
        valid_priority_keys = TUNABLE_KEYS[:3]

    MAX_PARAMS = min(3, len(valid_priority_keys))
    ROUNDS_PER_PARAM = 3
    global_round_id: int = 0

    for param_index in range(MAX_PARAMS):
        key = valid_priority_keys[param_index]
        print("\n====================================================")
        print(f"=== Tuning parameter {param_index + 1}: {key} (3 rounds max) ===")

        param_best_score: float = -1e9
        param_best_round: Optional[int] = None
        param_best_value = current_config.get(key, None)

        for inner_round in range(1, ROUNDS_PER_PARAM + 1):
            global_round_id += 1
            print("\n----------------------------------------------------")
            print(f"Parameter {key} round {inner_round}/{ROUNDS_PER_PARAM} (global round #{global_round_id})")
            print("Current config:")
            print(json.dumps(export_config_for_agent(current_config), ensure_ascii=False, indent=2))

            training_summary, full_metrics = train_one_round(current_config, round_id=global_round_id)
            main_score = float(training_summary["main_score"])
            print(f"[SCORE] Round {global_round_id} main_score = {main_score:.4f}")

            history_item: Dict[str, Any] = {
                "round_id": global_round_id,
                "tuned_key": key,
                "inner_round_index": inner_round,
                "config_for_agent": export_config_for_agent(current_config),
                "main_score": main_score,
                "metrics": full_metrics,
            }
            history_for_agent.append(history_item)

            cur_value = current_config.get(key, None)
            if main_score > param_best_score:
                param_best_score = main_score
                param_best_round = global_round_id
                param_best_value = cur_value

            if main_score > best_score:
                best_score = main_score
                best_round = global_round_id
                best_config = export_config_for_agent(current_config)
                best_output_dir = training_summary.get("output_dir", None)
                print(f"[BEST] Global best updated to round #{best_round}, score={best_score:.4f}")

            try:
                suggestion = ask_gpt_for_new_config(
                    export_config_for_agent(current_config),
                    training_summary,
                    model="gpt-3.5-turbo",
                    history=history_for_agent,
                    primary_key=key,
                )
            except Exception as e:
                print("\n[WARN] GPT call failed, ending this parameter's tuning. Error:", repr(e))
                break

            print("\n===== GPT Comment =====")
            print(suggestion["comment"])

            new_cfg_from_agent = suggestion.get("new_config") or {}

            if key in new_cfg_from_agent:
                new_val = new_cfg_from_agent[key]
                print(f"\n[GPT] Suggested new {key} = {new_val!r}")
                current_config = apply_new_config(current_config, {key: new_val})
            else:
                print(f"\n[WARN] GPT new_config missing {key}, skipping update.")

            if inner_round < ROUNDS_PER_PARAM:
                ans = input(f"\nContinue tuning {key} next round? (y/n): ").strip().lower()
                if ans not in ("y", "yes", "1"):
                    print("[STOP] User chose to stop.")
                    break

        if param_best_value is not None:
            print(f"\n[SUCCESS] Parameter {key} tuning done.")
            print(f"   Best for {key}: round #{param_best_round}, score={param_best_score:.4f}, value={param_best_value!r}")
            current_config = apply_new_config(current_config, {key: param_best_value})
        else:
            print(f"\n[WARN] Parameter {key} no valid best value, keeping current.")

        print("\n[CHECKPOINT] Current global best:")
        print(f"   best_round = {best_round}, best_score = {best_score:.4f}")
        if best_config is not None:
            print("   Best config:")
            print(json.dumps(best_config, ensure_ascii=False, indent=2))

        if param_index < MAX_PARAMS - 1:
            next_key = valid_priority_keys[param_index + 1]
            ans = input(f"\nContinue tuning next parameter ({next_key})? (y/n): ").strip().lower()
            if ans not in ("y", "yes", "1"):
                print("[STOP] User chose to stop remaining parameters.")
                break

    print("\n============================")
    print("Auto-Tuning Agent v6 Complete")
    print(f"Best round: {best_round}, Best score: {best_score:.4f}")
    if best_config is not None:
        print("Best config:")
        for k, v in sorted(best_config.items()):
            print(f"  {k}: {v}")

    if best_output_dir is not None:
        try:
            parent_dir = os.path.dirname(best_output_dir.rstrip("/\\"))
            best_overall_dir = os.path.join(parent_dir, "best_overall_model")
            print(f"\n[COPY] Copying best model to {best_overall_dir}")
            shutil.copytree(best_output_dir, best_overall_dir, dirs_exist_ok=True)
            print("[SUCCESS] Best model copied.")
        except Exception as e:
            print("\n[WARN] Error copying model:", repr(e))

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
            print("\n[WARN] Error generating report:", repr(e))
    except Exception:
        pass

    try:
        overall_comment = ask_gpt_for_overall_summary(
            history_for_agent,
            best_round if best_round is not None else -1,
            float(best_score),
            best_config if best_config is not None else {},
            model="gpt-3.5-turbo",
        )
        print("\n===== Overall Summary (GPT) =====")
        print(overall_comment)
    except Exception as e:
        print("\n[WARN] Error generating summary:", repr(e))


if __name__ == "__main__":
    run_agent_v6()
