#!/usr/bin/env python3
"""Run quick single-round training for a list of LLM models and generate reports.

This script runs a minimal training (small subsets) to validate the pipeline runs end-to-end
and produces a Markdown report for each run (using `utils.report_generator.generate_run_report`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training import make_default_config, train_one_round, export_config_for_agent
from utils.report_generator import generate_run_report
from config import AGENT_SETTINGS

MODELS = [
    ("openai", "gpt-3.5-turbo"),
    ("openrouter", "xiaomi/mimo-v2-flash"),
    ("openrouter", "nvidia/nemotron-3-nano-30b-a3b"),
    ("openrouter", "allenai/olmo-3.1-32b-think"),
    ("openrouter", "google/gemini-2.0-flash-exp:free"),
]

# Minimal quick-run overrides
QUICK_OVERRIDES = {
    "STSB_TRAIN_SPLIT": "train[:20]",
    "STSB_DEV_SPLIT": "validation[:5]",
    "NUM_TRAIN_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 4,
    "EVAL_BATCH_SIZE": 4,
    "EVAL_STEPS": 999999,  # disable intermediate evals for single-epoch small run
    "SAVE_STEPS": 999999,
}


def run_quick_test(source: str, model_id: str) -> Dict[str, Any]:
    # Make config and apply overrides
    cfg = make_default_config()
    for k, v in QUICK_OVERRIDES.items():
        cfg[k] = v

    # Set AGENT settings so logs/report mention the chosen model
    AGENT_SETTINGS.LLM_SOURCE = source
    AGENT_SETTINGS.GPT_MODEL = model_id
    AGENT_SETTINGS.INTERACTIVE_MODE = False

    print(f"\n=== Running quick test for {source}:{model_id} ===")
    # Run a single training round
    summary, metrics = train_one_round(cfg, round_id=1)

    # Build a fake history entry
    history = [
        {
            "round_id": 1,
            "tuned_key": "quick_test",
            "inner_round_index": 1,
            "config_for_agent": export_config_for_agent(cfg),
            "main_score": summary.get("main_score", 0.0),
            "metrics": metrics,
        }
    ]

    # Generate report
    model_label = f"{source}:{model_id}"
    report_path = generate_run_report(history, 1, summary.get("main_score", 0.0), export_config_for_agent(cfg), [], {}, model_label=model_label)

    return {"model": model_id, "summary": summary, "report": report_path}


if __name__ == "__main__":
    results = []
    for src, mid in MODELS:
        try:
            res = run_quick_test(src, mid)
            results.append(res)
        except Exception as e:
            print(f"Error testing {src}:{mid}: {e!r}")

    print("\nAll tests complete. Reports generated:")
    for r in results:
        print(f" - {r['model']}: {r['report']}")
