from agent_main_v6 import generate_run_report

# 构造简单的假历史用于测试（不触发训练或 LLM）
history = [
    {
        "round_id": 1,
        "tuned_key": "LEARNING_RATE",
        "inner_round_index": 1,
        "config_for_agent": {"LEARNING_RATE": 2e-5, "TRAIN_BATCH_SIZE": 8},
        "main_score": 0.7123,
        "metrics": {"loss": 0.5}
    },
    {
        "round_id": 2,
        "tuned_key": "LEARNING_RATE",
        "inner_round_index": 2,
        "config_for_agent": {"LEARNING_RATE": 3e-5, "TRAIN_BATCH_SIZE": 8},
        "main_score": 0.7256,
        "metrics": {"loss": 0.45}
    }
]

priority_keys = ["LEARNING_RATE", "TRAIN_BATCH_SIZE"]
base_cfg = {"LEARNING_RATE": 2e-5}

report_path = generate_run_report(history, best_round=2, best_score=0.7256, best_config={"LEARNING_RATE":3e-5}, priority_keys=priority_keys, base_cfg=base_cfg)
print("Report generated at:", report_path)
