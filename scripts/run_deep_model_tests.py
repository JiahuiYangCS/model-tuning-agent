#!/usr/bin/env python3
"""
深度测试脚本 / Deep Testing Script
增加数据量和训练周期，对5个模型进行深度性能测试

Phase 1 (快速验证): train[:20], validation[:5], 1 epoch
Phase 2 (深度测试): train[:1500], validation[:300], 3 epochs
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AGENT_SETTINGS
from core.training import train_one_round
from utils.report_generator import generate_run_report


# 测试模型列表 / Models to test
MODELS = [
    ("openai", "gpt-3.5-turbo", "openai:gpt-3.5-turbo"),
    ("openrouter", "xiaomi/mimo-v2-flash", "openrouter:xiaomi/mimo-v2-flash"),
    ("openrouter", "nvidia/nemotron-3-nano-30b-a3b", "openrouter:nvidia/nemotron-3-nano-30b-a3b"),
    ("openrouter", "allenai/olmo-3.1-32b-think", "openrouter:allenai/olmo-3.1-32b-think"),
    ("openrouter", "google/gemini-2.0-flash-exp:free", "openrouter:google/gemini-2.0-flash-exp:free"),
]


# 深度测试配置 / Deep test configuration
DEEP_TEST_CONFIG = {
    "STSB_TRAIN_SPLIT": "train[:1500]",      # 从20增加到1500 (75倍)
    "STSB_DEV_SPLIT": "validation[:300]",    # 从5增加到300 (60倍)
    "NUM_TRAIN_EPOCHS": 3,                   # 从1增加到3 (3倍)
    "TRAIN_BATCH_SIZE": 16,                  # 从4增加到16 (4倍，利用GPU)
    "EVAL_BATCH_SIZE": 16,                   # 从4增加到16
    "GRAD_ACC_STEPS": 1,
    "LEARNING_RATE": 2e-5,
    "WARMUP_RATIO": 0.1,
    "EVAL_STRATEGY": "steps",
    "EVAL_STEPS": 50,                        # 每50步评估一次
    "LOGGING_STEPS": 20,
    "SAVE_STRATEGY": "epoch",
    "SAVE_STEPS": 999999,
    "SAVE_TOTAL_LIMIT": 1,
    "LOGGING_FIRST_STEP": True,
    "ENABLE_TRIPLET_EVAL": False,
    "ENABLE_QUORA_TEST": False,
}


def run_deep_test(llm_source: str, model_name: str, model_label: str) -> dict:
    """运行单个模型的深度测试"""
    print(f"\n{'='*80}")
    print(f"🚀 开始深度测试模型 / Starting deep test for: {model_label}")
    print(f"{'='*80}\n")
    
    # 设置模型
    AGENT_SETTINGS.LLM_SOURCE = llm_source
    AGENT_SETTINGS.GPT_MODEL = model_name
    
    # 使用深度测试配置覆盖
    from config import DEFAULT_CONFIG
    test_config = DEFAULT_CONFIG.copy()
    test_config.update(DEEP_TEST_CONFIG)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行训练（使用train_one_round）
    training_summary, metrics = train_one_round(test_config, round_id=1)
    
    # 计算总用时
    total_time = time.time() - start_time
    
    # 构建result字典以兼容generate_run_report
    main_score = metrics.get('eval_stsb_dev_spearman_cosine', 0.0)
    result = {
        'best_score': main_score,
        'best_round': 1,
        'best_config': test_config,
        'all_rounds': [
            {
                'round_id': 1,
                'tuned_key': 'deep_test',
                'inner_round_index': 1,
                'config_for_agent': test_config,
                'main_score': main_score,  # 使用main_score字段名
                'metrics': metrics,
                'summary': training_summary,
            }
        ],
        'final_config': test_config,
        'base_config': {},
    }
    
    # 生成报告（使用正确的参数）
    report_path = generate_run_report(
        history=result['all_rounds'],
        best_round=result['best_round'],
        best_score=result['best_score'],
        best_config=result['best_config'],
        priority_keys=list(DEEP_TEST_CONFIG.keys()),
        base_cfg={},
        model_label=model_label
    )
    
    print(f"\n✅ 模型 {model_label} 深度测试完成")
    print(f"   最优分数 / Best Score: {result['best_score']:.4f}")
    print(f"   总用时 / Total Time: {total_time:.2f}秒")
    print(f"   报告 / Report: {report_path}\n")
    
    return {
        "model_label": model_label,
        "score": result['best_score'],
        "time": total_time,
        "report": report_path,
    }


def main():
    print("\n" + "="*80)
    print("🔬 深度模型测试 / Deep Model Testing")
    print("="*80)
    print("\n📊 测试配置 / Test Configuration:")
    print(f"  训练样本 / Train Samples: 20 → 1500 (增加 75倍)")
    print(f"  验证样本 / Validation Samples: 5 → 300 (增加 60倍)")
    print(f"  训练轮数 / Epochs: 1 → 3 (增加 3倍)")
    print(f"  批次大小 / Batch Size: 4 → 16 (增加 4倍)")
    print(f"  预计每个模型用时 / Estimated Time per Model: 2-5分钟\n")
    
    # 记录所有结果
    all_results = []
    overall_start = time.time()
    
    # 依次测试每个模型
    for llm_source, model_name, model_label in MODELS:
        try:
            result = run_deep_test(llm_source, model_name, model_label)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 模型 {model_label} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    overall_time = time.time() - overall_start
    print("\n" + "="*80)
    print("✅ 所有深度测试完成 / All Deep Tests Completed")
    print("="*80)
    print(f"\n总用时 / Total Time: {overall_time/60:.2f} 分钟")
    print(f"\n测试结果汇总 / Summary:")
    for r in all_results:
        print(f"  {r['model_label']}: {r['score']:.4f} (用时{r['time']:.1f}秒)")
    
    print("\n💡 提示：运行以下命令生成深度对比报告 / Tip: Run to generate deep comparison report:")
    print("    python scripts/generate_deep_comparison_report.py")


if __name__ == "__main__":
    main()
