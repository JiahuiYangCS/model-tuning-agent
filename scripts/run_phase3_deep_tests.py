#!/usr/bin/env python3
"""
Phase 3 超深度测试脚本 / Phase 3 Ultra-Deep Testing Script
大幅增加数据量和训练周期，接近完整数据集训练

Phase 1 (快速验证): train[:20], validation[:5], 1 epoch (~1秒/模型)
Phase 2 (深度测试): train[:1500], validation[:300], 3 epochs (~17秒/模型)
Phase 3 (超深度): train[:4500], validation[:900], 6 epochs (~2-3分钟/模型)
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


# Phase 3 超深度测试配置 / Ultra-deep test configuration
PHASE3_CONFIG = {
    "STSB_TRAIN_SPLIT": "train[:4500]",      # 从1500增加到4500 (3倍，接近完整5749)
    "STSB_DEV_SPLIT": "validation[:900]",    # 从300增加到900 (3倍，接近完整1500)
    "NUM_TRAIN_EPOCHS": 6,                   # 从3增加到6 (2倍)
    "TRAIN_BATCH_SIZE": 16,                  # 保持16，避免显存溢出
    "EVAL_BATCH_SIZE": 16,                   # 保持16
    "GRAD_ACC_STEPS": 1,
    "LEARNING_RATE": 2e-5,
    "WARMUP_RATIO": 0.1,
    "EVAL_STRATEGY": "steps",
    "EVAL_STEPS": 100,                       # 每100步评估一次（比Phase2的50更稀疏）
    "LOGGING_STEPS": 50,                     # 每50步记录一次
    "SAVE_STRATEGY": "epoch",
    "SAVE_STEPS": 999999,
    "SAVE_TOTAL_LIMIT": 1,
    "LOGGING_FIRST_STEP": True,
    "ENABLE_TRIPLET_EVAL": False,
    "ENABLE_QUORA_TEST": False,
}


def run_phase3_test(llm_source: str, model_name: str, model_label: str) -> dict:
    """运行单个模型的Phase 3超深度测试"""
    print(f"\n{'='*80}")
    print(f"🚀 开始Phase 3超深度测试 / Starting Phase 3 ultra-deep test")
    print(f"   模型 / Model: {model_label}")
    print(f"{'='*80}\n")
    
    # 设置模型
    AGENT_SETTINGS.LLM_SOURCE = llm_source
    AGENT_SETTINGS.GPT_MODEL = model_name
    
    # 使用Phase 3配置覆盖
    from config import DEFAULT_CONFIG
    test_config = DEFAULT_CONFIG.copy()
    test_config.update(PHASE3_CONFIG)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行训练
        training_summary, metrics = train_one_round(test_config, round_id=1)
        
        # 计算总用时
        total_time = time.time() - start_time
        
        # 获取最终分数
        main_score = metrics.get('eval_stsb_dev_spearman_cosine', 0.0)
        
        # 构建result字典以兼容generate_run_report
        result = {
            'best_score': main_score,
            'best_round': 1,
            'best_config': test_config,
            'all_rounds': [
                {
                    'round_id': 1,
                    'tuned_key': 'phase3_ultra_deep',
                    'inner_round_index': 1,
                    'config_for_agent': test_config,
                    'main_score': main_score,
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
            priority_keys=list(PHASE3_CONFIG.keys()),
            base_cfg={},
            model_label=model_label
        )
        
        print(f"\n✅ 模型 {model_label} Phase 3测试完成")
        print(f"   最优分数 / Best Score: {main_score:.4f}")
        print(f"   总用时 / Total Time: {total_time:.1f}秒 ({total_time/60:.2f}分钟)")
        print(f"   报告 / Report: {report_path}\n")
        
        return {
            "model_label": model_label,
            "score": main_score,
            "time": total_time,
            "report": report_path,
            "success": True,
        }
    
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ 模型 {model_label} Phase 3测试失败")
        print(f"   错误 / Error: {e}")
        print(f"   用时 / Time: {total_time:.1f}秒\n")
        import traceback
        traceback.print_exc()
        
        return {
            "model_label": model_label,
            "score": 0.0,
            "time": total_time,
            "report": None,
            "success": False,
            "error": str(e),
        }


def main():
    print("\n" + "="*80)
    print("🔬 Phase 3 超深度模型测试 / Phase 3 Ultra-Deep Model Testing")
    print("="*80)
    print("\n📊 Phase对比 / Phase Comparison:")
    print("  Phase 1: train[:20], val[:5], 1 epoch → ~1秒/模型")
    print("  Phase 2: train[:1500], val[:300], 3 epochs → ~17秒/模型")
    print("  Phase 3: train[:4500], val[:900], 6 epochs → ~2-3分钟/模型")
    print("\n🎯 Phase 3测试配置 / Phase 3 Configuration:")
    print(f"  训练样本 / Train Samples: 1500 → 4500 (增加 3倍)")
    print(f"  验证样本 / Validation Samples: 300 → 900 (增加 3倍)")
    print(f"  训练轮数 / Epochs: 3 → 6 (增加 2倍)")
    print(f"  批次大小 / Batch Size: 16 (保持不变)")
    print(f"  预计总用时 / Estimated Total Time: 10-15分钟 (5个模型)\n")
    
    user_confirm = input("⚠️  此测试将持续约15分钟，确认开始? (y/n): ")
    if user_confirm.lower() != 'y':
        print("❌ 测试已取消")
        return
    
    # 记录所有结果
    all_results = []
    overall_start = time.time()
    
    # 依次测试每个模型
    for idx, (llm_source, model_name, model_label) in enumerate(MODELS, 1):
        print(f"\n{'='*80}")
        print(f"进度 / Progress: {idx}/{len(MODELS)}")
        print(f"{'='*80}")
        
        result = run_phase3_test(llm_source, model_name, model_label)
        all_results.append(result)
    
    # 总结
    overall_time = time.time() - overall_start
    print("\n" + "="*80)
    print("✅ Phase 3 所有超深度测试完成 / All Phase 3 Tests Completed")
    print("="*80)
    print(f"\n总用时 / Total Time: {overall_time/60:.2f} 分钟 ({overall_time:.1f}秒)")
    print(f"\n测试结果汇总 / Summary:")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    if successful:
        print(f"\n✅ 成功 ({len(successful)}/{len(MODELS)}):")
        for r in successful:
            print(f"  {r['model_label']}: {r['score']:.4f} (用时{r['time']:.1f}秒 = {r['time']/60:.2f}分钟)")
    
    if failed:
        print(f"\n❌ 失败 ({len(failed)}/{len(MODELS)}):")
        for r in failed:
            print(f"  {r['model_label']}: {r.get('error', 'Unknown error')}")
    
    print("\n💡 下一步 / Next Step:")
    print("    运行以下命令生成Phase 3对比报告:")
    print("    python scripts/generate_phase3_comparison_report.py")


if __name__ == "__main__":
    main()
