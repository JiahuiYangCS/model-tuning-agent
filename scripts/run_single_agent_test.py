#!/usr/bin/env python3
"""
单一Agent调参测试 / Single Agent Tuning Test

这个脚本的正确逻辑：
1. 选择一个LLM作为"调参顾问"（不是被评测对象）
2. 让这个LLM顾问多轮调整超参数
3. 在本地sentence-transformers模型上测试这些超参数
4. 评测的是"超参数配置"的效果，而不是LLM模型
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AGENT_SETTINGS
from run import run_agent


def main():
    print("\n" + "="*80)
    print("🤖 单一Agent调参测试 / Single Agent Hyperparameter Tuning")
    print("="*80)
    
    print("\n📝 说明 / Description:")
    print("  本测试使用一个LLM作为'调参顾问'，对本地sentence-transformers模型进行")
    print("  超参数优化。LLM只负责给出建议，实际训练在本地进行。")
    print("  评测的是'超参数配置效果'，而不是LLM模型本身。\n")
    
    # 选择LLM作为顾问
    print("可用的LLM顾问 / Available LLM Advisors:")
    advisors = [
        ("openai", "gpt-3.5-turbo"),
        ("openai", "gpt-4"),
        ("openrouter", "google/gemini-2.0-flash-exp:free"),
        ("openrouter", "openai/gpt-4o-mini"),
    ]
    
    for idx, (source, model) in enumerate(advisors, 1):
        cost = "付费/Paid" if source == "openai" and "gpt-4" in model else "免费/Free" if "free" in model else "低成本/Low-cost"
        print(f"  {idx}. {model} ({source}, {cost})")
    
    choice = input("\n请选择LLM顾问编号 (默认1) / Choose advisor number: ").strip()
    try:
        idx = int(choice) if choice else 1
        source, model = advisors[idx - 1]
    except:
        print("无效选择，使用默认")
        source, model = advisors[0]
    
    # 设置Agent配置
    AGENT_SETTINGS.LLM_SOURCE = source
    AGENT_SETTINGS.GPT_MODEL = model
    
    print(f"\n✅ 已选择LLM顾问: {model}")
    print(f"   来源: {source}")
    print("\n开始调参流程...\n")
    
    # 运行Agent调参
    run_agent()
    
    print("\n" + "="*80)
    print("✅ 调参完成 / Tuning Complete")
    print("="*80)
    print("\n💡 提示 / Tips:")
    print("  - 生成的报告在 docs/reports/ 目录")
    print("  - 报告展示的是'超参数调优过程'，而不是LLM对比")
    print("  - 最佳模型保存在 models/best_overall_model/")
    

if __name__ == "__main__":
    main()
