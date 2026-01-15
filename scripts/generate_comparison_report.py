#!/usr/bin/env python3
"""Generate a comparison report from multiple model test reports."""
from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_report_info(report_path: Path) -> Dict[str, Any]:
    """Extract key information from a report file."""
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    info = {
        "path": str(report_path),
        "filename": report_path.name,
        "model": "未知",
        "best_score": 0.0,
        "train_runtime": 0.0,
        "train_samples_per_second": 0.0,
    }
    
    # Extract model name from title
    title_match = re.search(r"Model:\s*([^\n]+)", content)
    if title_match:
        info["model"] = title_match.group(1).strip()
    
    # Extract best score
    score_match = re.search(r"\*\*最优分数.*?Best Score.*?\*\*[:\s]*([\d.]+)", content)
    if score_match:
        info["best_score"] = float(score_match.group(1))
    
    # If best_score is 0, try to extract from detailed records
    if info["best_score"] == 0.0:
        main_score_match = re.search(r"\*\*分数 / Score:\*\* ([\d.]+)", content)
        if main_score_match:
            info["best_score"] = float(main_score_match.group(1))
    
    # If still 0, try to extract eval_spearman_cosine from metrics
    if info["best_score"] == 0.0:
        eval_spearman_match = re.search(r"['\"]eval_spearman_cosine['\"]:\s*([\d.]+)", content)
        if eval_spearman_match:
            info["best_score"] = float(eval_spearman_match.group(1))
    
    # Extract training runtime from config section
    runtime_match = re.search(r"'train_runtime':\s*([\d.]+)", content)
    if runtime_match:
        info["train_runtime"] = float(runtime_match.group(1))
    
    samples_match = re.search(r"'train_samples_per_second':\s*([\d.]+)", content)
    if samples_match:
        info["train_samples_per_second"] = float(samples_match.group(1))
    
    return info


def generate_comparison_report(report_paths: List[Path], output_path: Path) -> None:
    """Generate a comparison report from multiple individual reports."""
    reports_info = [extract_report_info(p) for p in report_paths]
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    lines = []
    lines.append(f"# 五个模型对比报告 / Five-Model Comparison Report ({ts})\n")
    lines.append("## 概述 / Overview\n")
    lines.append("本报告对比了五个不同的 LLM 模型在相同训练配置下的表现。")
    lines.append("所有模型均使用相同的 BASE_MODEL（sentence-transformers/all-MiniLM-L6-v2）进行 embedding 微调，")
    lines.append("区别在于使用不同的 LLM（OpenAI GPT 或 OpenRouter 模型）作为 Agent 来分析训练结果并给出建议。\n")
    lines.append("**注意**：由于本次测试使用极小数据集（train[:20], validation[:5]）和单轮训练，")
    lines.append("主要目的是验证流程可行性而非评估模型真实性能。\n")
    
    # Summary table
    lines.append("## 模型对比表 / Model Comparison Table\n")
    lines.append("| 模型 / Model | 最优分数 / Best Score | 训练用时(秒) / Train Time (s) | 样本速度 / Samples/s |")
    lines.append("|---|---:|---:|---:|")
    
    for info in reports_info:
        model = info["model"]
        score = info["best_score"]
        runtime = info["train_runtime"]
        speed = info["train_samples_per_second"]
        lines.append(f"| {model} | {score:.4f} | {runtime:.2f} | {speed:.2f} |")
    
    lines.append("\n")
    
    # Individual model analysis
    lines.append("## 各模型详细分析 / Detailed Analysis by Model\n")
    
    for i, info in enumerate(reports_info, start=1):
        model = info["model"]
        score = info["best_score"]
        runtime = info["train_runtime"]
        speed = info["train_samples_per_second"]
        
        lines.append(f"### {i}. {model}\n")
        lines.append(f"- **最优分数 / Best Score:** {score:.4f}")
        lines.append(f"- **训练用时 / Training Time:** {runtime:.2f} 秒")
        lines.append(f"- **训练速度 / Training Speed:** {speed:.2f} samples/s")
        lines.append(f"- **报告文件 / Report:** {info['filename']}\n")
        
        # Add characteristics based on model type
        if "openai" in info["model"].lower() or "gpt" in info["model"].lower():
            lines.append("**特点 / Characteristics:**")
            lines.append("- 使用 OpenAI 官方 GPT 模型")
            lines.append("- 商业服务，稳定性高")
            lines.append("- 需要付费 API key\n")
        elif "xiaomi" in info["model"].lower() or "mimo" in info["model"].lower():
            lines.append("**特点 / Characteristics:**")
            lines.append("- 小米开源模型")
            lines.append("- 支持大上下文（262k tokens）")
            lines.append("- OpenRouter 免费访问\n")
        elif "nvidia" in info["model"].lower() or "nemotron" in info["model"].lower():
            lines.append("**特点 / Characteristics:**")
            lines.append("- NVIDIA 开源模型")
            lines.append("- 针对推理优化")
            lines.append("- OpenRouter 免费访问\n")
        elif "allen" in info["model"].lower() or "olmo" in info["model"].lower():
            lines.append("**特点 / Characteristics:**")
            lines.append("- Allen AI 开源模型")
            lines.append("- 学术研究背景")
            lines.append("- OpenRouter 免费访问\n")
        elif "gemini" in info["model"].lower() or "google" in info["model"].lower():
            lines.append("**特点 / Characteristics:**")
            lines.append("- Google Gemini 模型")
            lines.append("- 多模态支持")
            lines.append("- OpenRouter 免费访问（实验版本）\n")
        else:
            lines.append("\n")
    
    # Overall comparison
    lines.append("## 综合对比与建议 / Overall Comparison and Recommendations\n")
    
    # Find best/worst performance
    sorted_by_score = sorted(reports_info, key=lambda x: x["best_score"], reverse=True)
    sorted_by_speed = sorted(reports_info, key=lambda x: x["train_samples_per_second"], reverse=True)
    
    lines.append("### 性能排名 / Performance Ranking\n")
    lines.append("**按最优分数排序 / By Best Score:**")
    for i, info in enumerate(sorted_by_score, start=1):
        lines.append(f"{i}. {info['model']}: {info['best_score']:.4f}")
    lines.append("\n**按训练速度排序 / By Training Speed:**")
    for i, info in enumerate(sorted_by_speed, start=1):
        lines.append(f"{i}. {info['model']}: {info['train_samples_per_second']:.2f} samples/s")
    lines.append("\n")
    
    lines.append("### 结论 / Conclusions\n")
    lines.append("1. **准确度方面**：所有模型在本次极小数据集测试中达到了相似的分数（~0.99），")
    lines.append("   这是因为测试规模太小（仅20个训练样本）无法体现模型差异。")
    lines.append("   建议使用更大数据集（如完整 STSb train split）进行更有意义的对比。\n")
    
    lines.append("2. **速度方面**：训练速度的差异主要来自硬件和批处理效率，")
    lines.append("   与使用哪个 LLM 做 Agent 无直接关系（Agent 仅在训练后分析结果）。\n")
    
    lines.append("3. **成本方面**：")
    lines.append("   - OpenAI GPT: 需要付费 API，但服务稳定")
    lines.append("   - OpenRouter 免费模型: 无需付费，适合实验和学习")
    lines.append("   - 建议: 开发测试阶段使用免费模型，生产环境根据需求选择\n")
    
    lines.append("4. **推荐选择 / Recommendations**:")
    lines.append("   - **学习/实验**: 使用 OpenRouter 免费模型（如 xiaomi/mimo-v2-flash）")
    lines.append("   - **生产/商业**: 使用 OpenAI GPT-3.5/GPT-4（稳定性和支持更好）")
    lines.append("   - **研究**: 可尝试不同开源模型（AllenAI OLMo, NVIDIA Nemotron）了解差异\n")
    
    lines.append("### 下一步建议 / Next Steps\n")
    lines.append("1. 使用完整数据集重新测试（修改 STSB_TRAIN_SPLIT 为 'train' 而非 'train[:20]'）")
    lines.append("2. 增加训练轮数（NUM_TRAIN_EPOCHS > 1）观察模型收敛情况")
    lines.append("3. 对比不同 BASE_MODEL（如 all-mpnet-base-v2）的效果")
    lines.append("4. 在真实业务场景中测试 Agent 建议的质量和准确性\n")
    
    lines.append("---\n")
    lines.append(f"**生成时间 / Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**原始报告 / Source Reports:**\n")
    for info in reports_info:
        lines.append(f"- {info['filename']}\n")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"对比报告已生成 / Comparison report generated: {output_path}")


if __name__ == "__main__":
    reports_dir = ROOT / "docs" / "reports"
    
    # Find the most recent 5 reports (or specify them explicitly)
    report_files = sorted(reports_dir.glob("agent_run_report_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if len(report_files) < 5:
        print(f"警告: 只找到 {len(report_files)} 份报告，需要至少 5 份")
        selected_reports = report_files
    else:
        selected_reports = report_files[:5]
    
    print(f"使用以下报告生成对比:")
    for r in selected_reports:
        print(f"  - {r.name}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"comparison_report_5models_{ts}.md"
    
    generate_comparison_report(selected_reports, output_path)
