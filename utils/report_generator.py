"""
Report Generator module / 报告生成模块

生成训练运行的中英双语 Markdown 报告
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# 修复 Windows 控制台编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def extract_score(main_score: float, metrics: Optional[Dict[str, Any]] = None) -> float:
    """
    提取评估分数：如果 main_score 有效则直接使用，否则尝试从 metrics 中提取
    
    参数 / Args:
        main_score: 主评估分数
        metrics: 完整的评估指标字典
    
    返回 / Returns:
        提取到的分数
    """
    # 如果 main_score 有效（非 0），直接使用
    if main_score and main_score != 0.0:
        return main_score
    
    # 否则尝试从 metrics 中提取
    if metrics:
        # 尝试多个可能的键名（按优先级）
        for key in ["eval_stsb_dev_spearman_cosine", "eval_spearman_cosine", "eval_cosine", "spearman_cosine"]:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (ValueError, TypeError):
                    continue
    
    # 全部失败，返回 0
    return 0.0


def generate_run_report(
    history: List[Dict[str, Any]],
    best_round: Optional[int],
    best_score: float,
    best_config: Optional[Dict[str, Any]],
    priority_keys: List[str],
    base_cfg: Dict[str, Any],
    model_label: Optional[str] = None,
) -> str:
    """
    生成报告：先显示最终结果摘要，后显示每轮详细记录
    
    参数 / Args:
        history: 训练历史记录列表
        best_round: 最佳轮次编号
        best_score: 最佳分数
        best_config: 最佳配置
        priority_keys: 优先调参的键列表
        base_cfg: 初始 base_config
    
    返回 / Returns:
        报告文件的绝对路径
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.getcwd(), "docs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    filename = f"agent_run_report_{ts}.md"
    report_path = os.path.join(report_dir, filename)

    lines: List[str] = []
    title = f"超参数调优报告 / Hyperparameter Tuning Report ({ts})"
    if model_label:
        title = f"{title} — Advisor: {model_label}"
    lines.append(f"# {title}\n")
    
    lines.append("## 📋 项目说明 / Project Description\n")
    lines.append("本报告记录了使用LLM作为'调参顾问'优化sentence-transformers模型的过程。\n")
    lines.append("**重要说明：** LLM只负责给出超参数建议，实际训练在本地GPU上进行。\n")
    lines.append("因此，本报告评测的是**超参数配置的效果**，而非LLM模型本身。\n\n")
    
    # ===== 最终结果摘要（放在最上方）=====
    lines.append("## 最终结果摘要 / Final Results Summary\n")
    if model_label:
        lines.append(f"**LLM顾问 / LLM Advisor:** {model_label}\n")
        lines.append(f"**作用 / Role:** 提供超参数优化建议（不参与训练）\n")
    lines.append(f"**训练模型 / Training Model:** sentence-transformers/all-MiniLM-L6-v2 (本地)\n")
    lines.append(f"**最优轮次 / Best Round:** {best_round}\n")
    
    # 使用辅助函数提取分数
    display_score = best_score
    if display_score == 0.0 and history and best_round:
        for h in history:
            if h.get("round_id") == best_round:
                display_score = extract_score(
                    h.get("main_score", 0.0),
                    h.get("metrics")
                )
                break
    
    lines.append(f"**最优分数 / Best Score:** {display_score:.4f}\n")
    lines.append(f"**调整的参数 / Tuned Parameters:** {', '.join(priority_keys)}\n")
    lines.append(f"\n**最优配置 / Best Configuration:**\n")
    if best_config:
        for k, v in sorted(best_config.items()):
            lines.append(f"  - {k}: {v}\n")
    lines.append(f"\n**初始配置 / Base Config:**\n")
    if base_cfg:
        for k, v in sorted(base_cfg.items()):
            lines.append(f"  - {k}: {v}\n")
    
    # 生成效果改进说明
    lines.append(f"\n**优化效果 / Optimization Result:**\n")
    lines.append(f"通过 {len(history) if history else 0} 轮迭代优化，找到了最佳超参数配置。")
    lines.append(f"最优分数达到 {display_score:.4f}（轮次 #{best_round}）。\n")
    lines.append(f"\n**重要提示：** 该分数反映的是本地sentence-transformers模型在最优超参数下的性能，")
    lines.append(f"与LLM顾问模型的质量无直接关系。LLM仅作为智能助手提供优化建议。\n")
    
    # ===== 详细逐轮记录（后面）=====
    lines.append("## 详细逐轮记录 / Detailed Per-Round Log\n")
    if not history:
        lines.append("无历史记录 / No history recorded.\n")
    else:
        for h in history:
            rid = h.get("round_id")
            key = h.get("tuned_key")
            inner = h.get("inner_round_index")
            cfg = h.get("config_for_agent")
            
            # 使用辅助函数提取分数
            score = extract_score(h.get("main_score", 0.0), h.get("metrics"))

            lines.append(f"### 轮次 / Round {rid} — 参数 / Key: {key} (inner {inner})\n")
            lines.append(f"**配置 / Config:** {json.dumps(cfg, ensure_ascii=False)}\n")
            lines.append(f"**分数 / Score:** {score:.4f}\n")
            
            # 添加metrics信息（如果存在）
            if "metrics" in h and h["metrics"]:
                metrics = h["metrics"]
                # 检查是否有任何需要显示的metrics
                has_metrics = "train_runtime" in metrics or any(
                    key in metrics for key in ["eval_stsb_dev_spearman_cosine", "eval_spearman_cosine", "eval_cosine"]
                )
                if has_metrics:
                    lines.append(f"\n**训练指标 / Training Metrics:**\n")
                    if "train_runtime" in metrics:
                        lines.append(f"- 训练时间 / Runtime: {metrics['train_runtime']:.2f}秒\n")
                    if "train_samples_per_second" in metrics:
                        lines.append(f"- 样本速度 / Speed: {metrics['train_samples_per_second']:.2f} samples/s\n")
                    # 尝试多个可能的分数键名
                    for eval_key in ["eval_stsb_dev_spearman_cosine", "eval_spearman_cosine", "eval_cosine"]:
                        if eval_key in metrics:
                            lines.append(f"- 评估分数 / Eval Score (Spearman): {metrics[eval_key]:.4f}\n")
                            break
            
            lines.append(f"\n**说明 / Note:** 本轮对 `{key}` 进行单变量调优，记录参数值与评估分数以确定最优值。\n")
            lines.append("\n")

    lines.append("## 建议 / Recommendations\n")
    lines.append(f"1. 可将上述最优配置用于更长训练（增加 NUM_TRAIN_EPOCHS）\n")
    lines.append(f"2. 可扩大数据集（修改 STSB_TRAIN_SPLIT）进行验证\n")
    lines.append(f"3. 可继续调整其他未触及的参数以进一步优化\n")

    # 写文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 打印报告路径并返回
    print(f"\n报告已生成 / Report generated: {report_path}")
    return report_path

