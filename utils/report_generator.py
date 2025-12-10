"""
Report Generator module / 报告生成模块

生成训练运行的中英双语 Markdown 报告
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


def generate_run_report(
    history: List[Dict[str, Any]],
    best_round: Optional[int],
    best_score: float,
    best_config: Optional[Dict[str, Any]],
    priority_keys: List[str],
    base_cfg: Dict[str, Any],
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
    lines.append(f"# Agent 运行报告 / Agent Run Report ({ts})\n")
    
    # ===== 最终结果摘要（放在最上方）=====
    lines.append("## 最终结果摘要 / Final Results Summary\n")
    lines.append(f"**最优轮次 / Best Round:** {best_round}\n")
    lines.append(f"**最优分数 / Best Score:** {best_score:.4f}\n")
    lines.append(f"**调整的参数 / Tuned Parameters:** {', '.join(priority_keys)}\n")
    lines.append(f"\n**最终配置 / Final Configuration:**\n")
    if best_config:
        for k, v in sorted(best_config.items()):
            lines.append(f"  - {k}: {v}\n")
    lines.append(f"\n**初始建议 / Base Config Applied:**\n")
    if base_cfg:
        for k, v in sorted(base_cfg.items()):
            lines.append(f"  - {k}: {v}\n")
    
    # 生成效果改进说明
    lines.append(f"\n**效果说明 / Performance Note:**\n")
    lines.append(f"本次调参通过 GPT 代理对选定参数进行单变量优化。")
    lines.append(f"从初始配置开始，逐轮测试不同的参数值，保留最优值。")
    lines.append(f"最终获得的最优分数为 {best_score:.4f}（轮次 #{best_round}）。\n")
    
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
            score = h.get("main_score")

            lines.append(f"### 轮次 / Round {rid} — 参数 / Key: {key} (inner {inner})\n")
            lines.append(f"**配置 / Config:** {json.dumps(cfg, ensure_ascii=False)}\n")
            lines.append(f"**分数 / Score:** {score:.4f}\n")
            lines.append(f"**说明 / Note:** 本轮对 `{key}` 进行单变量调优，记录参数值与评估分数以确定最优值。\n")
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

