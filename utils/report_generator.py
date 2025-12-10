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
    生成一份简单的中英双语报告，总结 agent 在本次运行中每一步做了什么，并把报告写到 `docs/reports/`
    
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
    lines.append("## 概要 / Summary\n")
    lines.append(f"- 优先调参列表 / Priority keys: {priority_keys}\n")
    lines.append(f"- 应用的 base_config / Base config applied: {json.dumps(base_cfg, ensure_ascii=False)}\n")
    lines.append(f"- 历史最佳轮次 / Best round: {best_round}, 最佳分数 / Best score: {best_score:.4f}\n")

    lines.append("## 逐轮记录 / Per-round log (CN/EN explanations)\n")
    if not history:
        lines.append("无历史记录 / No history recorded.\n")
    else:
        for h in history:
            rid = h.get("round_id")
            key = h.get("tuned_key")
            inner = h.get("inner_round_index")
            cfg = h.get("config_for_agent")
            score = h.get("main_score")

            lines.append(f"### 轮次 / Round {rid} — 调参键 / Tuned key: {key} (inner {inner})\n")
            lines.append(f"- 本轮使用的配置 / Config used: {json.dumps(cfg, ensure_ascii=False)}\n")
            lines.append(f"- 本轮主评估分数 / Main score: {score}\n")
            lines.append(f"- 简要说明（中文）/ Brief (CN): 本轮对 `{key}` 进行了单变量调参，记录了当前取值与评估分数，用于比较是否优于之前的取值。\n")
            lines.append(f"- Brief (EN): This round tuned the single key `{key}` and recorded its value and evaluation score to compare with previous values.\n")
            lines.append("\n")

    lines.append("## 结论与下一步建议 / Conclusions & Next Steps\n")
    lines.append("- 结论（中文）/ Conclusion (CN): 请查看 above 的每轮评分，选择评分最高的配置作为最终使用或进一步验证。\n")
    lines.append("- Conclusion (EN): Inspect per-round scores above and pick the best-scoring configuration for final use or further validation.\n")
    lines.append("- 建议 / Suggestion: 可将 best_config 用于后续更长训练，或扩大数据/修改底模以进一步提升。\n")

    # 写文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 打印报告路径并返回
    print(f"\n📄 运行报告已生成 / Report generated: {report_path}")
    return report_path
