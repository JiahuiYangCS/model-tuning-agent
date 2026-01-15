# 报告生成逻辑说明

## 流程概览

```
训练执行 → 数据收集 → 报告生成 → Markdown文件
```

## 核心步骤

### 1. 训练阶段 (`core/training.py`)
```python
train_one_round(config) 
  ↓ 返回
(summary, metrics)  # 包含 main_score 和详细指标
```

### 2. 构建历史记录
```python
history = [{
    "round_id": 1,
    "tuned_key": "参数名",
    "main_score": 0.9397,
    "metrics": {
        "eval_stsb_dev_spearman_cosine": 0.9397,
        "train_runtime": 17.32,
        ...
    }
}]
```

### 3. 生成报告 (`utils/report_generator.py`)
```python
generate_run_report(
    history,        # 训练历史
    best_round,     # 最佳轮次
    best_score,     # 最佳分数
    best_config,    # 最佳配置
    ...
)
```

## 报告内容结构

```markdown
# Agent 运行报告

## 最终结果摘要
- 最优轮次 / 最优分数
- 调整的参数列表
- 最终配置详情

## 详细逐轮记录
- 每轮配置和分数
- 训练时间和速度
- 评估指标

## 建议
- 优化方向建议
```

## 输出位置

```
docs/reports/agent_run_report_YYYYMMDD_HHMMSS.md
```

## 分数提取逻辑

优先级顺序：
1. `best_score` 参数
2. `history[best_round]["main_score"]`
3. `metrics["eval_stsb_dev_spearman_cosine"]`
4. `metrics["eval_spearman_cosine"]`（备选）
5. `metrics["eval_cosine"]`（备选）

## 调用示例

**快速测试：**
```python
# scripts/run_quick_model_tests.py
summary, metrics = train_one_round(config)
generate_run_report(history=[...], best_round=1, ...)
```

**多轮调参：**
```python
# run.py
for round in rounds:
    summary, metrics = train_one_round(config)
    history.append({...})
generate_run_report(history, best_round, ...)
```

---

**特性：** 中英双语 | Markdown格式 | 自动时间戳 | 多轮记录 | 容错机制
