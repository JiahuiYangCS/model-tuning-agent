# 项目逻辑说明 / Project Logic Explanation

## 核心概念 / Core Concept

这个项目的逻辑是：**使用LLM作为"调参顾问"，优化本地embedding模型**

```
┌─────────────────────────────────────────────────────────────┐
│                      正确的理解                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LLM (GPT/Gemini等)                本地训练                 │
│  ┌──────────────┐                 ┌──────────────┐         │
│  │              │   提供建议       │              │         │
│  │  调参顾问    │ ─────────────>  │ sentence-    │         │
│  │  Advisor     │   超参数配置     │ transformers │         │
│  │              │ <───────────    │  模型训练     │         │
│  └──────────────┘   训练结果       └──────────────┘         │
│                                                              │
│  角色：智囊/助手                   角色：被优化对象          │
│  位置：远程API                     位置：本地GPU            │
│  评测：❌ 不评测                   评测：✅ 评测这个        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 错误的做法 ❌

**之前的 `run_deep_model_tests.py` 存在逻辑错误：**

```python
# ❌ 错误的逻辑
for llm_model in [GPT, Gemini, Nemotron, ...]:
    # 用这个LLM给建议
    advice = llm_model.suggest_hyperparameters()
    
    # 在本地训练
    result = train_locally(sentence_transformer, advice)
    
    # ❌ 错误：给LLM打分
    llm_score[llm_model] = result.score  # 实际上都是0.8735！
```

**为什么所有LLM得分都一样？**

因为：
1. 所有LLM使用相同的training pipeline
2. 实际训练的是同一个本地模型（sentence-transformers/all-MiniLM-L6-v2）
3. LLM只是提供文本建议，不参与训练
4. 训练是确定性的（相同seed + 相同数据 = 相同结果）

**结果：** 所有模型得分 0.8735，完全一样！

## 正确的做法 ✅

### 方案1：单一LLM顾问调参

```python
# ✅ 正确的逻辑
# 选择一个LLM作为顾问
advisor = choose_llm("gpt-3.5-turbo")

# 多轮调参
for round in range(max_rounds):
    # 顾问给出建议
    new_config = advisor.suggest_next_config(history)
    
    # 本地测试这个配置
    result = train_locally(sentence_transformer, new_config)
    
    # ✅ 记录配置的效果
    config_scores[new_config] = result.score
    
# ✅ 评测的是"配置"，不是LLM
best_config = max(config_scores, key=config_scores.get)
```

### 方案2：如果真的要对比LLM

如果你确实想评测不同LLM作为顾问的能力，应该这样：

```python
# ✅ 对比LLM的"调参能力"
results = {}

for advisor_llm in [GPT, Gemini, Claude]:
    print(f"使用 {advisor_llm} 作为调参顾问")
    
    # 让这个LLM进行完整的多轮调参
    best_config, best_score = run_full_tuning_with_advisor(advisor_llm)
    
    # 记录这个LLM顾问找到的最佳结果
    results[advisor_llm] = {
        'best_config': best_config,
        'best_score': best_score,
        'rounds': 10
    }

# 对比：哪个LLM找到了更好的配置？
print("各LLM顾问找到的最佳配置：")
for llm, result in results.items():
    print(f"{llm}: {result['best_score']}")
```

**关键区别：**
- 不是测试"训练1次的结果"（那肯定一样）
- 而是测试"经过N轮优化后，哪个LLM找到了更好的配置"

## 文件用途说明

### 推荐使用的脚本

| 脚本 | 用途 | 是否合理 |
|------|------|----------|
| `run.py` | 使用选定的LLM进行多轮调参 | ✅ 合理 |
| `run_single_agent_test.py` | 单一LLM顾问调参（新增） | ✅ 合理 |

### 不推荐/需要修改的脚本

| 脚本 | 问题 | 建议 |
|------|------|------|
| `run_deep_model_tests.py` | 循环多个LLM但结果都一样 | ❌ 删除或重构 |
| `run_quick_model_tests.py` | 同上 | ❌ 删除或重构 |
| `generate_deep_comparison_report.py` | 对比LLM模型（不合逻辑） | ❌ 修改逻辑 |

## 正确的工作流程

### 快速验证流程

```bash
# 1. 选择一个LLM顾问
python run_single_agent_test.py

# 2. 查看调参报告
cat docs/reports/agent_run_report_*.md
```

### 对比不同顾问能力（可选）

```bash
# 使用GPT-3.5作为顾问
python run.py  # 选择GPT-3.5

# 使用Gemini作为顾问
python run.py  # 选择Gemini

# 手动对比两次运行的best_score
```

## 报告应该展示什么

### ✅ 应该展示的内容

```markdown
# Agent调参报告

## 使用的LLM顾问
- 模型：gpt-3.5-turbo
- 作用：提供超参数优化建议

## 调参过程
| 轮次 | 调整参数 | 分数 | 说明 |
|------|----------|------|------|
| 1 | LEARNING_RATE=2e-5 | 0.8500 | 基线 |
| 2 | LEARNING_RATE=3e-5 | 0.8650 | 提升 |
| 3 | LEARNING_RATE=4e-5 | 0.8450 | 过高 |
| 4 | LEARNING_RATE=3e-5 | 0.8735 | 最优 |

## 最佳配置
- LEARNING_RATE: 3e-5
- BATCH_SIZE: 16
- EPOCHS: 3
```

### ❌ 不应该展示的内容

```markdown
# 错误的报告

## LLM模型对比 ❌
| LLM | 分数 | 说明 |
|-----|------|------|
| GPT-3.5 | 0.8735 | ❌ 没意义！|
| Gemini | 0.8735 | ❌ 都一样！|
| Claude | 0.8735 | ❌ 逻辑错误！|
```

## 总结

**一句话说明：**
> LLM是"军师"，sentence-transformers是"将军"。  
> 我们要评测的是将军的战斗力（模型效果），而不是军师说话好不好听。

**记住：**
- ✅ 评测：超参数配置的效果
- ✅ 评测：本地训练模型的性能
- ❌ 不评测：LLM模型本身（它们只是顾问）
- ❌ 不对比：不同LLM在单次运行中的结果（肯定一样）
