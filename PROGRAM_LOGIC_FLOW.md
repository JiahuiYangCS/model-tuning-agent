# STSb Auto-Tune Agent v6 - 程序逻辑流程详解

本文档用 **简单的中英对照表格** 方式展示整个程序的执行流程和函数说明。可以用任何浏览器打开查看，或用 Markdown 编辑器浏览。

---

## 1. 整体程序执行流程

```
┌─────────────────────────────────────────────────────────────┐
│  用户启动: python agent_main_v6.py                          │
│  User starts: python agent_main_v6.py                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤 0] 初始化 + 调用 GPT 获取初始计划                    │
│  [Step 0] Initialize + Call GPT for initial plan            │
│  ├─ ask_gpt_for_initial_plan()                              │
│  └─ 返回: base_config + priority_keys (最多3个)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  [步骤 1] 对每个 priority_key 循环执行单变量调参            │
│  [Step 1] Loop over each priority_key for single-var tuning │
│  ├─ key = priority_keys[0], priority_keys[1], ...           │
│  └─ MAX 3个参数, 每个参数 MAX 3轮                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────────┐
         │ [内层循环] 对当前参数做N轮训练   │
         │ [Inner] Train N rounds for key  │
         │                                 │
         │ 每轮:                           │
         │ - train_one_round()             │
         │ - ask_gpt_for_new_config()      │
         │ - 用户确认是否继续 (y/n)        │
         └────────────┬────────────────────┘
                      │
    ┌─────────────────┴──────────────────┐
    │ (每轮后)                           │
    ▼ 更新该参数为内部最佳值              │
    固定该参数, 转向下一个参数     │
    │◄──────────────────────────┘
    │
    ▼ (所有参数调完后)
┌─────────────────────────────────────────────────────────────┐
│  [步骤 2] 后处理: 复制最佳模型 + 生成报告 + GPT总结          │
│  [Step 2] Post-processing: Copy best model + Report + Summ. │
│  ├─ shutil.copytree() - 复制最佳模型                        │
│  ├─ generate_run_report() - 生成中英报告                    │
│  └─ ask_gpt_for_overall_summary() - GPT整体总结             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  [完成] 程序结束, 打印最佳结果                              │
│  [Done] Program completes, print best results               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 详细函数说明表

| 函数名 | 输入参数 | 输出/返回值 | 中文说明 | English Description |
|-------|---------|-----------|--------|---------------------|
| **make_default_config()** | 无 | Dict[str, Any] | 生成一份包含所有可调超参的默认配置字典，方便初始化和后续修改 | Generate a default config dict with all tunable parameters for initialization |
| **export_config_for_agent(config)** | config: Dict | Dict (子集) | 从完整 config 中抽取 TUNABLE_KEYS 中的键，作为给 GPT 看的"安全视图"，避免泄露非调参键 | Extract only TUNABLE_KEYS from config as a safe view for GPT, preventing exposure of non-tunable keys |
| **set_global_seed(seed)** | seed: int (default=42) | 无 | 设置 Python、NumPy、Torch 的随机种子（包含 GPU），确保训练可复现 | Set random seeds for Python/NumPy/Torch (including GPU) to ensure reproducibility |
| **train_one_round(config, round_id)** | config: Dict, round_id: int | (summary, metrics) | 按当前 config 执行一次完整的训练流程：加载数据→初始化模型→训练→评估→保存，返回本轮得分和指标 | Execute one complete training round: load data → init model → train → eval → save, return score & metrics |
| **run_quora_test_if_enabled(config, model)** | config: Dict, model: SentenceTransformer | 无 | 可选：若 ENABLE_QUORA_TEST=True，在 Quora 数据集上运行额外测试，计算重复/非重复问对的平均相似度并绘图 | Optional: If enabled, run Quora duplicate test, compute avg similarity for dup/non-dup pairs and plot |
| **build_agent_input(...)** | config, summary, history, primary_key | str (JSON) | 把当前轮的配置+训练结果+历史+当前调参键打包成 JSON 字符串，作为发给 GPT 的用户输入 | Pack config+summary+history+primary_key into JSON string as user input for GPT |
| **ask_gpt_for_initial_plan(config, model)** | config: Dict, model: str | Dict | 调用 GPT（Step 0）：让 GPT 基于当前 config 返回"稳妥的"base_config 和 1~3 个最值得调的 priority_keys | Call GPT: return recommended base_config + 1~3 priority_keys based on config |
| **ask_gpt_for_new_config(config, summary, ..., primary_key)** | config, summary, history, primary_key | Dict | 调用 GPT（每轮循环中）：在传入 primary_key 时，GPT 只允许修改该键，实现单变量调参；返回中文评价+新 config | Call GPT each round: in single-var mode (primary_key given), GPT only modifies that key; returns comment+new_config |
| **ask_gpt_for_overall_summary(history, best_round, best_score, best_config, model)** | history, best_round, best_score, best_config, model | str | 在全部调参完成后调用 GPT：让 GPT 给一份整体总结（趋势、质量评估、建议），返回中文自然段文本 | Call GPT after all tuning: return overall summary (trends, quality, suggestions) as Chinese text |
| **apply_new_config(base, new)** | base_config, new_config | Dict | 把 GPT 返回的 new_config 合并到当前 config 中，但只覆盖已存在的键（安全合并，不新增键） | Merge GPT's new_config into current config safely, only overwrite existing keys |
| **generate_run_report(history, best_round, best_score, best_config, priority_keys, base_cfg)** | 各项统计数据 | str (文件路径) | 生成一份中英双语的 Markdown 报告，总结每一轮的配置和得分，并保存到 `docs/reports/agent_run_report_<timestamp>.md` | Generate bilingual Markdown report summarizing all rounds' configs & scores, save to `docs/reports/` |
| **run_agent_v6()** | 无 | 无 | 主函数：协调整个自动调参流程，包括初始化、多轮循环、后处理、报告生成 | Main orchestration function: coordinate entire auto-tuning flow: init → loop → post-process → report |

---

## 3. 调用链详解（Call Chain）

### 3.1 初始化阶段 (Initialization Phase)

```
main(__name__ == '__main__')
    │
    ├─ run_agent_v6()
    │   │
    │   ├─ make_default_config()  ┐
    │   │   └─ 返回默认配置        │ 获取初始配置
    │   │                          ├─ 用 TUNABLE_KEYS 过滤
    │   ├─ export_config_for_agent(config) ┘
    │   │   └─ 返回子集 config
    │   │
    │   └─ [准备好，进入 GPT 调用]
```

### 3.2 Step 0: GPT 初始计划 (Step 0: GPT Initial Plan)

```
run_agent_v6()
    │
    └─ ask_gpt_for_initial_plan(config)
        │
        ├─ build_agent_input(config, ...) → JSON string
        │
        ├─ client.chat.completions.create(
        │      model="gpt-3.5-turbo",
        │      system_prompt="你是调参工程师...",
        │      user_input=JSON
        │   )
        │
        └─ JSON.parse(response) → {"comment": "...", "base_config": {...}, "priority_keys": [...]}
```

### 3.3 主循环：单变量多轮调参 (Main Loop: Single-Variable Multi-Round Tuning)

```
for each priority_key in priority_keys:  # 最多 3 个参数
    │
    ├─ for inner_round in range(1, ROUNDS_PER_PARAM + 1):  # 每个参数最多 3 轮
    │   │
    │   ├─ global_round_id += 1
    │   │
    │   ├─ train_one_round(current_config, round_id)
    │   │   ├─ set_global_seed(42)
    │   │   ├─ load_dataset("sentence-transformers/stsb")
    │   │   ├─ SentenceTransformer(config["BASE_MODEL"])
    │   │   ├─ CoSENTLoss(model)
    │   │   ├─ EmbeddingSimilarityEvaluator(...)
    │   │   ├─ SentenceTransformerTrainer.train()
    │   │   ├─ trainer.save_model(output_dir)
    │   │   ├─ stsb_evaluator(model) → main_score
    │   │   └─ return (summary, metrics)
    │   │
    │   ├─ 记录历史到 history_for_agent
    │   │
    │   ├─ 更新 param_best_score / global best_score / best_config
    │   │
    │   ├─ ask_gpt_for_new_config(
    │   │      export_config_for_agent(current_config),
    │   │      training_summary,
    │   │      model="gpt-3.5-turbo",
    │   │      history=history_for_agent,
    │   │      primary_key=key  ← 关键：只调这一个参数
    │   │   )
    │   │   ├─ build_agent_input(...) → JSON with primary_key
    │   │   ├─ GPT 在 system_prompt 中被约束：只修改 primary_key
    │   │   └─ return {"comment": "...", "new_config": {...(仅包含该键)...}}
    │   │
    │   ├─ if key in new_config:
    │   │      current_config = apply_new_config(current_config, new_config)
    │   │
    │   └─ if inner_round < 3:
    │       └─ input("是否继续?(y/n)") ← 用户交互
    │
    └─ 该参数 3 轮结束后：
        ├─ 固定 current_config[key] = param_best_value
        │
        └─ if param_index < MAX_PARAMS - 1:
            └─ input("是否调下一个参数?(y/n)") ← 用户交互
```

### 3.4 后处理阶段 (Post-Processing Phase)

```
run_agent_v6()  [调参循环结束后]
    │
    ├─ if best_output_dir is not None:
    │   └─ shutil.copytree(best_output_dir, best_overall_dir)
    │       └─ 复制最佳轮次的模型权重到 best_overall_model/
    │
    ├─ generate_run_report(
    │      history_for_agent,
    │      best_round, best_score, best_config,
    │      priority_keys, base_cfg
    │   )
    │   ├─ 创建 docs/reports/ 目录
    │   ├─ 遍历 history，生成每轮记录
    │   ├─ 写入 Markdown 文件
    │   └─ return report_path
    │
    ├─ ask_gpt_for_overall_summary(
    │      history_for_agent,
    │      best_round, best_score, best_config,
    │      model="gpt-3.5-turbo"
    │   )
    │   ├─ 把所有历史打包给 GPT
    │   ├─ GPT 返回整体总结文本
    │   └─ print 到控制台
    │
    └─ print("程序完成 / Program Done")
```

---

## 4. 关键数据结构 (Key Data Structures)

### 4.1 config（配置字典）

```python
config = {
    # 模型选择
    "BASE_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "NUM_TRAIN_EPOCHS": 1,
    "LEARNING_RATE": 2e-5,
    "TRAIN_BATCH_SIZE": 8,
    # ... 等等 20+ 个可调参数
}
```

**说明 / Explanation:**
- 字典中的每个键都在 `TUNABLE_KEYS` 中
- 可通过 `export_config_for_agent()` 提取子集给 GPT
- GPT 的建议直接修改这些键的值

### 4.2 training_summary（训练结果摘要）

```python
training_summary = {
    "round_id": 1,
    "output_dir": "models/stv3_agent_demo_20251204_120000_r1",
    "device": "cuda",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "stsb_train_size": 200,
    "stsb_dev_size": 100,
    "main_score": 0.7245,  # ← 最关键的评估指标
    "metrics": { "loss": 0.45, ... }
}
```

**说明 / Explanation:**
- `main_score` 用于比较不同轮次的效果
- 根据 `main_score` 选择全局和参数内的最佳轮次

### 4.3 history_for_agent（历史记录列表）

```python
history_for_agent = [
    {
        "round_id": 1,
        "tuned_key": "LEARNING_RATE",
        "inner_round_index": 1,
        "config_for_agent": { "LEARNING_RATE": 2e-5, ... },
        "main_score": 0.7123,
        "metrics": { ... }
    },
    {
        "round_id": 2,
        "tuned_key": "LEARNING_RATE",
        "inner_round_index": 2,
        "config_for_agent": { "LEARNING_RATE": 3e-5, ... },
        "main_score": 0.7256,
        "metrics": { ... }
    },
    # ... 更多轮次
]
```

**说明 / Explanation:**
- 每一轮的记录都保存在这里
- 传给 GPT 作为历史上下文，帮助做出更好决策
- 用于最后生成运行报告

---

## 5. 关键决策点 (Decision Points)

| 决策点 | 中文 | 英文 | 影响 |
|-------|------|------|------|
| **Step 0 - GPT 返回有效结果？** | GPT 能否解析返回有效 JSON? | Can GPT parse and return valid JSON? | 若失败→使用兜底策略（TUNABLE_KEYS[:3]）；若成功→按 GPT 建议调参 |
| **单轮训练后 - 该参数是否改进？** | main_score > param_best_score? | Is this round's score better? | 更新 param_best_score/value；用于决定该参数的最终取值 |
| **全局比较 - 是否是全局最优？** | main_score > best_score? | Is this the global best so far? | 更新 best_round/best_config；用于最后复制最佳模型 |
| **用户交互 - 继续该参数下一轮？** | 用户输入 y/n | User input (y/n) | y→进入下一内轮；n→结束该参数调参，转向下一参数 |
| **用户交互 - 继续下一个参数？** | 用户输入 y/n | User input (y/n) | y→进入下一参数调参；n→跳过剩余参数，进入后处理 |
| **后处理 - best_output_dir 存在？** | 本次运行是否有最佳轮次模型？ | Does best model directory exist? | 是→复制到 best_overall_model；否→跳过复制 |

---

## 6. 文件输出位置 (Output File Locations)

```
项目根目录 / Project Root
│
├─ models/  ← 所有训练输出
│  ├─ stv3_agent_demo_20251204_120000_r1/  (第 1 轮)
│  │  ├─ pytorch_model.bin
│  │  ├─ config.json
│  │  ├─ ...
│  ├─ stv3_agent_demo_20251204_120500_r2/  (第 2 轮)
│  │  └─ ...
│  └─ best_overall_model/  ← 全局最佳模型的副本
│     └─ ...
│
├─ docs/
│  └─ reports/  ← 运行报告
│     └─ agent_run_report_20251204_120530.md  (本次运行的报告)
│
└─ agent_main_v6.py (主程序)
```

---

## 7. 简单工作流总结 (Simple Workflow Summary)

```
初始化 (Initialize)
    ↓
Step 0: 调用 GPT 获取初始计划
    ↓ (GPT 返回 priority_keys: [LEARNING_RATE, TRAIN_BATCH_SIZE, NUM_TRAIN_EPOCHS])
参数 1: LEARNING_RATE
    ├─ Round 1: 用 lr=2e-5 训练 → score=0.71 → GPT 建议改为 3e-5
    ├─ Round 2: 用 lr=3e-5 训练 → score=0.72 ✓ (更好) → GPT 建议改为 3.5e-5
    ├─ Round 3: 用 lr=3.5e-5 训练 → score=0.71 (变差) → 固定 lr=3e-5（本参数最佳）
    └─ [用户确认继续] → 转向参数 2
    ↓
参数 2: TRAIN_BATCH_SIZE (基于 LEARNING_RATE=3e-5 固定)
    ├─ Round 1: 用 bs=8 训练 → score=0.72
    ├─ Round 2: 用 bs=16 训练 → score=0.73 ✓ (更好) → 固定 bs=16
    └─ [用户确认继续] → 转向参数 3
    ↓
参数 3: NUM_TRAIN_EPOCHS (基于 LEARNING_RATE=3e-5, BATCH_SIZE=16 固定)
    ├─ Round 1: 用 epochs=1 训练 → score=0.73
    ├─ Round 2: 用 epochs=2 训练 → score=0.74 ✓ (更好)
    └─ [用户不继续] → 结束调参
    ↓
后处理 (Post-process)
    ├─ 复制最佳轮次（全局 score=0.74）的模型
    ├─ 生成运行报告到 docs/reports/
    ├─ 调用 GPT 给出整体总结
    └─ 程序完成
    ↓
用户可查看:
    1. docs/reports/agent_run_report_*.md  (中英双语报告)
    2. models/best_overall_model/  (最佳模型权重)
    3. 控制台输出  (完整日志和 GPT 总结)
```

---

## 8. 快速查询表 (Quick Reference)

| 我想了解... | 查看... |
|-----------|--------|
| 程序整体流程 | 第 1 章 流程图 |
| 某个函数的作用 | 第 2 章 函数表 |
| 函数之间如何调用 | 第 3 章 调用链 |
| 数据的存储方式 | 第 4 章 数据结构 |
| 程序如何做决策 | 第 5 章 决策点 |
| 输出文件在哪里 | 第 6 章 文件位置 |
| 一个完整例子 | 第 7 章 工作流总结 |

---

**文档更新于**: 2025-12-04  
**适用版本**: Agent v6  
**语言**: 中文 / English (Bilingual)

