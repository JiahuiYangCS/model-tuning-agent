# 深度测试对比报告 / Deep Testing Comparison Report
## Phase2_Deep - 20251223_113500

---

## 📋 第一部分：Phase 1 快速验证参�?

### Phase 1 - Quick Validation Parameters

**测试目的 / Purpose:** 验证流程可行性和模型连通�?

**测试配置 / Configuration:**

| 参数 / Parameter | �?/ Value | 说明 / Note |
|---|---|---|
| 训练样本 / Train Samples | `train[:20]` | �?0个样�?|
| 验证样本 / Validation Samples | `validation[:5]` | �?个样�?|
| 训练轮数 / Epochs | 1 | 单轮快速训�?|
| 批次大小 / Batch Size | 4 | 最小批�?|
| 学习�?/ Learning Rate | 2e-5 | 标准学习�?|
| GPU利用�?/ GPU Utilization | ~5-10% | 极低负载 |
| 单模型用�?/ Time per Model | ~1�?| 快速验�?|

**Phase 1 结果 / Results:** 所�?个模型均成功运行，分数均�?0.9915（数据集过小，无区分度）

---

## 🔧 第二部分：Phase 2 深度测试参数改动

### Phase 2 - Deep Testing Parameter Changes

**测试目的 / Purpose:** 增加数据量和训练周期，测试GPU负载下的真实性能

**参数改动对比 / Parameter Changes:**

| 参数 / Parameter | Phase 1 | Phase 2 | 增量 / Increase |
|---|---:|---:|---|
| 训练样本 / Train Samples | 20 | 1500 | **×75** |
| 验证样本 / Validation Samples | 5 | 300 | **×60** |
| 训练轮数 / Epochs | 1 | 3 | **×3** |
| 批次大小 / Batch Size | 4 | 16 | **×4** |
| 学习�?/ Learning Rate | 2e-5 | 2e-5 | 不变 / Same |
| GPU利用�?/ GPU Utilization | ~5-10% | ~60-80% | **显著提升** |

**训练时间变化 / Training Time Changes:**

- Phase 1 平均用时 / Phase 1 Avg Time: ~1.0 �?
- Phase 2 平均用时 / Phase 2 Avg Time: ~17.0 �?
- **时间增量 / Time Increase: 1600% (�?7�?**

**计算资源使用 / Compute Resource Usage:**

- GPU型号 / GPU Model: **NVIDIA RTX 3080 Ti**
- 显存占用 / VRAM Usage: ~6-8 GB (训练期间)
- GPU利用�?/ GPU Utilization: 60-80% (训练期间)
- 单模型总用�?/ Total Time per Model: 17-20 �?
- 总测试时�?/ Total Test Time: ~1.63 分钟 (5个模�?

---

## 📊 第三部分：模型表现对�?

### Phase 2 - Model Performance Comparison

**增加数据量后的模型表�?/ Model Performance After Data Increase:**

### 性能排行�?/ Performance Leaderboard

**基于终端输出的实际分�?(eval_spearman_cosine):**

| 排名 | 模型 / Model | 最优分�?/ Score | 训练用时 / Time (s) | 样本速度 / Speed (samples/s) |
|:---:|---|---:|---:|---:|
| 🥇 | openrouter:allenai/olmo-3.1-32b-think | **0.9397** | 17.32 | 259.8 |
| 🥈 | openrouter:google/gemini-2.0-flash-exp:free | **0.9397** | 17.19 | 261.7 |
| 🥉 | openrouter:xiaomi/mimo-v2-flash | **0.9397** | 16.84 | 266.9 |
| 4. | openrouter:nvidia/nemotron-3-nano-30b-a3b | **0.9397** | 16.75 | 268.7 |
| 5. | openai:gpt-3.5-turbo | **0.9397** | 17.24 | 261.0 |

**注意 / Note:** 所有模型在Phase 2中的表现完全一致，分数均为 **0.9397**。这是因为：
1. 所有模型使用相同的BASE_MODEL (sentence-transformers/all-MiniLM-L6-v2) 进行embedding训练
2. LLM (GPT/OpenRouter模型) 仅用于Agent分析，不影响实际训练过程
3. 在相同的训练配置和数据集下，embedding模型的微调结果是确定性的

### 详细分析 / Detailed Analysis

#### 关键发现 / Key Findings:

1. **性能一致�?/ Performance Consistency**
   - 所�?个模型的最终Spearman相关系数均为 **0.9397**
   - 这证明了不同LLM作为Agent时，对embedding训练本身没有影响
   
2. **训练速度对比 / Training Speed Comparison**
   - 最快：openrouter:nvidia/nemotron-3-nano-30b-a3b (268.7 samples/s)
   - 最慢：openrouter:allenai/olmo-3.1-32b-think (259.8 samples/s)
   - 速度差异�?%，可忽略不计

3. **GPU利用�?/ GPU Utilization**
   - Phase 2成功提升GPU利用率至60-80%
   - 3080 Ti显卡在batch_size=16, epochs=3配置下表现良�?
   - 训练时间从~1秒增加到~17秒，增长17�?

4. **数据量影�?/ Data Volume Impact**
   - Phase 1 (20 samples): 分数 0.9915 (过拟�?
   - Phase 2 (1500 samples): 分数 0.9397 (更真�?
   - 增加数据量后分数略降，但更具代表�?

---

## 🎯 综合结论 / Overall Conclusions

### 最佳模�?/ Best Model

**所�?个模型在Phase 2中表现完全相�?*
- 最优分�? **0.9397**
- 这验证了项目架构的正确性：LLM仅作为分析Agent，不参与实际训练

### Phase 1 vs Phase 2 对比 / Comparison

| 指标 / Metric | Phase 1 | Phase 2 | 变化 / Change |
|---|---|---|---|
| 数据规模 | 微小 (20+5) | 中等 (1500+300) | **75�?* |
| 训练时间 | ~1�?| ~17�?| **17�?* |
| GPU利用�?| 5-10% | 60-80% | **显著提升** |
| 分数可信�?| �?(过拟�? | �?(泛化�? | **提升** |

### 模型推荐 / Recommendations

**对于本项目的特殊架构�?*

1. **任何免费OpenRouter模型均可 / Any Free OpenRouter Model Works**
   - xiaomi/mimo-v2-flash: 262k上下文，适合长文本分�?
   - nvidia/nemotron-3-nano-30b-a3b: 速度最�?
   - allenai/olmo-3.1-32b-think: 学术透明度高
   - google/gemini-2.0-flash-exp:free: 多模态能�?

2. **OpenAI GPT-3.5-turbo**
   - 稳定性最高，但需付费
   - 性能与免费模型无差异

3. **下一步优�?/ Next Steps**
   - 可继续增加到完整数据�?(5749 training samples)
   - 可增加训练轮数到 5-10 epochs
   - 当前配置已能充分利用3080 Ti性能

### 技术验�?/ Technical Validation

�?**已验证事�?/ Validated:**
- OpenRouter API集成成功
- 5个LLM模型均可正常调用
- 训练流程在增加数据量后仍稳定运行
- GPU (3080 Ti) 负载提升至合理水�?
- 报告生成功能完整

�?**关键理解 / Key Understanding:**
- BASE_MODEL (sentence-transformers) 负责实际的embedding学习
- LLM (GPT/OpenRouter) 仅作为Agent分析训练结果和建议超�?
- 因此不同LLM不会影响最终embedding模型的性能
- 选择LLM时可以优先考虑成本(免费 vs 付费)而非性能

---

## 📌 报告元数�?/ Report Metadata

- **生成时间 / Generated:** 2025-12-23 12:40:15
- **测试模型数量 / Models Tested:** 5
- **总测试时�?/ Total Time:** 1.63 分钟
- **GPU设备 / GPU Device:** NVIDIA RTX 3080 Ti
- **数据�?/ Dataset:** STSb (Semantic Textual Similarity Benchmark)
- **BASE_MODEL:** sentence-transformers/all-MiniLM-L6-v2
- **报告类型 / Report Type:** Deep Testing Comparison (Phase 2)

---

**结论 / Conclusion:** 
本次深度测试成功验证了项目在增加数据量后的性能表现。所�?个LLM模型作为Agent均能有效工作，最终embedding模型达到 **0.9397** 的Spearman相关系数，证明了训练流程的稳定性和有效性。建议在实际生产环境中选择免费的OpenRouter模型以节省成本，同时保持相同的训练效果�?
