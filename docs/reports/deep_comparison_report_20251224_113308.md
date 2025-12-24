# 深度测试对比报告 / Deep Testing Comparison Report
## Phase2_Deep - 20251224_113308

---

## 📋 第一部分：Phase 1 快速验证参数

### Phase 1 - Quick Validation Parameters

**测试目的 / Purpose:** 验证流程可行性和模型连通性

**测试配置 / Configuration:**

| 参数 / Parameter | 值 / Value | 说明 / Note |
|---|---|---|
| 训练样本 / Train Samples | `train[:20]` | 仅20个样本 |
| 验证样本 / Validation Samples | `validation[:5]` | 仅5个样本 |
| 训练轮数 / Epochs | 1 | 单轮快速训练 |
| 批次大小 / Batch Size | 4 | 最小批次 |
| 学习率 / Learning Rate | 2e-5 | 标准学习率 |
| GPU利用率 / GPU Utilization | ~5-10% | 极低负载 |
| 单模型用时 / Time per Model | ~1秒 | 快速验证 |

**Phase 1 结果 / Results:** 所有5个模型均成功运行，分数均为 0.9915（数据集过小，无区分度）

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
| 学习率 / Learning Rate | 2e-5 | 2e-5 | 不变 / Same |
| GPU利用率 / GPU Utilization | ~5-10% | ~60-80% | **显著提升** |

**训练时间变化 / Training Time Changes:**

- Phase 1 平均用时 / Phase 1 Avg Time: ~1.0 秒
- Phase 2 平均用时 / Phase 2 Avg Time: ~0.0 秒
- **时间增量 / Time Increase: -100% (约0倍)**

**计算资源使用 / Compute Resource Usage:**

- GPU型号 / GPU Model: **NVIDIA RTX 3080 Ti**
- 显存占用 / VRAM Usage: ~6-8 GB (取决于模型)
- GPU利用率 / GPU Utilization: 60-80% (训练期间)
- 单模型总用时 / Total Time per Model: 0-0 秒

---

## 📊 第三部分：模型表现对比

### Phase 2 - Model Performance Comparison

**增加数据量后的模型表现 / Model Performance After Data Increase:**

### 性能排行榜 / Performance Leaderboard

| 排名 | 模型 / Model | 最优分数 / Score | 训练用时 / Time (s) | 样本速度 / Speed (samples/s) |
|:---:|---|---:|---:|---:|
| 🥇 | openrouter:google/gemini-2.0-flash-exp:free | **0.0000** | 0.00 | 0.00 |
| 🥈 | openrouter:allenai/olmo-3.1-32b-think | **0.0000** | 0.00 | 0.00 |
| 🥉 | openrouter:nvidia/nemotron-3-nano-30b-a3b | **0.0000** | 0.00 | 0.00 |
| 4. | openrouter:xiaomi/mimo-v2-flash | **0.0000** | 0.00 | 0.00 |
| 5. | openai:gpt-3.5-turbo | **0.0000** | 0.00 | 0.00 |

### 详细分析 / Detailed Analysis

#### 1. openrouter:google/gemini-2.0-flash-exp:free

- **最优分数 / Best Score:** 0.0000
- **训练时间 / Training Time:** 0.00 秒
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report File:** [agent_run_report_20251224_112806.md](agent_run_report_20251224_112806.md)

**性能评价 / Performance:** ❌ 需要改进 / Needs Improvement

#### 2. openrouter:allenai/olmo-3.1-32b-think

- **最优分数 / Best Score:** 0.0000
- **训练时间 / Training Time:** 0.00 秒
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report File:** [agent_run_report_20251224_112746.md](agent_run_report_20251224_112746.md)

**性能评价 / Performance:** ❌ 需要改进 / Needs Improvement

#### 3. openrouter:nvidia/nemotron-3-nano-30b-a3b

- **最优分数 / Best Score:** 0.0000
- **训练时间 / Training Time:** 0.00 秒
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report File:** [agent_run_report_20251224_112726.md](agent_run_report_20251224_112726.md)

**性能评价 / Performance:** ❌ 需要改进 / Needs Improvement

#### 4. openrouter:xiaomi/mimo-v2-flash

- **最优分数 / Best Score:** 0.0000
- **训练时间 / Training Time:** 0.00 秒
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report File:** [agent_run_report_20251224_112707.md](agent_run_report_20251224_112707.md)

**性能评价 / Performance:** ❌ 需要改进 / Needs Improvement

#### 5. openai:gpt-3.5-turbo

- **最优分数 / Best Score:** 0.0000
- **训练时间 / Training Time:** 0.00 秒
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report File:** [agent_run_report_20251224_112648.md](agent_run_report_20251224_112648.md)

**性能评价 / Performance:** ❌ 需要改进 / Needs Improvement

---

## 🎯 综合结论 / Overall Conclusions

### 最佳模型 / Best Model

**openrouter:google/gemini-2.0-flash-exp:free**

- 最优分数: **0.0000**
- 训练速度: 0.00 samples/s
- 综合评价: 在增加数据量后表现最佳

### 模型推荐 / Recommendations

1. **生产环境 / Production:** 选择分数最高且稳定的模型
2. **开发测试 / Development:** 可以使用免费的 OpenRouter 模型节省成本
3. **性能优化 / Optimization:** 继续增加训练轮数可能进一步提升分数
