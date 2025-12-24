# 五个模型对比报告 / Five-Model Comparison Report (20251217_170025)

## 概述 / Overview

本报告对比了五个不同�?LLM 模型在相同训练配置下的表现�?
所有模型均使用相同�?BASE_MODEL（sentence-transformers/all-MiniLM-L6-v2）进�?embedding 微调�?
区别在于使用不同�?LLM（OpenAI GPT �?OpenRouter 模型）作�?Agent 来分析训练结果并给出建议�?

**注意**：由于本次测试使用极小数据集（train[:20], validation[:5]）和单轮训练�?
主要目的是验证流程可行性而非评估模型真实性能�?

## 模型对比�?/ Model Comparison Table

| 模型 / Model | 最优分�?/ Best Score | 训练用时(�? / Train Time (s) | 样本速度 / Samples/s |
|---|---:|---:|---:|
| openrouter:google/gemini-2.0-flash-exp:free | 0.9915 | 0.00 | 0.00 |
| openrouter:allenai/olmo-3.1-32b-think | 0.9915 | 0.00 | 0.00 |
| openrouter:nvidia/nemotron-3-nano-30b-a3b | 0.9915 | 0.00 | 0.00 |
| openrouter:xiaomi/mimo-v2-flash | 0.9915 | 0.00 | 0.00 |
| openai:gpt-3.5-turbo | 0.9915 | 0.00 | 0.00 |


## 各模型详细分�?/ Detailed Analysis by Model

### 1. openrouter:google/gemini-2.0-flash-exp:free

- **最优分�?/ Best Score:** 0.9915
- **训练用时 / Training Time:** 0.00 �?
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report:** agent_run_report_20251224_110949.md

**特点 / Characteristics:**
- Google Gemini 模型
- 多模态支�?
- OpenRouter 免费访问（实验版本）

### 2. openrouter:allenai/olmo-3.1-32b-think

- **最优分�?/ Best Score:** 0.9915
- **训练用时 / Training Time:** 0.00 �?
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report:** agent_run_report_20251224_110947.md

**特点 / Characteristics:**
- Allen AI 开源模�?
- 学术研究背景
- OpenRouter 免费访问

### 3. openrouter:nvidia/nemotron-3-nano-30b-a3b

- **最优分�?/ Best Score:** 0.9915
- **训练用时 / Training Time:** 0.00 �?
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report:** agent_run_report_20251224_110944.md

**特点 / Characteristics:**
- NVIDIA 开源模�?
- 针对推理优化
- OpenRouter 免费访问

### 4. openrouter:xiaomi/mimo-v2-flash

- **最优分�?/ Best Score:** 0.9915
- **训练用时 / Training Time:** 0.00 �?
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report:** agent_run_report_20251224_110942.md

**特点 / Characteristics:**
- 小米开源模�?
- 支持大上下文�?62k tokens�?
- OpenRouter 免费访问

### 5. openai:gpt-3.5-turbo

- **最优分�?/ Best Score:** 0.9915
- **训练用时 / Training Time:** 0.00 �?
- **训练速度 / Training Speed:** 0.00 samples/s
- **报告文件 / Report:** agent_run_report_20251224_110939.md

**特点 / Characteristics:**
- 使用 OpenAI 官方 GPT 模型
- 商业服务，稳定性高
- 需要付�?API key

## 综合对比与建�?/ Overall Comparison and Recommendations

### 性能排名 / Performance Ranking

**按最优分数排�?/ By Best Score:**
1. openrouter:google/gemini-2.0-flash-exp:free: 0.9915
2. openrouter:allenai/olmo-3.1-32b-think: 0.9915
3. openrouter:nvidia/nemotron-3-nano-30b-a3b: 0.9915
4. openrouter:xiaomi/mimo-v2-flash: 0.9915
5. openai:gpt-3.5-turbo: 0.9915

**按训练速度排序 / By Training Speed:**
1. openrouter:google/gemini-2.0-flash-exp:free: 0.00 samples/s
2. openrouter:allenai/olmo-3.1-32b-think: 0.00 samples/s
3. openrouter:nvidia/nemotron-3-nano-30b-a3b: 0.00 samples/s
4. openrouter:xiaomi/mimo-v2-flash: 0.00 samples/s
5. openai:gpt-3.5-turbo: 0.00 samples/s


### 结论 / Conclusions

1. **准确度方�?*：所有模型在本次极小数据集测试中达到了相似的分数（~0.99），
   这是因为测试规模太小（仅20个训练样本）无法体现模型差异�?
   建议使用更大数据集（如完�?STSb train split）进行更有意义的对比�?

2. **速度方面**：训练速度的差异主要来自硬件和批处理效率，
   与使用哪�?LLM �?Agent 无直接关系（Agent 仅在训练后分析结果）�?

3. **成本方面**�?
   - OpenAI GPT: 需要付�?API，但服务稳定
   - OpenRouter 免费模型: 无需付费，适合实验和学�?
   - 建议: 开发测试阶段使用免费模型，生产环境根据需求选择

4. **推荐选择 / Recommendations**:
   - **学习/实验**: 使用 OpenRouter 免费模型（如 xiaomi/mimo-v2-flash�?
   - **生产/商业**: 使用 OpenAI GPT-3.5/GPT-4（稳定性和支持更好�?
   - **研究**: 可尝试不同开源模型（AllenAI OLMo, NVIDIA Nemotron）了解差�?

### 下一步建�?/ Next Steps

1. 使用完整数据集重新测试（修改 STSB_TRAIN_SPLIT �?'train' 而非 'train[:20]'�?
2. 增加训练轮数（NUM_TRAIN_EPOCHS > 1）观察模型收敛情�?
3. 对比不同 BASE_MODEL（如 all-mpnet-base-v2）的效果
4. 在真实业务场景中测试 Agent 建议的质量和准确�?

---

**生成时间 / Generated:** 2025-12-24 11:10:27

**原始报告 / Source Reports:**

- agent_run_report_20251224_110949.md

- agent_run_report_20251224_110947.md

- agent_run_report_20251224_110944.md

- agent_run_report_20251224_110942.md

- agent_run_report_20251224_110939.md
