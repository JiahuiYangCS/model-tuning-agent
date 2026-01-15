"""
Training and configuration module / 训练和配置模块

核心训练函数和默认配置
"""

import os
import random
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator, SimilarityFunction
)
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments, BatchSamplers,
)
from sentence_transformers.trainer import SentenceTransformerTrainer

# Import from centralized config
from config import DEFAULT_CONFIG, TUNABLE_KEYS as GLOBAL_TUNABLE_KEYS


# 导出 TUNABLE_KEYS 以保持向后兼容性
TUNABLE_KEYS = GLOBAL_TUNABLE_KEYS


def get_gpu_info() -> str:
    """
    自动检测GPU信息，如果不可用返回"CPU"
    
    返回 / Returns:
        GPU 名称或 "CPU"
    """
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA GPU (Unknown Model)"
    else:
        return "CPU"


def make_default_config() -> Dict[str, Any]:
    """从 config.py 返回默认配置"""
    return DEFAULT_CONFIG.copy()


def export_config_for_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """从当前 config 中抽取一份给 GPT 看的视图（只包含 TUNABLE_KEYS 中的键）"""
    cfg = {}
    for k in TUNABLE_KEYS:
        if k in config:
            cfg[k] = config[k]
    return cfg


def set_global_seed(seed: int = 42):
    """设置全局随机种子（Python、NumPy、PyTorch、CUDA）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_round(config: Dict[str, Any], round_id: int = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    按当前 config 训练一轮，返回 (training_summary, metrics)
    
    参数 / Args:
        config: 包含所有超参的字典
        round_id: 轮次编号
    
    返回 / Returns:
        (summary, metrics): 训练摘要和评估指标
    """

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config["OUTPUT_DIR_ROOT"],
        f"{config['RUN_NAME_PREFIX']}{run_ts}_r{round_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n====== 第 {round_id} 轮训练开始 ======")
    print("设备 / Device:", device)
    print("输出目录 / Output dir:", output_dir)

    # 加载数据集 / Load dataset - 支持多种数据集
    dataset_name = config.get("DATASET_NAME", "stsb")
    
    if dataset_name == "stsb":
        # STSb 数据集
        train_data = load_dataset("sentence-transformers/stsb", split=config["STSB_TRAIN_SPLIT"])
        dev_data = load_dataset("sentence-transformers/stsb", split=config["STSB_DEV_SPLIT"])
        print(f"数据集 / Dataset: STSb")
        print(f"训练样本 / Train size: {len(train_data)}")
        print(f"验证样本 / Dev size: {len(dev_data)}")
        
        # 初始化评估器
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1 = dev_data["sentence1"],
            sentences2 = dev_data["sentence2"],
            scores     = dev_data["score"],
            main_similarity = SimilarityFunction.COSINE,
        )
        
    elif dataset_name == "allnli":
        # AllNLI 数据集
        train_data = load_dataset("sentence-transformers/all-nli", split=config.get("ALLNLI_TRAIN_SPLIT", "train[:50000]"))
        dev_data = load_dataset("sentence-transformers/all-nli", split=config.get("ALLNLI_DEV_SPLIT", "validation"))
        print(f"数据集 / Dataset: AllNLI (SNLI + MultiNLI)")
        print(f"训练样本 / Train size: {len(train_data)}")
        print(f"验证样本 / Dev size: {len(dev_data)}")
        
        # AllNLI 使用不同的评估器（NLI任务）
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1 = dev_data["sentence1"],
            sentences2 = dev_data["sentence2"],
            scores     = [1.0 if label == 0 else 0.0 for label in dev_data["label"]],  # entailment=1, 其他=0
            main_similarity = SimilarityFunction.COSINE,
        )
        
    else:
        # 默认回退到STSb
        print(f"⚠️ 未知数据集: {dataset_name}，使用默认STSb")
        train_data = load_dataset("sentence-transformers/stsb", split=config["STSB_TRAIN_SPLIT"])
        dev_data = load_dataset("sentence-transformers/stsb", split=config["STSB_DEV_SPLIT"])
        print(f"训练样本 / Train size: {len(train_data)}")
        print(f"验证样本 / Dev size: {len(dev_data)}")
        
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1 = dev_data["sentence1"],
            sentences2 = dev_data["sentence2"],
            scores     = dev_data["score"],
            main_similarity = SimilarityFunction.COSINE,
        )

    # 初始化模型、损失函数 / Initialize model, loss
    model = SentenceTransformer(config["BASE_MODEL"], device=device)
    loss = CoSENTLoss(model)

    # 训练参数 / Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["NUM_TRAIN_EPOCHS"],
        per_device_train_batch_size=config["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=config["EVAL_BATCH_SIZE"],
        gradient_accumulation_steps=config["GRAD_ACC_STEPS"],
        learning_rate=config["LEARNING_RATE"],
        warmup_ratio=config["WARMUP_RATIO"],
        fp16=torch.cuda.is_available(),
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,

        eval_strategy=config["EVAL_STRATEGY"],
        eval_steps=config["EVAL_STEPS"],
        logging_steps=config["LOGGING_STEPS"],
        save_strategy=config["SAVE_STRATEGY"],
        save_steps=config["SAVE_STEPS"],
        save_total_limit=config["SAVE_TOTAL_LIMIT"],
        logging_first_step=config["LOGGING_FIRST_STEP"],

        report_to=["tensorboard"],
        run_name=config["RUN_NAME"],
    )

    print(args)

    # 创建训练器并训练 / Create trainer and train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        loss=loss,
        evaluator=evaluator,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)

    # 获取主评估分数 / Get main evaluation score
    main_score_raw = evaluator(model)
    if isinstance(main_score_raw, dict):
        if "cosine" in main_score_raw:
            main_score = float(main_score_raw["cosine"])
        else:
            main_score = float(list(main_score_raw.values())[0])
    else:
        main_score = float(main_score_raw)

    print(f"本轮主评估分数（STSb evaluator 返回值）/ Main score: {main_score:.4f}")

    summary = {
        "round_id": round_id,
        "output_dir": output_dir,
        "device": device,
        "base_model": config["BASE_MODEL"],
        "dataset": dataset_name,
        "train_size": len(train_data),
        "dev_size": len(dev_data),
        "main_score": main_score,
        "metrics": {},
    }

    if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
        summary["metrics"] = train_result.metrics
    elif isinstance(train_result, dict):
        summary["metrics"] = train_result
    else:
        summary["metrics"] = {}
    
    # 确保metrics中包含主评估分数（用于报告生成器）
    # Ensure metrics contains the main evaluation score (for report generator)
    if "eval_stsb_dev_spearman_cosine" not in summary["metrics"]:
        summary["metrics"]["eval_stsb_dev_spearman_cosine"] = main_score

    return summary, summary["metrics"]


from sentence_transformers import util
import pandas as pd
import matplotlib.pyplot as plt


def run_quora_test_if_enabled(config: Dict[str, Any], model: SentenceTransformer):
    """
    仅在 ENABLE_QUORA_TEST=True 时运行 Quora 重复问题检测测试
    Run Quora duplicate question test only if ENABLE_QUORA_TEST=True
    """
    if not config.get("ENABLE_QUORA_TEST", False):
        print("跳过 Quora 测试 / Skip Quora test（ENABLE_QUORA_TEST=False）")
        return

    print(f"加载 Quora Duplicate Questions 数据集 / Load Quora dataset（{config['QUORA_SPLIT']}）...")
    quora = load_dataset("quora", split=config["QUORA_SPLIT"], trust_remote_code=True)

    for i, row in enumerate(quora):
        if i >= config["QUORA_MAX_PRINT"]:
            break
        print(f"{i+1}. {row['questions']['text'][0]} <-> {row['questions']['text'][1]} | label={row['is_duplicate']}")

    pairs = []
    for row in quora:
        q1, q2 = row["questions"]["text"]
        label = row["is_duplicate"]
        emb1 = model.encode(q1, convert_to_tensor=True, normalize_embeddings=True)
        emb2 = model.encode(q2, convert_to_tensor=True, normalize_embeddings=True)
        cos_sim = util.cos_sim(emb1, emb2).item()
        pairs.append({"Q1": q1, "Q2": q2, "Label": label, "CosineSim": cos_sim})

    df = pd.DataFrame(pairs)
    avg_dup = df[df["Label"] == 1]["CosineSim"].mean()
    avg_non = df[df["Label"] == 0]["CosineSim"].mean()
    print(f"\n平均相似度（重复问对）/ Avg similarity (duplicate): {avg_dup:.3f}")
    print(f"平均相似度（非重复问对）/ Avg similarity (non-dup): {avg_non:.3f}")

    figsize = config["QUORA_FIGSIZE"]
    if isinstance(figsize, list):
        figsize = tuple(figsize)

    plt.figure(figsize=figsize)
    plt.hist(df[df["Label"] == 1]["CosineSim"], bins=config["QUORA_HIST_BINS"], alpha=0.7, label="Duplicate (1)")
    plt.hist(df[df["Label"] == 0]["CosineSim"], bins=config["QUORA_HIST_BINS"], alpha=0.7, label="Non-duplicate (0)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Sentence Transformer 在 Quora 上的相似度分布")
    plt.show()
