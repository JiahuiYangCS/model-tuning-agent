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


# =============== 配置 ===============

def make_default_config() -> Dict[str, Any]:
    """返回一份可调参配置的默认值（缩小版，方便测试）。"""
    return {
        # 开关
        "ENABLE_TRIPLET_EVAL": False,
        "ENABLE_QUORA_TEST":   False,  # 默认先关掉 Quora 测试减轻负担

        # 模型 & 输出路径
        "BASE_MODEL":        "sentence-transformers/all-MiniLM-L6-v2",
        "OUTPUT_DIR_ROOT":   "models",
        "RUN_NAME_PREFIX":   "stv3_agent_demo_",
        "RUN_NAME":          "agent_autotune_demo",

        # 数据子集（进一步调小）
        "STSB_TRAIN_SPLIT": "train[:200]",
        "STSB_DEV_SPLIT":   "validation[:100]",

        # 训练超参（缩小版）
        "NUM_TRAIN_EPOCHS":   1,
        "TRAIN_BATCH_SIZE":   8,
        "EVAL_BATCH_SIZE":    8,
        "GRAD_ACC_STEPS":     1,
        "LEARNING_RATE":      2e-5,
        "WARMUP_RATIO":       0.1,

        "EVAL_STRATEGY":      "steps",
        "EVAL_STEPS":         50,
        "LOGGING_STEPS":      10,
        "SAVE_STRATEGY":      "epoch",
        "SAVE_STEPS":         50,
        "SAVE_TOTAL_LIMIT":   1,
        "LOGGING_FIRST_STEP": True,

        # Quora 相关（默认用不到）
        "QUORA_SPLIT":       "train[:100]",
        "QUORA_MAX_PRINT":   2,
        "QUORA_FIGSIZE":     [6, 4],
        "QUORA_HIST_BINS":   10,
    }


# 允许 GPT 修改的键
TUNABLE_KEYS = [
    "BASE_MODEL",
    "ENABLE_TRIPLET_EVAL",
    "ENABLE_QUORA_TEST",

    "STSB_TRAIN_SPLIT",
    "STSB_DEV_SPLIT",

    "NUM_TRAIN_EPOCHS",
    "TRAIN_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "GRAD_ACC_STEPS",
    "LEARNING_RATE",
    "WARMUP_RATIO",

    "EVAL_STRATEGY",
    "EVAL_STEPS",
    "LOGGING_STEPS",
    "SAVE_STRATEGY",
    "SAVE_STEPS",
    "SAVE_TOTAL_LIMIT",
    "LOGGING_FIRST_STEP",

    "QUORA_SPLIT",
    "QUORA_MAX_PRINT",
    "QUORA_FIGSIZE",
    "QUORA_HIST_BINS",
]


def export_config_for_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    """从当前 config 中抽取一份给 GPT 看的视图。"""
    cfg = {}
    for k in TUNABLE_KEYS:
        if k in config:
            cfg[k] = config[k]
    return cfg


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_round(config: Dict[str, Any], round_id: int = 1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """按当前 config 训练一轮，返回 (training_summary, metrics)。"""

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config["OUTPUT_DIR_ROOT"],
        f"{config['RUN_NAME_PREFIX']}{run_ts}_r{round_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n====== 第 {round_id} 轮训练开始 ======")
    print("设备:", device)
    print("输出目录:", output_dir)

    # 数据
    stsb_train = load_dataset("sentence-transformers/stsb", split=config["STSB_TRAIN_SPLIT"])
    stsb_dev   = load_dataset("sentence-transformers/stsb", split=config["STSB_DEV_SPLIT"])
    print("STSb train size:", len(stsb_train))
    print("STSb dev size:", len(stsb_dev))

    # 模型 & loss & evaluator
    model = SentenceTransformer(config["BASE_MODEL"], device=device)
    loss = CoSENTLoss(model)

    stsb_evaluator = EmbeddingSimilarityEvaluator(
        sentences1 = stsb_dev["sentence1"],
        sentences2 = stsb_dev["sentence2"],
        scores     = stsb_dev["score"],
        main_similarity = SimilarityFunction.COSINE,
    )

    # 训练参数
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

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=stsb_train,
        eval_dataset=stsb_dev,
        loss=loss,
        evaluator=stsb_evaluator,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)

    # 主评估分数（兼容 float/dict）
    main_score_raw = stsb_evaluator(model)
    if isinstance(main_score_raw, dict):
        if "cosine" in main_score_raw:
            main_score = float(main_score_raw["cosine"])
        else:
            main_score = float(list(main_score_raw.values())[0])
    else:
        main_score = float(main_score_raw)

    print(f"本轮主评估分数（STSb evaluator 返回值）: {main_score:.4f}")

    summary = {
        "round_id": round_id,
        "output_dir": output_dir,
        "device": device,
        "base_model": config["BASE_MODEL"],
        "stsb_train_size": len(stsb_train),
        "stsb_dev_size": len(stsb_dev),
        "main_score": main_score,   # 存 float，方便后续使用
        "metrics": {},
    }

    if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
        summary["metrics"] = train_result.metrics
    elif isinstance(train_result, dict):
        summary["metrics"] = train_result
    else:
        summary["metrics"] = {}

    return summary, summary["metrics"]


from sentence_transformers import util
import pandas as pd
import matplotlib.pyplot as plt


def run_quora_test_if_enabled(config: Dict[str, Any], model: SentenceTransformer):
    """仅在 ENABLE_QUORA_TEST=True 时运行。"""
    if not config.get("ENABLE_QUORA_TEST", False):
        print("跳过 Quora 测试（ENABLE_QUORA_TEST=False）")
        return

    print(f"加载 Quora Duplicate Questions 数据集（{config['QUORA_SPLIT']}）...")
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
    print(f"\n平均相似度（重复问对）：{avg_dup:.3f}")
    print(f"平均相似度（非重复问对）：{avg_non:.3f}")

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
