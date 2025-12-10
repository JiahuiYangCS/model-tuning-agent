"""
🎯 Agent 项目配置文件 / Configuration File for Agent Project

在这个文件里可以一键调整所有 agent 的运行参数。
You can adjust all agent parameters in one place here.

结构 / Structure:
  - DEFAULT_CONFIG：默认的训练超参
  - TUNABLE_KEYS：允许 GPT 修改的参数列表
  - AGENT_SETTINGS：Agent 全局设置（迭代轮数、模式等）
  - MODEL_SETTINGS：模型相关设置
"""

from typing import Dict, Any, List


# ============================================================================
# 🎓 默认训练配置 / DEFAULT CONFIG
# ============================================================================
# 这些参数会在训练时使用。改这些值就能改变训练行为。
# These parameters are used during training. Change them to alter training behavior.

DEFAULT_CONFIG: Dict[str, Any] = {
    # -------- 功能开关 / Feature Flags --------
    "ENABLE_TRIPLET_EVAL": False,      # 是否启用 Triplet 评估
    "ENABLE_QUORA_TEST": False,         # 是否在 Quora 数据上测试（关闭加快速度）

    # -------- 模型 & 数据路径 / Model & Data Paths --------
    "BASE_MODEL": "sentence-transformers/all-MiniLM-L6-v2",  # 底层模型选择
    "OUTPUT_DIR_ROOT": "models",        # 训练输出根目录
    "RUN_NAME_PREFIX": "stv3_agent_demo_",  # 输出文件夹前缀
    "RUN_NAME": "agent_autotune_demo",  # 训练 run 名称

    # -------- 数据子集 / Data Subset --------
    # 💡 提示：改这些值可以快速测试（train[:200] 用 200 个样本）
    # Tip: Change these for quick testing (e.g., train[:200] uses 200 samples)
    "STSB_TRAIN_SPLIT": "train[:200]",  # STSb 训练集（默认只用前 200 个加快速度）
    "STSB_DEV_SPLIT": "validation[:100]",  # STSb 验证集

    # -------- 训练超参 / Training Hyperparameters --------
    # 💡 这些参数直接影响训练效果，可由 GPT Agent 自动调整
    "NUM_TRAIN_EPOCHS": 1,              # 训练轮数（增加可能更优但更慢）
    "TRAIN_BATCH_SIZE": 8,              # 训练批大小（显存允许可增加到 16/32）
    "EVAL_BATCH_SIZE": 8,               # 评估批大小
    "GRAD_ACC_STEPS": 1,                # 梯度累积步数（可用来模拟更大 batch）
    "LEARNING_RATE": 2e-5,              # 学习率（通常 1e-5 ~ 5e-4）
    "WARMUP_RATIO": 0.1,                # 预热比例

    # -------- 评估 & 保存策略 / Evaluation & Saving Strategy --------
    "EVAL_STRATEGY": "steps",           # 评估频率 (steps / epoch)
    "EVAL_STEPS": 50,                   # 每 N 步评估一次
    "LOGGING_STEPS": 10,                # 每 N 步记录一次日志
    "SAVE_STRATEGY": "epoch",           # 保存策略 (steps / epoch)
    "SAVE_STEPS": 50,                   # 每 N 步保存一次模型
    "SAVE_TOTAL_LIMIT": 1,              # 最多保存最近 N 个检查点（节省磁盘）
    "LOGGING_FIRST_STEP": True,         # 是否记录第一步

    # -------- Quora 测试相关（通常不用） / Quora Test Related --------
    "QUORA_SPLIT": "train[:100]",       # Quora 数据子集
    "QUORA_MAX_PRINT": 2,               # 最多打印几个 Quora 样例
    "QUORA_FIGSIZE": [6, 4],            # 绘图大小
    "QUORA_HIST_BINS": 10,              # 直方图 bin 数
}


# ============================================================================
# 🔧 允许 GPT 修改的参数列表 / TUNABLE KEYS
# ============================================================================
# 这个列表里的参数，GPT Agent 可以自动调整。
# Parameters in this list can be automatically adjusted by the GPT Agent.

TUNABLE_KEYS: List[str] = [
    # 模型相关
    "BASE_MODEL",
    "ENABLE_TRIPLET_EVAL",
    "ENABLE_QUORA_TEST",

    # 数据相关
    "STSB_TRAIN_SPLIT",
    "STSB_DEV_SPLIT",

    # 训练超参（最常调的）
    "NUM_TRAIN_EPOCHS",
    "TRAIN_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "GRAD_ACC_STEPS",
    "LEARNING_RATE",
    "WARMUP_RATIO",

    # 评估和保存
    "EVAL_STRATEGY",
    "EVAL_STEPS",
    "LOGGING_STEPS",
    "SAVE_STRATEGY",
    "SAVE_STEPS",
    "SAVE_TOTAL_LIMIT",
    "LOGGING_FIRST_STEP",

    # Quora 相关
    "QUORA_SPLIT",
    "QUORA_MAX_PRINT",
    "QUORA_FIGSIZE",
    "QUORA_HIST_BINS",
]


# ============================================================================
# ⚙️ Agent 全局设置 / AGENT SETTINGS
# ============================================================================
# 控制 Agent 的行为方式

class AGENT_SETTINGS:
    """Agent 运行时设置"""

    # 调参轮数
    MAX_PRIORITY_PARAMS = 3             # 最多同时优化多少个参数（1~3）
    ROUNDS_PER_PARAM = 3                # 每个参数最多尝试几轮

    # GPT 模型选择
    GPT_MODEL = "gpt-3.5-turbo"         # 使用哪个 GPT 模型（gpt-3.5-turbo / gpt-4）
                                        # 💡 gpt-3.5-turbo 更便宜，gpt-4 更聪明

    # 是否进行交互式提示
    INTERACTIVE_MODE = True             # True 时会询问用户是否继续，False 时自动运行

    # 报告相关
    GENERATE_REPORT = True              # 是否在结束时生成 Markdown 报告
    REPORT_DIR = "docs/reports"         # 报告存储目录

    # 日志级别
    LOG_LEVEL = "INFO"                  # DEBUG / INFO / WARNING


# ============================================================================
# 🤖 模型相关设置 / MODEL SETTINGS
# ============================================================================

class MODEL_SETTINGS:
    """模型和数据集相关常量"""

    # 默认模型选择
    MODELS = {
        "miniLM": "sentence-transformers/all-MiniLM-L6-v2",      # 小模型，快速
        "mpnet": "sentence-transformers/all-mpnet-base-v2",      # 中等模型
        "roberta": "sentence-transformers/all-roberta-large-v1", # 大模型，更强
    }

    # 数据集
    DATASET = "sentence-transformers/stsb"

    # 评估函数
    SIMILARITY_FUNCTION = "cosine"      # cosine / euclidean

    # 设备
    USE_GPU = True                      # 自动检测 GPU 并使用


# ============================================================================
# 📝 快速修改指南 / QUICK MODIFICATION GUIDE
# ============================================================================
"""
想要快速调整 Agent？看这里：

1️⃣ 改变数据量（加快测试）:
   改这两行：
   - STSB_TRAIN_SPLIT: "train[:200]"  → "train[:500]"  (用 500 个样本)
   - STSB_DEV_SPLIT: "validation[:100]"  → "validation[:200]"

2️⃣ 改变底层模型：
   改这一行：
   - BASE_MODEL: "sentence-transformers/all-MiniLM-L6-v2"  
     → "sentence-transformers/all-mpnet-base-v2"

3️⃣ 加快训练（测试用）：
   改这两行：
   - NUM_TRAIN_EPOCHS: 1  → 1  (保持 1)
   - EVAL_STEPS: 50  → 100  (减少评估频率)

4️⃣ 改变 GPT 模型：
   改这一行：
   - GPT_MODEL = "gpt-3.5-turbo"  → "gpt-4"  (更聪明但贵)

5️⃣ 改变调参策略：
   改这两行：
   - MAX_PRIORITY_PARAMS = 3  → 2  (只优化 2 个参数)
   - ROUNDS_PER_PARAM = 3  → 2  (每个参数只尝试 2 轮)

6️⃣ 关闭交互提示（自动化）：
   改这一行：
   - INTERACTIVE_MODE = True  → False  (不会问用户)
"""

