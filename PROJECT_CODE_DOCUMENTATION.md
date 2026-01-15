# Model Tuning Agent - 完整代码文档

> 本文档详细讲解项目中所有代码、脚本、函数的工作原理和调用链接

---

## 📑 目录

1. [项目架构](#项目架构)
2. [核心模块](#核心模块)
3. [工具模块](#工具模块)
4. [脚本集合](#脚本集合)
5. [配置系统](#配置系统)
6. [数据流程](#数据流程)
7. [完整调用链](#完整调用链)

---

## 项目架构

```
model-tuning-agent/
├── 📁 core/                    # 核心训练逻辑
│   ├── __init__.py
│   └── training.py            # ⭐ 训练核心
├── 📁 agents/                  # GPT Agent
│   ├── __init__.py
│   └── gpt_agent.py           # ⭐ GPT交互
├── 📁 utils/                   # 工具函数
│   ├── __init__.py
│   ├── llm.py                 # LLM统一接口
│   ├── openai_client.py       # OpenAI客户端
│   └── report_generator.py    # 报告生成
├── 📁 scripts/                 # 测试脚本
│   ├── run_quick_model_tests.py
│   ├── run_deep_model_tests.py
│   ├── run_phase3_deep_tests.py
│   ├── discover_openrouter_models.py
│   ├── generate_comparison_report.py
│   ├── generate_deep_comparison_report.py
│   └── generate_phase3_comparison_report.py
├── 📁 models/                  # 训练输出
├── 📁 docs/reports/            # 报告目录
├── config.py                  # ⭐ 配置中心
├── openrouter_client.py       # OpenRouter客户端
├── run.py                     # ⭐ 主程序入口
├── setup_api_key.py           # API Key设置
└── setup_openrouter_api_key.py
```

---

## 核心模块

### 1. `run.py` - 主程序入口

**功能：** 协调整个自动调参流程

#### 核心函数

##### `apply_new_config(base_config, new_config)`
```python
def apply_new_config(base_config, new_config) -> Dict:
    """应用GPT建议的配置到当前配置"""
    cfg = deepcopy(base_config)
    for k, v in new_config.items():
        if k in cfg:
            cfg[k] = v
    return cfg
```
- **输入：** 基础配置 + GPT新建议
- **输出：** 合并后的配置
- **用途：** 每轮训练后应用GPT的参数调整建议

##### `run_agent()`
```python
def run_agent() -> None:
    """主协调函数"""
    # 1. 初始化配置
    current_config = make_default_config()
    
    # 2. 选择LLM模型
    select_llm_model()
    
    # 3. 获取初始计划
    plan = ask_gpt_for_initial_plan(config)
    base_cfg = plan["base_config"]
    priority_keys = plan["priority_keys"]
    
    # 4. 多轮调参循环
    for key in priority_keys:
        for inner_round in range(MAX_ROUNDS_PER_KEY):
            # 训练
            summary, metrics = train_one_round(current_config)
            
            # 获取GPT建议
            suggestion = ask_gpt_for_new_config(config, summary)
            
            # 应用建议
            current_config = apply_new_config(current_config, suggestion)
            
            # 更新最佳记录
            if score > best_score:
                best_score = score
                best_config = config
    
    # 5. 生成报告
    generate_run_report(history, best_round, best_score, ...)
    
    # 6. 复制最佳模型
    shutil.copytree(best_output_dir, "best_overall_model")
```

**调用链：**
```
run_agent()
  ├→ make_default_config()           [config.py]
  ├→ select_llm_model()               [内部函数]
  ├→ ask_gpt_for_initial_plan()      [agents/gpt_agent.py]
  ├→ train_one_round()                [core/training.py]
  ├→ ask_gpt_for_new_config()        [agents/gpt_agent.py]
  ├→ apply_new_config()               [内部函数]
  ├→ generate_run_report()            [utils/report_generator.py]
  └→ ask_gpt_for_overall_summary()   [agents/gpt_agent.py]
```

---

### 2. `core/training.py` - 训练核心

#### 核心函数

##### `make_default_config()`
```python
def make_default_config() -> Dict[str, Any]:
    """从config.py返回默认配置"""
    return DEFAULT_CONFIG.copy()
```

##### `export_config_for_agent(config)`
```python
def export_config_for_agent(config) -> Dict:
    """提取可调参数给GPT看"""
    cfg = {}
    for k in TUNABLE_KEYS:
        if k in config:
            cfg[k] = config[k]
    return cfg
```

##### `set_global_seed(seed=42)`
```python
def set_global_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

##### `train_one_round(config, round_id)` ⭐⭐⭐
```python
def train_one_round(config, round_id=1) -> Tuple[Dict, Dict]:
    """执行一轮训练并返回结果"""
    
    # 1. 设置随机种子
    set_global_seed(42)
    
    # 2. 加载数据集
    stsb_train = load_dataset("sentence-transformers/stsb", 
                              split=config["STSB_TRAIN_SPLIT"])
    stsb_dev = load_dataset("sentence-transformers/stsb", 
                            split=config["STSB_DEV_SPLIT"])
    
    # 3. 加载模型
    model = SentenceTransformer(config["BASE_MODEL"])
    
    # 4. 定义损失函数
    loss = CoSENTLoss(model)
    
    # 5. 创建评估器
    evaluator = EmbeddingSimilarityEvaluator(...)
    
    # 6. 配置训练参数
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["NUM_TRAIN_EPOCHS"],
        per_device_train_batch_size=config["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=config["EVAL_BATCH_SIZE"],
        learning_rate=config["LEARNING_RATE"],
        ...
    )
    
    # 7. 创建训练器
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=stsb_train,
        eval_dataset=stsb_dev,
        loss=loss,
        evaluator=evaluator,
    )
    
    # 8. 执行训练
    train_result = trainer.train()
    trainer.save_model(output_dir)
    
    # 9. 获取评估分数
    main_score = evaluator(model)
    
    # 10. 构建返回数据
    summary = {
        "round_id": round_id,
        "output_dir": output_dir,
        "main_score": main_score,
        "metrics": train_result.metrics,
        ...
    }
    
    # 确保metrics包含主评估分数
    if "eval_stsb_dev_spearman_cosine" not in summary["metrics"]:
        summary["metrics"]["eval_stsb_dev_spearman_cosine"] = main_score
    
    return summary, summary["metrics"]
```

**返回值结构：**
```python
summary = {
    "round_id": 1,
    "output_dir": "models/stv3_agent_demo_20251224_112628_r1",
    "device": "cuda",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "stsb_train_size": 1500,
    "stsb_dev_size": 300,
    "main_score": 0.9397,
    "metrics": {
        "train_runtime": 17.32,
        "train_samples_per_second": 259.8,
        "eval_stsb_dev_spearman_cosine": 0.9397
    }
}
```

---

### 3. `agents/gpt_agent.py` - GPT交互模块

#### 核心函数

##### `build_agent_input(...)`
```python
def build_agent_input(config_for_agent, training_summary, 
                      history=None, primary_key=None) -> str:
    """构建发送给GPT的JSON输入"""
    payload = {
        "config": config_for_agent,
        "training_summary": training_summary,
    }
    if history:
        payload["history"] = history[-10:]  # 只保留最近10轮
    if primary_key:
        payload["primary_key"] = primary_key
    return json.dumps(payload, ensure_ascii=False, indent=2)
```

##### `ask_gpt_for_initial_plan(config)` ⭐
```python
def ask_gpt_for_initial_plan(config, model="gpt-3.5-turbo") -> Dict:
    """第0步：让GPT制定初始调参计划"""
    
    # 1. 构建系统提示词
    system_prompt = """
    你是深度学习工程师...
    请返回：
    {
      "comment": "选择理由",
      "base_config": {...},
      "priority_keys": ["key1", "key2", "key3"]
    }
    """
    
    # 2. 构建用户输入
    user_input = json.dumps({
        "config": config,
        "tunable_keys": TUNABLE_KEYS
    })
    
    # 3. 调用LLM
    response = chat(
        source=AGENT_SETTINGS.LLM_SOURCE,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    # 4. 解析响应
    return json.loads(response)
```

**返回示例：**
```json
{
  "comment": "学习率和训练轮数影响最大",
  "base_config": {
    "LEARNING_RATE": 2e-5,
    "NUM_TRAIN_EPOCHS": 3
  },
  "priority_keys": ["LEARNING_RATE", "NUM_TRAIN_EPOCHS", "TRAIN_BATCH_SIZE"]
}
```

##### `ask_gpt_for_new_config(...)` ⭐
```python
def ask_gpt_for_new_config(config, summary, model, 
                           history=None, primary_key=None) -> Dict:
    """每轮训练后，让GPT建议新的参数值"""
    
    # 1. 构建输入
    user_input = build_agent_input(config, summary, history, primary_key)
    
    # 2. 系统提示词
    system_prompt = """
    基于训练结果，建议下一轮的参数值...
    返回：
    {
      "comment": "分析和建议",
      "new_config": {"参数名": 新值}
    }
    """
    
    # 3. 调用LLM
    response = chat(
        source=AGENT_SETTINGS.LLM_SOURCE,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    return json.loads(response)
```

##### `ask_gpt_for_overall_summary(...)`
```python
def ask_gpt_for_overall_summary(history, best_round, 
                                best_score, best_config, model) -> str:
    """生成整体总结评价"""
    # 让GPT总结整个调参过程的效果
    ...
```

---

### 4. `utils/llm.py` - LLM统一接口

#### 核心函数

##### `chat(source, model, messages, temperature)` ⭐
```python
def chat(source="openai", model="gpt-3.5-turbo", 
         messages=None, temperature=0.3, **kwargs) -> str:
    """统一的LLM调用接口"""
    
    if source == "openai":
        # 使用OpenAI SDK
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        return completion.choices[0].message.content
    
    elif source == "openrouter":
        # 使用OpenRouter客户端
        api_key = _load_openrouter_api_key()
        client = OpenRouterClient(api_key=api_key)
        resp = client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs
        )
        # 解析多种可能的响应格式
        return _extract_content_from_response(resp)
    
    else:
        raise ValueError(f"Unknown source: {source}")
```

**支持的LLM源：**
- `openai`: OpenAI GPT (需要API Key)
- `openrouter`: OpenRouter免费模型

##### `list_openrouter_free_models(limit=5)`
```python
def list_openrouter_free_models(limit=5) -> List[Dict]:
    """列出OpenRouter的免费模型"""
    client = OpenRouterClient()
    models = client.list_models(free_only=True)
    
    # 按关键词过滤和排序
    scored = []
    for m in models:
        score = _score_by_keywords(m)
        scored.append((score, m))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]
```

---

### 5. `utils/report_generator.py` - 报告生成器

#### 核心函数

##### `generate_run_report(...)` ⭐
```python
def generate_run_report(history, best_round, best_score, 
                        best_config, priority_keys, base_cfg,
                        model_label=None) -> str:
    """生成训练报告Markdown文件"""
    
    # 1. 创建文件路径
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_run_report_{ts}.md"
    report_path = os.path.join("docs/reports", filename)
    
    # 2. 构建内容
    lines = []
    
    # 标题
    lines.append(f"# Agent 运行报告 ({ts}) — Model: {model_label}\n")
    
    # 最终结果摘要
    display_score = _extract_best_score(best_score, history, best_round)
    lines.append("## 最终结果摘要\n")
    lines.append(f"**最优分数:** {display_score:.4f}\n")
    lines.append(f"**调整的参数:** {', '.join(priority_keys)}\n")
    
    # 详细配置
    for k, v in sorted(best_config.items()):
        lines.append(f"  - {k}: {v}\n")
    
    # 详细逐轮记录
    lines.append("## 详细逐轮记录\n")
    for h in history:
        score = _extract_score_from_history(h)
        lines.append(f"### 轮次 {h['round_id']}\n")
        lines.append(f"**分数:** {score:.4f}\n")
        lines.append(f"**配置:** {json.dumps(h['config_for_agent'])}\n")
        
        # 训练指标
        if "metrics" in h:
            metrics = h["metrics"]
            lines.append("**训练指标:**\n")
            lines.append(f"- 训练时间: {metrics.get('train_runtime', 0):.2f}秒\n")
            lines.append(f"- 样本速度: {metrics.get('train_samples_per_second', 0):.2f}\n")
    
    # 建议
    lines.append("## 建议\n")
    lines.append("1. 可增加训练轮数\n")
    lines.append("2. 可扩大数据集\n")
    
    # 3. 写入文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return report_path
```

**分数提取优先级：**
```python
def _extract_score_from_history(h):
    # 1. 从main_score字段
    if h.get("main_score", 0.0) != 0.0:
        return h["main_score"]
    
    # 2. 从metrics中的多种键名
    if "metrics" in h:
        for key in ["eval_stsb_dev_spearman_cosine", 
                    "eval_spearman_cosine", 
                    "eval_cosine"]:
            if key in h["metrics"]:
                return float(h["metrics"][key])
    
    return 0.0
```

---

## 工具模块

### 1. `utils/openai_client.py` - OpenAI客户端

```python
def _load_api_key():
    """加载API Key（优先级：.env > 环境变量）"""
    # 从.env文件读取
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        # 解析OPENAI_API_KEY=xxx
        ...
    
    # 从环境变量读取
    return os.environ.get("OPENAI_API_KEY")

# 初始化全局客户端
api_key = _load_api_key()
client = OpenAI(api_key=api_key)
```

### 2. `openrouter_client.py` - OpenRouter客户端

```python
class OpenRouterClient:
    """OpenRouter API客户端"""
    
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"
    
    def list_models(self, free_only=True) -> List[Dict]:
        """列出可用模型"""
        endpoint = f"{self.base_url}/models"
        resp = requests.get(endpoint, headers=self._headers())
        data = resp.json()
        
        # 过滤免费模型
        if free_only:
            return [m for m in data["models"] if self._is_free(m)]
        return data["models"]
    
    def chat_completion(self, model, messages, **kwargs) -> Dict:
        """调用chat/completions端点"""
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        resp = requests.post(endpoint, json=payload, headers=self._headers())
        return resp.json()
```

---

## 脚本集合

### 1. 测试脚本

#### `scripts/run_quick_model_tests.py` - 快速测试
```python
"""快速验证测试（20个样本，1轮训练）"""

MODELS = [
    ("openai", "gpt-3.5-turbo"),
    ("openrouter", "xiaomi/mimo-v2-flash"),
    ("openrouter", "nvidia/nemotron-3-nano-30b-a3b"),
    ...
]

QUICK_OVERRIDES = {
    "STSB_TRAIN_SPLIT": "train[:20]",
    "STSB_DEV_SPLIT": "validation[:5]",
    "NUM_TRAIN_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 4,
}

def run_quick_test(source, model_id):
    # 1. 准备配置
    cfg = make_default_config()
    cfg.update(QUICK_OVERRIDES)
    
    # 2. 设置模型
    AGENT_SETTINGS.LLM_SOURCE = source
    AGENT_SETTINGS.GPT_MODEL = model_id
    
    # 3. 运行训练
    summary, metrics = train_one_round(cfg, round_id=1)
    
    # 4. 构建history
    history = [{
        "round_id": 1,
        "tuned_key": "quick_test",
        "main_score": summary.get("main_score", 0.0),
        "metrics": metrics,
    }]
    
    # 5. 生成报告
    generate_run_report(history, 1, summary["main_score"], cfg, ...)
```

**运行：** `python scripts/run_quick_model_tests.py`

#### `scripts/run_deep_model_tests.py` - 深度测试
```python
"""深度测试（1500个样本，3轮训练）"""

DEEP_TEST_CONFIG = {
    "STSB_TRAIN_SPLIT": "train[:1500]",
    "STSB_DEV_SPLIT": "validation[:300]",
    "NUM_TRAIN_EPOCHS": 3,
    "TRAIN_BATCH_SIZE": 16,
}

def run_deep_test(llm_source, model_name, model_label):
    # 与quick_test类似，但使用更大的数据集和更多轮次
    ...
```

**运行：** `python scripts/run_deep_model_tests.py`

#### `scripts/run_phase3_deep_tests.py` - 超深度测试
```python
"""超深度测试（4500个样本，6轮训练）"""

PHASE3_CONFIG = {
    "STSB_TRAIN_SPLIT": "train[:4500]",
    "STSB_DEV_SPLIT": "validation[:900]",
    "NUM_TRAIN_EPOCHS": 6,
    "TRAIN_BATCH_SIZE": 16,
}
```

**运行：** `python scripts/run_phase3_deep_tests.py`

### 2. 工具脚本

#### `scripts/discover_openrouter_models.py`
```python
"""发现并列出OpenRouter可用模型"""

def discover_models():
    client = OpenRouterClient()
    models = client.list_models(free_only=True)
    
    # 按类别分组
    for model in models:
        print(f"- {model['id']}")
        print(f"  Context: {model.get('context_length', 'N/A')}")
        print(f"  Provider: {model.get('provider', 'N/A')}")
```

**运行：** `python scripts/discover_openrouter_models.py`

#### `scripts/generate_comparison_report.py`
```python
"""从多个测试报告生成对比报告"""

def extract_report_info(report_path):
    """从报告文件提取关键信息"""
    # 解析Markdown，提取：
    # - 模型名称
    # - 最优分数
    # - 训练时间
    # - 样本速度
    ...

def generate_comparison_report(report_files):
    """生成对比表格"""
    results = []
    for report_file in report_files:
        info = extract_report_info(report_file)
        results.append(info)
    
    # 排序并生成Markdown表格
    results.sort(key=lambda x: x["best_score"], reverse=True)
    ...
```

**运行：** `python scripts/generate_comparison_report.py`

### 3. 设置脚本

#### `setup_api_key.py`
```python
"""设置OpenAI API Key到.env文件"""

def setup_api_key(api_key=None):
    if not api_key:
        api_key = getpass.getpass("API Key: ")
    
    # 写入.env文件
    env_file = Path(__file__).parent / ".env"
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print("✓ .env文件已创建")
```

**运行：** `python setup_api_key.py`

#### `setup_openrouter_api_key.py`
```python
"""设置OpenRouter API Key"""
# 类似setup_api_key.py，但设置OPENROUTER_API_KEY
```

**运行：** `python setup_openrouter_api_key.py`

---

## 配置系统

### `config.py` - 配置中心

```python
# ===== 默认训练配置 =====
DEFAULT_CONFIG = {
    # 模型和数据
    "BASE_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "STSB_TRAIN_SPLIT": "train[:200]",
    "STSB_DEV_SPLIT": "validation[:100]",
    
    # 训练超参
    "NUM_TRAIN_EPOCHS": 1,
    "TRAIN_BATCH_SIZE": 8,
    "EVAL_BATCH_SIZE": 8,
    "LEARNING_RATE": 2e-5,
    "WARMUP_RATIO": 0.1,
    
    # 评估策略
    "EVAL_STRATEGY": "steps",
    "EVAL_STEPS": 50,
    "LOGGING_STEPS": 10,
    
    # 功能开关
    "ENABLE_TRIPLET_EVAL": False,
    "ENABLE_QUORA_TEST": False,
}

# ===== 可调参数列表 =====
TUNABLE_KEYS = [
    "BASE_MODEL",
    "STSB_TRAIN_SPLIT",
    "STSB_DEV_SPLIT",
    "NUM_TRAIN_EPOCHS",
    "TRAIN_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "LEARNING_RATE",
    "WARMUP_RATIO",
    ...
]

# ===== Agent设置 =====
class AgentSettings:
    GPT_MODEL = "gpt-3.5-turbo"
    LLM_SOURCE = "openai"  # "openai" or "openrouter"
    MAX_ROUNDS = 10
    MAX_ROUNDS_PER_KEY = 5
    INTERACTIVE_MODE = True

AGENT_SETTINGS = AgentSettings()

# ===== OpenRouter配置 =====
OPENROUTER = {
    "API_KEY_ENV": "OPENROUTER_API_KEY",
    "API_BASE": "https://openrouter.ai/api/v1",
}
```

---

## 数据流程

### 完整数据流

```
┌─────────────┐
│ 用户启动    │
│ python run.py│
└──────┬──────┘
       ▼
┌─────────────────────────────────────────┐
│ 1. 初始化配置                           │
│    - 加载DEFAULT_CONFIG                 │
│    - 选择LLM模型（OpenAI/OpenRouter）   │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 2. GPT制定初始计划                      │
│    ask_gpt_for_initial_plan()          │
│    ├→ 返回base_config                   │
│    └→ 返回priority_keys                 │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 3. 多轮调参循环                         │
│    for key in priority_keys:           │
│      for round in range(MAX_ROUNDS):   │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 4. 执行训练                             │
│    train_one_round(config, round_id)   │
│    ├→ 加载数据集（STSb）                │
│    ├→ 加载模型（SentenceTransformer）   │
│    ├→ 配置训练器（Trainer）             │
│    ├→ 执行训练（trainer.train()）       │
│    └→ 评估并返回(summary, metrics)      │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 5. 记录结果                             │
│    history.append({                     │
│      "round_id": id,                    │
│      "config": config,                  │
│      "main_score": score,               │
│      "metrics": metrics                 │
│    })                                   │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 6. GPT分析建议                          │
│    ask_gpt_for_new_config()            │
│    ├→ 输入：当前config + 训练结果       │
│    └→ 返回：new_config建议              │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 7. 应用建议                             │
│    current_config = apply_new_config()  │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 8. 更新最佳记录                         │
│    if score > best_score:               │
│      best_score = score                 │
│      best_config = config               │
│      best_round = round_id              │
└──────┬──────────────────────────────────┘
       ▼
       │ 重复步骤3-8，直到所有参数调整完成
       ▼
┌─────────────────────────────────────────┐
│ 9. 生成报告                             │
│    generate_run_report()                │
│    └→ 输出：docs/reports/xxx.md         │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 10. 复制最佳模型                        │
│     shutil.copytree()                   │
│     └→ 输出：models/best_overall_model/ │
└──────┬──────────────────────────────────┘
       ▼
┌─────────────────────────────────────────┐
│ 11. GPT生成总结                         │
│     ask_gpt_for_overall_summary()      │
└─────────────────────────────────────────┘
```

---

## 完整调用链

### 主程序调用链（run.py）

```
main()
└─ run_agent()
   ├─ make_default_config()                    [core/training.py]
   ├─ list_openrouter_free_models()            [utils/llm.py]
   │  └─ OpenRouterClient.list_models()        [openrouter_client.py]
   ├─ ask_gpt_for_initial_plan()               [agents/gpt_agent.py]
   │  └─ chat()                                 [utils/llm.py]
   │     ├─ openai_client.chat.completions.create()  [OpenAI SDK]
   │     └─ OpenRouterClient.chat_completion()      [openrouter_client.py]
   │
   ├─ 多轮调参循环
   │  ├─ train_one_round()                     [core/training.py]
   │  │  ├─ set_global_seed()
   │  │  ├─ load_dataset()                     [datasets库]
   │  │  ├─ SentenceTransformer()              [sentence-transformers]
   │  │  ├─ CoSENTLoss()
   │  │  ├─ EmbeddingSimilarityEvaluator()
   │  │  ├─ SentenceTransformerTrainer.train()
   │  │  └─ evaluator(model)
   │  │
   │  ├─ ask_gpt_for_new_config()              [agents/gpt_agent.py]
   │  │  ├─ build_agent_input()
   │  │  └─ chat()                             [utils/llm.py]
   │  │
   │  └─ apply_new_config()                    [run.py]
   │
   ├─ generate_run_report()                    [utils/report_generator.py]
   ├─ shutil.copytree()                        [Python标准库]
   └─ ask_gpt_for_overall_summary()            [agents/gpt_agent.py]
      └─ chat()                                 [utils/llm.py]
```

### 测试脚本调用链（scripts/run_deep_model_tests.py）

```
main()
├─ for each model in MODELS:
│  └─ run_deep_test(model)
│     ├─ AGENT_SETTINGS设置
│     ├─ make_default_config()               [core/training.py]
│     ├─ train_one_round(config, 1)          [core/training.py]
│     │  └─ (完整训练流程，见上)
│     ├─ 构建history字典
│     └─ generate_run_report()               [utils/report_generator.py]
│        ├─ 创建文件路径
│        ├─ 构建Markdown内容
│        └─ 写入文件
│
└─ generate_comparison_report()              [scripts/generate_comparison_report.py]
   ├─ 读取所有报告文件
   ├─ 提取关键信息
   ├─ 排序比较
   └─ 生成对比表格
```

---

## 关键数据结构

### 1. Config字典
```python
{
    "BASE_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "STSB_TRAIN_SPLIT": "train[:1500]",
    "STSB_DEV_SPLIT": "validation[:300]",
    "NUM_TRAIN_EPOCHS": 3,
    "TRAIN_BATCH_SIZE": 16,
    "EVAL_BATCH_SIZE": 16,
    "LEARNING_RATE": 2e-5,
    "WARMUP_RATIO": 0.1,
    "EVAL_STRATEGY": "steps",
    "EVAL_STEPS": 50,
    ...
}
```

### 2. Summary字典（训练结果）
```python
{
    "round_id": 1,
    "output_dir": "models/stv3_agent_demo_20251224_112628_r1",
    "device": "cuda",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "stsb_train_size": 1500,
    "stsb_dev_size": 300,
    "main_score": 0.9397,
    "metrics": {
        "train_runtime": 17.32,
        "train_samples_per_second": 259.8,
        "train_loss": 0.0234,
        "eval_stsb_dev_spearman_cosine": 0.9397
    }
}
```

### 3. History列表（多轮记录）
```python
[
    {
        "round_id": 1,
        "tuned_key": "LEARNING_RATE",
        "inner_round_index": 1,
        "config_for_agent": {...},
        "main_score": 0.9200,
        "metrics": {...},
        "summary": {...}
    },
    {
        "round_id": 2,
        "tuned_key": "LEARNING_RATE",
        "inner_round_index": 2,
        "config_for_agent": {...},
        "main_score": 0.9350,
        "metrics": {...},
        "summary": {...}
    },
    ...
]
```

### 4. GPT响应格式

**初始计划：**
```json
{
  "comment": "学习率和训练轮数影响最大",
  "base_config": {
    "LEARNING_RATE": 2e-5,
    "NUM_TRAIN_EPOCHS": 3
  },
  "priority_keys": ["LEARNING_RATE", "NUM_TRAIN_EPOCHS", "TRAIN_BATCH_SIZE"]
}
```

**新配置建议：**
```json
{
  "comment": "当前学习率过大，建议降低",
  "new_config": {
    "LEARNING_RATE": 1e-5
  }
}
```

---

## 运行示例

### 1. 主程序（交互式多轮调参）
```bash
python run.py
```

**执行流程：**
1. 选择LLM模型（OpenAI/OpenRouter）
2. GPT制定调参计划（选择priority_keys）
3. 逐个参数进行多轮调优
4. 每轮训练后GPT分析并建议新值
5. 自动记录最佳配置和分数
6. 生成详细报告
7. 复制最佳模型

### 2. 快速测试（单次，5个模型）
```bash
python scripts/run_quick_model_tests.py
```

**参数：** 20个样本，1轮训练，batch_size=4  
**用时：** 约5-10秒  
**输出：** 5份报告

### 3. 深度测试（单次，5个模型）
```bash
python scripts/run_deep_model_tests.py
```

**参数：** 1500个样本，3轮训练，batch_size=16  
**用时：** 约1.5分钟  
**输出：** 5份报告 + 对比报告

### 4. 超深度测试（单次，5个模型）
```bash
python scripts/run_phase3_deep_tests.py
```

**参数：** 4500个样本，6轮训练，batch_size=16  
**用时：** 约10分钟  
**输出：** 5份报告 + 详细对比报告

### 5. 生成对比报告
```bash
python scripts/generate_comparison_report.py
```

**功能：** 从docs/reports/目录读取所有报告，生成对比表格

### 6. 发现可用模型
```bash
python scripts/discover_openrouter_models.py
```

**功能：** 列出OpenRouter上所有免费可用的模型

---

## 模块依赖关系

```
run.py
├── config.py
├── core/training.py
│   ├── config.py
│   ├── datasets (外部库)
│   ├── sentence-transformers (外部库)
│   └── torch (外部库)
├── agents/gpt_agent.py
│   ├── config.py
│   ├── core/training.py
│   └── utils/llm.py
│       ├── openrouter_client.py
│       │   ├── config.py
│       │   └── requests (外部库)
│       └── utils/openai_client.py
│           ├── openai (外部库)
│           └── pathlib
└── utils/report_generator.py
    ├── json
    ├── datetime
    └── pathlib

scripts/
├── run_quick_model_tests.py
│   ├── core/training.py
│   ├── config.py
│   └── utils/report_generator.py
├── run_deep_model_tests.py
│   └── (同上)
├── run_phase3_deep_tests.py
│   └── (同上)
└── generate_comparison_report.py
    ├── pathlib
    ├── re
    └── datetime
```

---

## 外部依赖库

```python
# 核心训练
torch                    # PyTorch深度学习框架
sentence-transformers    # Sentence Transformer模型
transformers            # Hugging Face Transformers
datasets                # Hugging Face Datasets

# LLM交互
openai                  # OpenAI SDK
requests               # HTTP请求（OpenRouter）

# 数据处理
numpy                   # 数值计算
pandas                 # 数据处理（可选，Quora测试用）

# 可视化
matplotlib             # 绘图（可选，Quora测试用）
```

**安装：**
```bash
pip install torch sentence-transformers transformers datasets openai requests numpy
```

---

## 总结

### 核心工作流程

1. **配置初始化** → `config.py` 提供默认配置
2. **GPT制定计划** → `agents/gpt_agent.py` 选择优先参数
3. **执行训练** → `core/training.py` 训练Sentence Transformer
4. **GPT分析建议** → `agents/gpt_agent.py` 给出新参数值
5. **迭代优化** → 重复3-4步，寻找最优配置
6. **生成报告** → `utils/report_generator.py` 记录完整过程

### 关键特性

- ✅ **模块化设计**：核心、Agent、工具分离
- ✅ **配置中心化**：所有参数在config.py统一管理
- ✅ **LLM统一接口**：支持OpenAI和OpenRouter
- ✅ **自动化调参**：GPT智能建议参数调整
- ✅ **完整记录**：每轮训练生成详细报告
- ✅ **容错机制**：多种分数键名支持，超时保护
- ✅ **测试脚本**：快速/深度/超深度三个测试级别

### 扩展建议

1. **添加新模型**：在`config.py`的`MODELS`列表中添加
2. **修改训练参数**：编辑`DEFAULT_CONFIG`
3. **自定义调参策略**：修改`agents/gpt_agent.py`的提示词
4. **添加新评估指标**：扩展`core/training.py`的评估器
5. **支持新LLM**：在`utils/llm.py`添加新的source分支

---

**文档版本：** v1.0  
**更新日期：** 2026-01-05  
**项目：** Model Tuning Agent
