# 本地 LLM 微调入门实战
## Unsloth + LoRA/QLoRA + Ollama 完整学习项目

> 目标：从零开始，微调一个本地 LLM，部署到 Ollama，跑在自己机器上。
> 本文档配合 Claude Code 生成完整学习代码，每节末尾附有代码生成提示词。

---

## 目录

1. [微调是什么：三种方式对比](#1-微调是什么三种方式对比)
2. [LoRA 原理深度解析](#2-lora-原理深度解析)
3. [工具链选择](#3-工具链选择)
4. [环境搭建](#4-环境搭建)
5. [数据集准备](#5-数据集准备)
6. [用 Unsloth 微调（核心）](#6-用-unsloth-微调核心)
7. [超参数调优指南](#7-超参数调优指南)
8. [导出 GGUF 并部署到 Ollama](#8-导出-gguf-并部署到-ollama)
9. [评估模型效果](#9-评估模型效果)
10. [完整学习项目：支付领域专家模型](#10-完整学习项目支付领域专家模型)

---

## 1. 微调是什么：三种方式对比

### 类比理解

```
预训练模型 = 一个博学的通才（读了互联网上所有文章）

你的目标 = 让这个通才变成你领域的专家

三种方式：

全量微调（FFT）：
  重新雇用这位博士，让他去你公司上班，
  完整参加所有岗前培训（成本极高）

LoRA 微调：
  给他戴一副特制眼镜（Adapter），
  只训练眼镜，不动他原有的知识

QLoRA 微调：
  先把博士"压缩"（4bit量化），
  再给他戴眼镜（更省内存的 LoRA）
```

### 三种方式详细对比

| 方式 | 原理 | VRAM 需求 | 训练速度 | 效果 | 适合场景 |
|------|------|----------|---------|------|---------|
| **FFT**（全量微调）| 更新所有参数（70亿个）| 极高（~80GB+）| 慢 | 最好 | 土豪/大公司 |
| **LoRA** | 只训练 ~1% 的新增参数 | 中（~16GB）| 快 | ≈FFT | **推荐** |
| **QLoRA** | 4bit 压缩 + LoRA | 低（~6GB）| 较快 | 略低 | **入门推荐** |

### 什么时候需要微调？

```
✅ 适合微调：
  - 想让模型说话风格固定（如：总用 JSON 回答）
  - 特定领域术语/格式（如：ISO 20022 支付消息）
  - 特定语言/方言（如：新西兰英语 + Māori 词汇）
  - 减少幻觉（用高质量领域数据训练）

❌ 不适合微调：
  - 想让模型知道最新信息 → 用 RAG
  - 数据集少于 50 条   → 用 few-shot prompting
  - 想要推理能力       → 用更大的基础模型
  - 任务复杂多变       → 用 Agent + Tools
```

### 微调 vs RAG

```
              微调                    RAG
数据存储    → 烧入模型权重           → 向量数据库
更新方式   → 重新训练               → 更新文档即可
推理速度   → 快（无检索步骤）        → 较慢（需检索）
适合数据   → 格式/风格/领域知识      → 频繁更新的文档
部署复杂度 → 低（就一个模型）        → 高（模型+DB+检索）

最佳实践：微调 + RAG 结合使用
  微调：让模型懂领域语言和格式
  RAG：让模型获取最新/私有数据
```

---

## 2. LoRA 原理深度解析

### Transformer 权重矩阵

```
LLM 的每一层 Transformer 包含多个权重矩阵：

Q_proj（Query 投影）    shape: [hidden, hidden]
K_proj（Key 投影）      shape: [hidden, hidden]
V_proj（Value 投影）    shape: [hidden, hidden]
O_proj（Output 投影）   shape: [hidden, hidden]
gate_proj, up_proj, down_proj（FFN 层）

对 Llama-3-8B：
  hidden_dim = 4096
  每个矩阵大小 = 4096 × 4096 = 16M 个参数
  32 层 × 7 个矩阵 = 3.6B 个参数需要更新
  → 全量微调：更新 36 亿个数字，显存爆炸
```

### LoRA 的核心思想

```
原始权重矩阵 W（4096 × 4096，冻结，不训练）

LoRA 在旁边加两个小矩阵：
  A（4096 × r）  r = rank，通常 8~64
  B（r × 4096）

推理时：output = x @ (W + A @ B)
              = x @ W + x @ A @ B
              ↑冻结      ↑只训练这里

参数量对比（r=16）：
  W: 4096 × 4096 = 16,777,216 个参数
  A+B: 4096×16 + 16×4096 = 131,072 个参数
  节省比例：131K / 16M = 0.78%

全模型的 LoRA 参数：
  约 1% 的原始参数 ≈ 几百万个
  vs FFT 的几十亿个
```

### LoRA 矩阵图解

```
前向传播：

输入 x（1 × 4096）
       │
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
[冻结的原始权重 W]                    [可训练的 LoRA]
(4096 × 4096)                        A (4096 × r)
       │                                      │
       │                                      ▼
       │                               B (r × 4096)
       │                                      │
       ▼                                      ▼
  W 的输出              +              A@B 的输出
       │                                      │
       └──────────────────┬───────────────────┘
                          │
                    最终输出（合并）

训练时：只有 A 和 B 的梯度被计算和更新
```

### QLoRA：进一步节省显存

```
普通 LoRA：
  W 用 FP16 存储（每个参数 2 字节）
  8B 模型权重 = 8B × 2 = 16 GB

QLoRA：
  W 用 NF4（4-bit Normal Float）量化存储
  8B 模型权重 = 8B × 0.5 = 4 GB
  （精度损失很小，4-bit 量化专为 LLM 优化）

  A 和 B 仍然用 BF16 训练（保证梯度精度）

显存对比（Llama-3-8B）：
  FFT:    ~80 GB VRAM
  LoRA:   ~16 GB VRAM
  QLoRA:  ~6 GB VRAM    ← RTX 3060/4060 可跑！
```

---

> **💡 Claude Code 提示词 — LoRA 原理可视化：**
> ```
> 用 Python + numpy 实现 LoRA 的核心机制演示（不需要 GPU）：
>
> 1. LoRALayer 类：
>    - __init__(in_features, out_features, r=16, alpha=32)
>    - 初始化：W（随机，冻结），A（随机小值），B（零初始化）
>    - forward(x) → x @ (W + scale * A @ B)
>      scale = alpha / r（LoRA 的缩放因子）
>    - count_params() → 打印 W 的参数量 vs A+B 的参数量及节省比例
>
> 2. 训练演示（用 numpy 手写简单梯度下降）：
>    - 目标：让 LoRALayer 学会把输入向量旋转 45 度
>    - 冻结 W，只更新 A 和 B
>    - 训练 100 步，打印 loss 曲线
>    - 对比：同样的任务，更新 W（FFT 方式）需要多少步？
>
> 3. rank 影响实验：
>    - 用 r = [1, 4, 8, 16, 32, 64] 分别训练
>    - 画出：参数量 vs 最终 loss 的曲线
>    - 展示 rank 与表达能力的权衡
>
> 4. QLoRA 量化模拟：
>    - 实现简单的 4-bit 量化（把 float32 映射到 16 个离散值）
>    - 计算量化误差
>    - 展示 W 量化后 + A@B 补偿的精度恢复效果
>
> 5. 所有结果用 matplotlib 可视化
> ```

---

## 3. 工具链选择

### 完整工具链

```
数据准备
  └── Python + HuggingFace datasets

微调训练
  ├── Unsloth ⭐    （速度最快，显存最省，推荐）
  ├── HF PEFT      （官方标准库，Unsloth 兼容）
  └── HF TRL       （SFTTrainer，Unsloth 使用它）

模型格式转换
  └── llama.cpp    （GGUF 格式，Ollama 使用）

本地部署
  ├── Ollama ⭐     （最简单，CLI + API）
  ├── LM Studio    （GUI 界面）
  └── Open WebUI   （Web 界面，连接 Ollama）

评估
  └── Python 自定义评估脚本
```

### Unsloth 为什么是首选？

```
相比 HuggingFace 原生：

速度：快 2×（自定义 CUDA kernel）
显存：省 70%（更高效的内存管理）
精度：完全相同（不降精度）
易用：几行代码即可

Unsloth 支持的模型（2025）：
  Llama 3.1/3.2/3.3（1B ~ 70B）
  Qwen 2.5/3.5（0.5B ~ 72B）
  Mistral / Mixtral
  Phi-3 / Phi-4
  Gemma 2 / 3
  ...几乎所有主流开源模型
```

### 模型选择建议

```
按显存选择基础模型：

VRAM    推荐模型（QLoRA）         推荐模型（LoRA）
4GB   → Qwen2.5-1.5B            -
6GB   → Llama-3.2-3B            Qwen2.5-1.5B
8GB   → Llama-3.1-8B ⭐         Qwen2.5-3B
12GB  → Llama-3.1-8B            Llama-3.2-3B ⭐
16GB  → Qwen2.5-14B             Llama-3.1-8B ⭐
24GB  → Qwen2.5-32B             Qwen2.5-14B
80GB  → Llama-3.1-70B           Qwen2.5-32B

没有 GPU？
  → Google Colab（免费 T4，15GB VRAM）
  → Kaggle（免费 P100，16GB VRAM）
  → RunPod / vast.ai（按小时租 GPU）

学习项目推荐：
  Qwen2.5-1.5B（QLoRA，4GB VRAM，速度极快）
  或 Llama-3.2-3B（QLoRA，6GB VRAM）
```

---

## 4. 环境搭建

### 方案 A：本地 GPU（推荐 NVIDIA GPU）

```bash
# 1. 创建 conda 环境
conda create -n finetune python=3.11 -y
conda activate finetune

# 2. 安装 PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 Unsloth（核心）
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 4. 安装其他依赖
pip install datasets trl peft transformers accelerate
pip install bitsandbytes  # QLoRA 需要
pip install wandb          # 训练监控（可选）

# 5. 验证安装
python -c "import unsloth; print('Unsloth OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 6. 安装 Ollama（部署用）
curl -fsSL https://ollama.ai/install.sh | sh
```

### 方案 B：Google Colab（免费，零配置）

```python
# Colab 第一个 Cell 运行：
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes

# 验证 GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 方案 C：Mac M 系列（Apple Silicon）

```bash
# Mac 使用 mlx-lm（Apple 官方的 Metal 加速库）
pip install mlx-lm

# 或者用 Unsloth 的 Mac 版本（实验性）
pip install "unsloth[mac]"

# 推荐：Mac 上用 Ollama + 小模型
# 训练用 Colab，推理部署用本地 Mac
```

---

## 5. 数据集准备

数据集质量 > 数据集数量。100 条高质量数据 > 1000 条低质量数据。

### 5.1 数据格式：ShareGPT 格式（推荐）

这是最通用的对话数据格式，Unsloth 直接支持：

```json
[
  {
    "conversations": [
      {"from": "human",  "value": "什么是 AM04 错误码？"},
      {"from": "gpt",    "value": "AM04 是 ISO 20022 标准中的错误码，表示账户余额不足（Insufficient Funds）。当发起方账户没有足够资金完成转账时，收单银行会返回此错误码。\n\n处理建议：\n1. 通知客户补充余额\n2. 最多重试 3 次（间隔 5 分钟）\n3. 记录错误日志，更新交易状态为 PENDING_RETRY"}
    ]
  }
]
```

### 5.2 数据格式：Alpaca 格式（简单任务）

```json
[
  {
    "instruction": "解释以下 ISO 20022 字段的含义",
    "input":       "MsgId",
    "output":      "MsgId（Message Identification）是报文的唯一标识符，由发送方生成，格式为字母数字，最长 35 个字符。用于追踪和匹配请求/响应报文对。"
  }
]
```

### 5.3 对话格式转换

```python
# 把原始数据转成 ShareGPT 格式
def convert_to_sharegpt(raw_data: list[dict]) -> list[dict]:
    """
    raw_data 格式：[{"question": "...", "answer": "..."}]
    输出格式：ShareGPT conversations
    """
    result = []
    for item in raw_data:
        result.append({
            "conversations": [
                {"from": "human", "value": item["question"]},
                {"from": "gpt",   "value": item["answer"]}
            ]
        })
    return result

# 多轮对话格式
def create_multiturn(history: list[tuple], system: str = "") -> dict:
    """
    history: [(question1, answer1), (question2, answer2), ...]
    """
    convs = []
    if system:
        convs.append({"from": "system", "value": system})
    for q, a in history:
        convs.append({"from": "human", "value": q})
        convs.append({"from": "gpt",   "value": a})
    return {"conversations": convs}
```

### 5.4 数据质量检查

```python
from datasets import Dataset
import json

def validate_dataset(data: list[dict]) -> dict:
    """数据集质量检查"""
    issues = []
    stats = {
        "total": len(data),
        "avg_input_len": 0,
        "avg_output_len": 0,
        "empty_outputs": 0,
        "too_short_outputs": 0,  # < 20 字
        "too_long": 0,           # > 2048 tokens（估算）
    }

    for i, item in enumerate(data):
        convs = item.get("conversations", [])
        for conv in convs:
            if conv["from"] == "gpt":
                answer = conv["value"]
                if not answer.strip():
                    issues.append(f"[{i}] 空答案")
                    stats["empty_outputs"] += 1
                if len(answer) < 20:
                    issues.append(f"[{i}] 答案过短: {answer!r}")
                    stats["too_short_outputs"] += 1
                if len(answer) > 4000:   # 粗略估算 token 数
                    stats["too_long"] += 1

    print(f"数据集统计：")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if issues:
        print(f"\n发现 {len(issues)} 个问题：")
        for issue in issues[:10]:  # 只显示前 10 个
            print(f"  {issue}")
    return stats


# 使用 HuggingFace 公开数据集
from datasets import load_dataset

# 通用指令数据集（英文）
alpaca = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# 中文指令数据集
chinese_alpaca = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:500]")

# 代码数据集
code_data = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:500]")
```

### 5.5 数据增强技巧

```python
import anthropic

client = anthropic.Anthropic()

def augment_qa_pair(question: str, answer: str) -> list[dict]:
    """
    用 Claude 生成同一问题的多种表达方式
    数据增强：1条数据 → 3条数据
    """
    response = client.messages.create(
        model="claude-haiku-4-5",  # 用便宜的模型做数据增强
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""给定以下问答对，生成 2 个语义相同但表达不同的问题版本。
只返回 JSON 数组，不要其他内容。

原始问题：{question}
答案：{answer}

返回格式：["问题版本2", "问题版本3"]"""
        }]
    )

    variations = json.loads(response.content[0].text)
    result = [{"question": question, "answer": answer}]
    for var in variations:
        result.append({"question": var, "answer": answer})
    return result
```

---

> **💡 Claude Code 提示词 — 数据集准备：**
> ```
> 用 Python 创建一个完整的数据集准备脚本 prepare_dataset.py：
>
> 场景：为"支付系统 FAQ 助手"准备微调数据集
>
> 1. 生成原始数据（generate_raw_data()）：
>    手动定义 30 条支付领域 Q&A（中英文混合）：
>    - ISO 20022 错误码（AM04, AC01, DUPL, RC01 等）
>    - Finacle API 相关问题
>    - 国际汇款流程（TTP/IMT）
>    - 合规和安全问题
>    每条包含：question, answer, category, difficulty
>
> 2. 数据增强（augment_with_claude()）：
>    - 用 claude-haiku-4-5 为每条数据生成 2 个同义问题
>    - 最终生成约 90 条数据
>    - 加入错误处理和重试逻辑
>
> 3. 格式转换（to_sharegpt_format()）：
>    - 转成 ShareGPT conversations 格式
>    - 加入统一的 system prompt：
>      "你是 Westpac FTH 支付系统专家，专门解答支付处理相关问题。
>       回答要准确、简洁，关键术语用英文标注。"
>
> 4. 数据集分割：
>    - train: 80%（约 72 条）
>    - val: 20%（约 18 条）
>
> 5. 质量验证（validate_dataset()）：
>    - 检查空答案、过短答案、重复问题
>    - 统计平均长度、token 估算
>    - 打印数据集样本预览
>
> 6. 保存：
>    - data/train.json
>    - data/val.json
>    - data/dataset_stats.json（统计信息）
>
> 7. 上传到 HuggingFace（可选）：
>    - 用 datasets 库打包并 push_to_hub
> ```

---

## 6. 用 Unsloth 微调（核心）

### 6.1 完整微调脚本

```python
# finetune.py
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import json

# ─── Step 1: 配置 ──────────────────────────────────────────
MODEL_NAME   = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # 4bit量化版，省显存
MAX_SEQ_LEN  = 2048      # 最大序列长度（context window）
LOAD_IN_4BIT = True      # QLoRA: 4bit 加载基础模型

LORA_RANK    = 16        # LoRA rank（越大能力越强，但显存越多）
LORA_ALPHA   = 32        # LoRA 缩放因子（通常 = 2 × rank）
LORA_DROPOUT = 0         # Unsloth 建议设为 0

OUTPUT_DIR   = "./output/payment-expert"
DATASET_PATH = "./data/train.json"

# ─── Step 2: 加载模型 ─────────────────────────────────────
print("📦 加载基础模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,        # 自动选择（BF16 on Ampere+）
    load_in_4bit    = LOAD_IN_4BIT,
)

# ─── Step 3: 添加 LoRA Adapter ───────────────────────────
print("🔧 添加 LoRA Adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    target_modules = [              # 对哪些矩阵做 LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",      # Attention 层
        "gate_proj", "up_proj", "down_proj",           # FFN 层
    ],
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # 节省显存（Unsloth 专有优化）
    random_state   = 42,
    use_rslora     = False,         # Rank-Stabilized LoRA（高 rank 时用）
)

# 打印可训练参数数量
model.print_trainable_parameters()
# 输出示例：trainable params: 13,631,488 || all params: 3,110,313,984 || trainable%: 0.4382

# ─── Step 4: 准备数据集 ──────────────────────────────────
print("📚 准备数据集...")

# 设置 chat template（与基础模型一致）
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")  # 或 llama-3

# 加载数据
with open(DATASET_PATH) as f:
    raw_data = json.load(f)

from datasets import Dataset

def format_conversations(examples):
    """把 ShareGPT 格式转成模型 token"""
    texts = []
    for convs in examples["conversations"]:
        # 转成 HuggingFace messages 格式
        messages = []
        for conv in convs:
            role = "user" if conv["from"] == "human" else "assistant"
            messages.append({"role": role, "content": conv["value"]})

        # 用 tokenizer 的 chat template 格式化
        text = tokenizer.apply_chat_template(
            messages,
            tokenize       = False,
            add_generation_prompt = False,
        )
        texts.append(text)
    return {"text": texts}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_conversations, batched=True)

print(f"训练样本数：{len(dataset)}")
print(f"样本预览：\n{dataset[0]['text'][:300]}...")

# ─── Step 5: 配置训练参数 ─────────────────────────────────
print("⚙️  配置训练参数...")
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,    # 数据预处理并行数
    args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = 3,          # 训练轮数
        per_device_train_batch_size = 2,      # 每 GPU 批大小
        gradient_accumulation_steps = 4,      # 梯度累积（等效 batch=8）
        warmup_steps            = 10,         # 学习率预热步数
        learning_rate           = 2e-4,       # 学习率
        fp16                    = not torch.cuda.is_bf16_supported(),
        bf16                    = torch.cuda.is_bf16_supported(),
        logging_steps           = 10,         # 每 10 步打印一次 loss
        save_strategy           = "epoch",    # 每 epoch 保存
        optim                   = "adamw_8bit",  # 8bit Adam，省显存
        weight_decay            = 0.01,
        lr_scheduler_type       = "linear",   # 学习率衰减策略
        seed                    = 42,
        report_to               = "none",     # 改成 "wandb" 启用监控
    ),
)

# ─── Step 6: 开始训练 ─────────────────────────────────────
print("🚀 开始训练...")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"GPU: {gpu_stats.name}, VRAM: {gpu_stats.total_memory / 1024**3:.1f} GB")

trainer_stats = trainer.train()

# 训练完成后打印统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\n✅ 训练完成！")
print(f"   总时间：{trainer_stats.metrics['train_runtime']:.0f}s")
print(f"   最终 loss：{trainer_stats.metrics['train_loss']:.4f}")
print(f"   显存使用：{used_memory} GB")
```

### 6.2 快速推理验证（训练后立即测试）

```python
# 训练完成后立即测试（无需导出）
FastLanguageModel.for_inference(model)  # 切换到推理模式（2× 速度）

messages = [
    {"role": "user", "content": "什么是 AM04 错误码？应该如何处理？"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize             = True,
    add_generation_prompt = True,
    return_tensors       = "pt",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids        = inputs,
        max_new_tokens   = 512,
        temperature      = 0.7,
        top_p            = 0.9,
        do_sample        = True,
        pad_token_id     = tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"模型回答：\n{response}")
```

### 6.3 保存 LoRA Adapter

```python
# 保存方式 1：只保存 LoRA Adapter（~100MB，最小）
model.save_pretrained("./output/lora_adapter")
tokenizer.save_pretrained("./output/lora_adapter")
print("✅ LoRA Adapter 已保存（约 100MB）")

# 保存方式 2：合并 LoRA 到基础模型并保存（完整模型，~6GB）
model.save_pretrained_merged(
    "./output/merged_model",
    tokenizer,
    save_method = "merged_16bit",  # FP16 完整模型
)
print("✅ 合并模型已保存")

# 保存方式 3：直接保存为 GGUF（Ollama 使用）← 下节详解
model.save_pretrained_gguf(
    "./output/gguf_model",
    tokenizer,
    quantization_method = "q4_k_m",  # 4-bit 量化（推荐）
)
print("✅ GGUF 模型已保存，可直接导入 Ollama")
```

---

> **💡 Claude Code 提示词 — 完整微调脚本：**
> ```
> 创建完整的 LLM 微调项目，目录结构如下：
>
> finetune_project/
> ├── finetune.py          ← 主训练脚本
> ├── inference.py         ← 推理测试脚本
> ├── prepare_dataset.py   ← 数据准备脚本（见第5节）
> ├── config.yaml          ← 超参数配置文件
> └── data/
>     ├── train.json
>     └── val.json
>
> finetune.py 要求：
> - 从 config.yaml 读取所有配置（模型名、LoRA参数、训练参数）
> - 支持命令行参数覆盖配置（argparse）
> - 详细的训练日志（每步loss、显存使用、预计剩余时间）
> - 每 epoch 在 val 集上评估并保存最佳 checkpoint
> - 训练完成后自动运行 5 个测试问题并打印模型回答
> - 支持断点续训（从 checkpoint 恢复）
>
> config.yaml 包含：
>   model:
>     name: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
>     max_seq_length: 2048
>     load_in_4bit: true
>   lora:
>     rank: 16
>     alpha: 32
>     dropout: 0
>     target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
>   training:
>     epochs: 3
>     batch_size: 2
>     gradient_accumulation: 4
>     learning_rate: 2e-4
>     warmup_steps: 10
>
> inference.py 要求：
> - 交互式命令行聊天界面（类似 ollama run）
> - 支持多轮对话（维护历史）
> - 显示每次生成的 token 数和耗时
>
> 数据集：使用第5节生成的支付系统 FAQ 数据集
> 基础模型：Qwen2.5-3B-Instruct（适合 8GB VRAM）
> ```

---

## 7. 超参数调优指南

### 7.1 关键超参数说明

```
LoRA 参数：

rank (r)
  控制 LoRA 矩阵的"宽度"，决定模型可以学习的复杂度
  推荐值：16（通用）/ 32（复杂任务）/ 8（简单任务）
  注意：rank 翻倍 → 参数量翻倍，显存增加不多

alpha (lora_alpha)
  控制 LoRA 更新的缩放强度
  推荐值：= 2 × rank（经验法则）
  效果：alpha 越大，LoRA 的影响越强

target_modules
  对哪些权重矩阵做 LoRA
  最少：["q_proj", "v_proj"]（只对 Q/V 做，最省显存）
  推荐：加上 k_proj, o_proj, gate_proj, up_proj, down_proj
  最全：所有层（效果最好，但显存最多）
```

```
训练参数：

learning_rate
  推荐范围：1e-4 ~ 5e-4（LoRA 通常比 FFT 大 10×）
  太大 → loss 震荡，模型损坏
  太小 → 收敛慢，欠拟合

num_train_epochs
  小数据集（<1000条）：3-5 epoch
  大数据集（>10000条）：1-3 epoch
  注意过拟合：val_loss 上升时停止

batch_size + gradient_accumulation
  等效 batch = per_device_batch × gradient_accumulation
  推荐等效 batch = 8~32
  显存不足时：减小 per_device_batch，增大 gradient_accumulation
```

### 7.2 过拟合的信号与处理

```python
# 训练监控：检测过拟合
import json
import matplotlib.pyplot as plt

def plot_training_curve(log_file: str):
    """
    log_file: trainer_state.json（Unsloth 自动保存）
    """
    with open(log_file) as f:
        logs = json.load(f)["log_history"]

    train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]
    eval_loss  = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

    steps_t, loss_t = zip(*train_loss) if train_loss else ([], [])
    steps_e, loss_e = zip(*eval_loss)  if eval_loss  else ([], [])

    plt.figure(figsize=(10, 4))
    plt.plot(steps_t, loss_t, label="Train Loss")
    plt.plot(steps_e, loss_e, label="Val Loss", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("训练曲线（Val Loss 上升 = 过拟合）")
    plt.savefig("training_curve.png")
    print("训练曲线已保存为 training_curve.png")

# 过拟合处理方案：
# 1. 减少 epoch 数（最直接）
# 2. 增加数据量（根本解决）
# 3. 降低 learning_rate
# 4. 增加 lora_dropout（0.05~0.1）
# 5. 减小 rank
```

### 7.3 超参数搜索（简单版）

```python
# 快速超参数实验（用小数据集 + 少 epoch）
CONFIGS = [
    {"rank": 8,  "lr": 2e-4, "name": "rank8_lr2e4"},
    {"rank": 16, "lr": 2e-4, "name": "rank16_lr2e4"},
    {"rank": 16, "lr": 5e-4, "name": "rank16_lr5e4"},
    {"rank": 32, "lr": 1e-4, "name": "rank32_lr1e4"},
]

results = []
for cfg in CONFIGS:
    # 每个配置只训练 1 epoch 快速评估
    loss = train_with_config(cfg, epochs=1)
    results.append({**cfg, "val_loss": loss})
    print(f"{cfg['name']}: val_loss = {loss:.4f}")

# 选 val_loss 最低的配置进行完整训练
best = min(results, key=lambda x: x["val_loss"])
print(f"\n最优配置：{best}")
```

---

## 8. 导出 GGUF 并部署到 Ollama

### 8.1 GGUF 格式与量化

```
GGUF（GPT-Generated Unified Format）
= llama.cpp 使用的模型格式
= Ollama 使用的底层格式

量化级别选择：
  q2_k   → 2-bit，最小（~1.5GB），质量较差
  q4_k_m → 4-bit，推荐（~2.5GB），质量/大小均衡 ⭐
  q5_k_m → 5-bit，较好（~3.2GB），更高质量
  q8_0   → 8-bit，高质量（~5GB），接近 FP16
  f16    → 16-bit，无损（~6GB），最大文件

学习项目推荐：q4_k_m（默认）
```

### 8.2 用 Unsloth 直接导出 GGUF

```python
# 在训练脚本末尾，或单独运行
from unsloth import FastLanguageModel

# 重新加载训练好的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "./output/lora_adapter",  # LoRA adapter 路径
    max_seq_length  = 2048,
    dtype           = None,
    load_in_4bit    = True,
)
FastLanguageModel.for_inference(model)

# 导出为 GGUF（Unsloth 自动合并 LoRA + 量化）
print("📦 导出 GGUF 模型...")
model.save_pretrained_gguf(
    "payment-expert",      # 输出目录
    tokenizer,
    quantization_method = "q4_k_m",   # 推荐量化方式
)
# 输出文件：payment-expert/unsloth.Q4_K_M.gguf（约 2GB）
print("✅ GGUF 导出完成！")

# Unsloth 也会自动生成 Modelfile（Ollama 配置）
# payment-expert/Modelfile 内容类似：
# FROM ./unsloth.Q4_K_M.gguf
# TEMPLATE ...（自动使用正确的 chat template）
# PARAMETER stop "<|endoftext|>"
```

### 8.3 导入 Ollama 并运行

```bash
# 进入导出目录
cd payment-expert

# 查看 Unsloth 生成的 Modelfile
cat Modelfile

# 创建 Ollama 模型
ollama create payment-expert -f Modelfile

# 运行模型（交互模式）
ollama run payment-expert

# 测试：
# >>> 什么是 AM04 错误码？
# （模型根据微调数据回答）

# 查看已安装的模型
ollama list

# 用 API 调用
curl http://localhost:11434/api/chat -d '{
  "model": "payment-expert",
  "messages": [{"role":"user","content":"解释 ISO 20022 的 MsgId 字段"}]
}'
```

### 8.4 自定义 Modelfile

```dockerfile
# Modelfile（自定义版）
FROM ./unsloth.Q4_K_M.gguf

# 系统提示（全局行为设定）
SYSTEM """你是 Westpac FTH 支付系统专家。
专门解答以下领域的问题：
- ISO 20022 国际支付标准
- Finacle 核心银行系统 API
- IBM MQ 消息队列
- 支付错误码处理（AM04, AC01, DUPL 等）

回答要求：
- 准确、简洁
- 技术术语使用英文
- 如果不确定，请明确说明"""

# 模型参数
PARAMETER temperature 0.7     # 较低温度 = 更确定的回答
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096        # Context 窗口大小
PARAMETER num_predict 512     # 最大生成长度

# 停止词（根据模型的 chat template）
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

```bash
# 用自定义 Modelfile 创建模型
ollama create payment-expert-v2 -f Modelfile

# 运行
ollama run payment-expert-v2
```

---

> **💡 Claude Code 提示词 — 导出与部署：**
> ```
> 创建 export_and_deploy.py 脚本，完成以下工作：
>
> 1. 导出 GGUF：
>    - 从 ./output/lora_adapter 加载 LoRA adapter
>    - 用 Unsloth 导出 q4_k_m GGUF 到 ./deploy/payment-expert/
>    - 同时导出 q8_0 版本（对比质量用）
>
> 2. 生成自定义 Modelfile：
>    - 包含完整的 system prompt（支付系统专家角色）
>    - 配置合适的 temperature(0.7), num_ctx(4096)
>    - 自动检测模型类型并设置正确的 stop tokens
>
> 3. 自动注册到 Ollama：
>    - 用 subprocess 运行 ollama create
>    - 验证注册成功（ollama list）
>
> 4. 自动化测试（部署后）：
>    - 用 requests 调用 Ollama API
>    - 运行 5 个测试问题
>    - 对比微调前（基础模型）和微调后的回答
>    - 打印对比报告
>
> 5. 性能测试：
>    - 测量 TTFT（Time to First Token）
>    - 测量 tokens/second
>    - 对比 q4_k_m vs q8_0 的速度差异
>
> 创建 chat.py：
>    - 命令行聊天界面，连接本地 Ollama
>    - 支持多轮对话历史
>    - 支持 /clear 清除历史、/model 切换模型
>    - 显示每次生成的 token/s
> ```

---

## 9. 评估模型效果

### 9.1 评估维度

```
微调后的模型需要从三个维度评估：

1. 任务准确性
   模型的回答是否正确？
   评估方法：人工标注 + 自动评分

2. 格式遵循
   模型是否按要求的格式输出？
   评估方法：正则匹配 + JSON 解析成功率

3. 通用能力保留
   微调后有没有"遗忘"原有能力？
   评估方法：在通用 benchmark 上对比分数
```

### 9.2 自动评估脚本

```python
# evaluate.py
import anthropic
import json
from ollama import Client as OllamaClient

ollama = OllamaClient()
claude = anthropic.Anthropic()

def get_model_answer(model_name: str, question: str) -> str:
    """从 Ollama 获取模型回答"""
    response = ollama.chat(
        model   = model_name,
        messages = [{"role": "user", "content": question}]
    )
    return response["message"]["content"]

def score_with_claude(
    question: str,
    reference_answer: str,
    model_answer: str
) -> dict:
    """用 Claude 作为裁判评分"""
    response = claude.messages.create(
        model      = "claude-haiku-4-5",
        max_tokens = 200,
        messages   = [{
            "role": "user",
            "content": f"""评估以下模型回答的质量，只返回 JSON。

问题：{question}
参考答案：{reference_answer}
模型回答：{model_answer}

评分标准（每项0-10分）：
- accuracy: 事实准确性
- completeness: 内容完整性
- format: 格式清晰度

返回格式：{{"accuracy": N, "completeness": N, "format": N, "comment": "..."}}"""
        }]
    )
    return json.loads(response.content[0].text)

def run_evaluation(
    model_name: str,
    eval_dataset: list[dict]
) -> dict:
    """完整评估流程"""
    scores = []
    print(f"\n评估模型：{model_name}")
    print(f"测试集大小：{len(eval_dataset)}")

    for i, item in enumerate(eval_dataset):
        question  = item["question"]
        reference = item["answer"]

        # 获取模型回答
        model_answer = get_model_answer(model_name, question)

        # Claude 评分
        score = score_with_claude(question, reference, model_answer)
        scores.append(score)

        avg = sum(score[k] for k in ["accuracy","completeness","format"]) / 3
        print(f"[{i+1}/{len(eval_dataset)}] 平均分: {avg:.1f} - {score['comment'][:50]}")

    # 汇总
    avg_scores = {
        key: sum(s[key] for s in scores) / len(scores)
        for key in ["accuracy", "completeness", "format"]
    }
    avg_scores["overall"] = sum(avg_scores.values()) / 3

    print(f"\n{'='*40}")
    print(f"模型：{model_name}")
    print(f"准确性：  {avg_scores['accuracy']:.2f}/10")
    print(f"完整性：  {avg_scores['completeness']:.2f}/10")
    print(f"格式：    {avg_scores['format']:.2f}/10")
    print(f"总体：    {avg_scores['overall']:.2f}/10")

    return avg_scores


# 对比微调前后
if __name__ == "__main__":
    with open("data/val.json") as f:
        val_data = json.load(f)

    # 转成 {question, answer} 格式
    eval_set = []
    for item in val_data[:10]:  # 先评估前 10 条
        convs = item["conversations"]
        q = next(c["value"] for c in convs if c["from"] == "human")
        a = next(c["value"] for c in convs if c["from"] == "gpt")
        eval_set.append({"question": q, "answer": a})

    base_scores     = run_evaluation("qwen2.5:3b",       eval_set)
    finetune_scores = run_evaluation("payment-expert",   eval_set)

    print("\n📊 微调前后对比：")
    for key in ["accuracy", "completeness", "format", "overall"]:
        delta = finetune_scores[key] - base_scores[key]
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {key}: {base_scores[key]:.2f} → {finetune_scores[key]:.2f} {arrow}{abs(delta):.2f}")
```

---

> **💡 Claude Code 提示词 — 评估系统：**
> ```
> 创建完整的模型评估系统 evaluate.py：
>
> 1. 评估指标（3类）：
>    a. 自动指标（不需要 Claude）：
>       - ROUGE-L（词汇重叠度）
>       - 关键词命中率（检查回答中是否包含预期关键词）
>       - 格式验证（JSON/结构化输出是否合法）
>
>    b. LLM 裁判（用 claude-haiku-4-5 评分，便宜）：
>       - 准确性 0-10
>       - 完整性 0-10
>       - 相关性 0-10
>
>    c. 行为测试：
>       - 不相关问题拒绝率（问"天气"，应该说"我专注于支付领域"）
>       - 格式一致性（100个问题，回答格式是否统一）
>
> 2. 对比报告：
>    - 对比 3 个模型：基础模型 vs 微调模型 vs Claude-Haiku（参考线）
>    - 生成 HTML 报告（包含表格和图表）
>    - 保存到 evaluation_report.html
>
> 3. 错误分析：
>    - 找出分数最低的 5 个问题
>    - 分析失败原因（数据问题 or 模型问题）
>    - 生成改进建议
>
> 4. 命令行界面：
>    python evaluate.py --model payment-expert --dataset data/val.json --n 20
> ```

---

## 10. 完整学习项目：支付领域专家模型

### 项目目标

```
目标：微调一个本地 LLM，使其成为支付系统 FAQ 助手

评判标准：
  - 能准确解释 ISO 20022 错误码
  - 能描述 Finacle API 的使用场景
  - 回答格式统一（结构化，有条理）
  - 对不相关问题礼貌拒绝

技术选型：
  基础模型：Qwen2.5-3B-Instruct（6GB VRAM，QLoRA）
  微调框架：Unsloth + TRL
  部署：Ollama
  评估：Claude-Haiku-4-5 作为裁判
```

### 项目结构

```
payment-llm-finetune/
├── README.md
├── config.yaml               ← 所有配置集中管理
├── requirements.txt
│
├── data/
│   ├── raw_qa.json           ← 手写的原始 Q&A（30条）
│   ├── train.json            ← 训练集（~72条，含增强）
│   └── val.json              ← 验证集（~18条）
│
├── scripts/
│   ├── prepare_dataset.py    ← 数据准备 + 增强
│   ├── finetune.py           ← 训练脚本
│   ├── export_gguf.py        ← 导出 GGUF + 注册 Ollama
│   ├── evaluate.py           ← 评估对比
│   └── chat.py               ← 交互式聊天
│
├── output/
│   ├── lora_adapter/         ← LoRA 权重
│   └── gguf/                 ← GGUF 文件
│
└── notebooks/
    └── quick_start.ipynb     ← Google Colab 快速上手版
```

### 端到端运行步骤

```bash
# 克隆项目
git clone <your-repo>
cd payment-llm-finetune

# 安装依赖
pip install -r requirements.txt

# Step 1: 准备数据（自动增强到 ~90 条）
python scripts/prepare_dataset.py

# Step 2: 微调（约 10-30 分钟，取决于 GPU）
python scripts/finetune.py

# Step 3: 导出并部署
python scripts/export_gguf.py

# Step 4: 评估效果
python scripts/evaluate.py --n 10

# Step 5: 开始聊天！
python scripts/chat.py
# 或者：
ollama run payment-expert
```

### 学习延伸方向

```
完成基础项目后，可以尝试：

1. 更复杂的数据
   → 加入多轮对话（模拟客服场景）
   → 加入工具调用格式（让模型输出 JSON 调用 API）

2. 更好的训练技术
   → DPO（Direct Preference Optimization）
     给模型正确 vs 错误回答对，学习偏好
   → GRPO（Group Relative Policy Optimization）
     强化学习式微调，提升推理能力

3. 持续学习
   → 每周加入新的 Q&A，定期重新微调
   → 避免灾难性遗忘（混入通用数据）

4. 评估体系升级
   → 接入真实用户测试
   → A/B 测试（基础模型 vs 微调模型）
```

---

> **💡 Claude Code 综合提示词 — 完整项目生成：**
> ```
> 创建完整的 payment-llm-finetune 项目，包含所有文件：
>
> 1. README.md：
>    - 项目介绍、快速开始、目录说明
>    - Google Colab 一键运行链接
>
> 2. requirements.txt：
>    - unsloth, trl, peft, datasets, transformers
>    - anthropic（用于数据增强和评估）
>    - ollama（Python 客户端）
>    - matplotlib, rouge-score（评估用）
>
> 3. config.yaml：
>    集中管理所有配置（模型、LoRA、训练、部署参数）
>
> 4. data/raw_qa.json：
>    手写 25 条支付系统 Q&A（中英混合），覆盖：
>    - ISO 20022 错误码（AM04/AC01/DUPL/RC01/FF01）
>    - 国际汇款流程（TTP/IMT 区别）
>    - Finacle API 调用（initiateTransaction, queryStatus）
>    - IBM MQ 消息处理
>    - 幂等性和重试逻辑
>
> 5. scripts/prepare_dataset.py：
>    - 读取 raw_qa.json
>    - 用 claude-haiku-4-5 增强到 ~75 条
>    - 转成 ShareGPT 格式（带统一 system prompt）
>    - 8:2 分割 → train.json / val.json
>    - 打印数据集统计和样本预览
>
> 6. scripts/finetune.py：
>    - 加载 config.yaml
>    - Unsloth + QLoRA 训练
>    - 实时打印 loss（带进度条）
>    - 每 epoch 在 val 集评估
>    - 训练后自动测试 5 个问题
>    - 保存 LoRA adapter
>
> 7. scripts/export_gguf.py：
>    - 导出 q4_k_m GGUF
>    - 生成自定义 Modelfile（带完整 system prompt）
>    - 自动注册到 Ollama
>    - 验证部署成功
>
> 8. scripts/evaluate.py：
>    - 对比基础模型 vs 微调模型（各 10 道题）
>    - Claude-Haiku 裁判打分
>    - 关键词命中率计算
>    - 生成对比报告（打印表格）
>
> 9. scripts/chat.py：
>    - 命令行多轮对话
>    - /clear /model /help 命令
>    - 显示每次的 tokens/s
>
> 10. notebooks/quick_start.ipynb：
>     Google Colab 版本（单文件，包含安装+数据+训练+测试）
>     适合没有本地 GPU 的用户
>
> 确保所有脚本都有清晰的注释和错误处理
> ```

---

## 附录：关键术语速查

| 术语 | 含义 |
|------|------|
| **FFT** | Full Fine-Tuning，全量微调，更新所有参数 |
| **LoRA** | Low-Rank Adaptation，只训练少量新增矩阵 |
| **QLoRA** | Quantized LoRA，4-bit 量化基础模型 + LoRA |
| **PEFT** | Parameter-Efficient Fine-Tuning，参数高效微调的统称 |
| **rank (r)** | LoRA 矩阵的秩，控制可学习参数量 |
| **alpha** | LoRA 缩放因子，通常 = 2×rank |
| **SFT** | Supervised Fine-Tuning，监督微调（用人工标注数据）|
| **DPO** | Direct Preference Optimization，偏好优化 |
| **GGUF** | llama.cpp / Ollama 使用的模型格式 |
| **adapter** | LoRA 训练产出的小文件（~100MB）|
| **Unsloth** | 高效微调框架，比 HF 快 2×，省 70% 显存 |
| **ShareGPT** | 常用的多轮对话数据集格式 |
| **chat template** | 模型专用的对话格式（如 Qwen 的 `<|im_start|>`）|
| **过拟合** | 模型记住训练数据，泛化能力下降（val_loss 上升）|
| **灾难性遗忘** | 微调后丢失原有通用能力 |
| **q4_k_m** | 4-bit GGUF 量化方式，推荐的平衡选项 |
| **warmup_steps** | 学习率从 0 逐渐升至设定值的步数 |
| **gradient_accumulation** | 多步累积梯度，等效增大 batch size |
