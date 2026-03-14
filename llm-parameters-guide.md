# LLM 关键参数完全参考手册

> 覆盖：推理参数 · 量化格式 · Embedding 参数 · RAG 检索参数 · 微调参数 · 部署参数
>
> 每个参数说明含义、取值范围、应用场景和推荐值。

---

## 目录

1. [推理控制参数（Inference Parameters）](#1-推理控制参数inference-parameters)
2. [量化格式（Quantization Formats）](#2-量化格式quantization-formats)
3. [上下文与内存参数（Context & Memory Parameters）](#3-上下文与内存参数context--memory-parameters)
4. [Embedding 参数](#4-embedding-参数)
5. [RAG 检索参数（Retrieval Parameters）](#5-rag-检索参数retrieval-parameters)
6. [微调参数（Fine-tuning Parameters）](#6-微调参数fine-tuning-parameters)
7. [部署性能参数（Deployment Parameters）](#7-部署性能参数deployment-parameters)
8. [参数速查表](#8-参数速查表)

---

## 1. 推理控制参数（Inference Parameters）

这些参数控制模型生成文本时的行为，是最常接触的一类参数。

---

### `temperature`（温度）

**含义：** 控制输出的随机性（Randomness）。本质是对模型输出的 logits（原始概率分布）做缩放——temperature 越高，概率分布越"平"，越难预测的 token 也有机会被选中；越低，分布越"尖"，模型倾向于选最高概率的词。

```
temperature = 0.0  → 完全确定性（Deterministic），每次输出相同
temperature = 0.5  → 较保守，偶尔有变化
temperature = 1.0  → 原始概率分布，标准随机性
temperature = 2.0  → 高随机性，可能胡言乱语
```

**取值范围：** `0.0 ~ 2.0`（部分模型支持更高）

| 场景 | 推荐值 | 原因 |
|------|--------|------|
| RAG 问答（事实性任务） | `0.1` | 需要准确，不要发挥 |
| 代码生成（Code Generation） | `0.2` | 语法有规律，略有弹性 |
| 数据提取、分类 | `0.0` | 必须一致，不容随机 |
| 聊天助手（Chatbot） | `0.7` | 自然感，但不失控 |
| 创意写作（Creative Writing） | `1.0 ~ 1.2` | 鼓励多样化表达 |
| 头脑风暴（Brainstorming） | `1.3 ~ 1.5` | 需要意想不到的想法 |

```python
# Ollama 设置
llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# OpenAI SDK 设置
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    messages=[...]
)
```

---

### `top_p`（核采样，Nucleus Sampling）

**含义：** 只从累计概率达到 `top_p` 的 token 集合中采样。例如 `top_p=0.9` 表示只考虑累计概率前 90% 的 token，排除长尾低概率词。

**取值范围：** `0.0 ~ 1.0`

```
top_p = 1.0  → 不过滤，考虑所有 token
top_p = 0.9  → 过滤掉最后 10% 的低概率 token（推荐默认）
top_p = 0.5  → 只考虑最高概率的那一半，输出更保守
```

**与 temperature 的关系：**
- 两者都影响随机性，通常只调一个
- 常见搭配：`temperature=0.7, top_p=0.9`（两者同时用）
- OpenAI 官方建议：调 temperature 时把 top_p 固定为 1，反之亦然

| 场景 | 推荐值 |
|------|--------|
| 事实性问答 | `0.9`（配合低 temperature） |
| 创意生成 | `0.95` |
| 代码生成 | `0.95` |
| 严格分类任务 | `1.0`（依靠 temperature=0 控制） |

---

### `top_k`（Top-K 采样）

**含义：** 每次只从概率最高的 K 个 token 中采样，直接裁掉其余所有选项。

**取值范围：** `1 ~ 词表大小`（通常几万）

```
top_k = 1   → 贪心解码（Greedy Decoding），每次选最高概率词
top_k = 40  → 从最高的40个候选中采样（Ollama 默认）
top_k = 100 → 候选池更大，更有创意空间
```

| 场景 | 推荐值 |
|------|--------|
| 精确任务（分类、提取） | `1 ~ 10` |
| 日常对话 | `40`（默认值通常够用） |
| 创意写作 | `80 ~ 100` |

> **实践建议：** top_k 和 top_p 经常一起用，形成双重过滤。先用 top_k 排除极低概率词，再用 top_p 做最终的核采样。

---

### `repeat_penalty`（重复惩罚，Repetition Penalty）

**含义：** 对已经出现过的 token 降低其被再次选中的概率，防止模型进入重复循环（"我我我我我……"）。

**取值范围：** `1.0 ~ 1.5`（1.0 表示不惩罚）

```
repeat_penalty = 1.0  → 不惩罚，可能产生重复
repeat_penalty = 1.1  → 轻微惩罚（推荐默认）
repeat_penalty = 1.3  → 强惩罚，几乎不重复，但可能变得跳跃
repeat_penalty = 1.5  → 过强，输出可能变得不连贯
```

| 场景 | 推荐值 |
|------|--------|
| 长文生成（易重复） | `1.1 ~ 1.15` |
| 结构化输出（JSON/代码） | `1.0`（不惩罚，结构词本来就要重复） |
| 对话（短回复） | `1.0 ~ 1.05` |

---

### `max_tokens` / `num_predict`（最大输出长度）

**含义：** 模型最多生成多少个 token。超过后强制截断。

**Ollama 中叫 `num_predict`，OpenAI 中叫 `max_tokens`。**

```
max_tokens = -1      → 不限制（Ollama 默认，让模型自然结束）
max_tokens = 256     → 短回复，适合 API 场景（节省成本）
max_tokens = 1024    → 中等长度
max_tokens = 4096    → 长文章、详细分析
```

| 场景 | 推荐值 |
|------|--------|
| 分类/提取（单词级输出） | `10 ~ 50` |
| 问答（段落级） | `256 ~ 512` |
| 文章生成 | `1024 ~ 4096` |
| 代码生成 | `2048`（代码可能较长） |

> ⚠️ **成本提示：** 云端 API 按 token 计费，输出 token 通常比输入贵 2-3 倍。生产环境一定要设置合理的 max_tokens 上限。

---

### `stop`（停止词，Stop Sequences）

**含义：** 当模型生成这些字符串时立即停止，不等待 max_tokens 触发。

```python
# 常见用法：强制 JSON 结构输出后停止
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "生成一个 JSON"}],
    stop=["```", "\n\n\n"]  # 遇到代码块结束或三个换行就停
)

# Ollama
llm = ChatOllama(model="qwen3:8b", stop=["<|end|>", "Human:"])
```

| 场景 | 推荐 stop |
|------|-----------|
| 多轮对话（防止模型自问自答） | `["Human:", "User:"]` |
| JSON 输出 | `["}"]`（最后一个花括号） |
| 列表生成（限制条数） | `["\n6."]`（停在第6条之前） |

---

### `seed`（随机种子，Random Seed）

**含义：** 固定随机数生成器的初始值，使输出可复现（Reproducible）。

```python
# 相同 seed + 相同 prompt → 相同输出（可复现）
llm = ChatOllama(model="qwen3:8b", seed=42, temperature=0.7)
```

| 场景 | 推荐 |
|------|------|
| 生产环境调试 | 固定 seed，方便定位问题 |
| A/B 测试（对比参数效果） | 固定 seed，排除随机干扰 |
| 创意生成 | 不设置（让随机性发挥） |

---

### `presence_penalty` / `frequency_penalty`（OpenAI 特有）

**含义（OpenAI API）：**
- `presence_penalty`：对已经出现过（不管几次）的 token 施加惩罚，鼓励引入新话题
- `frequency_penalty`：按出现**频次**施加惩罚，出现越多惩罚越重

**取值范围：** `-2.0 ~ 2.0`（负值表示鼓励重复）

```python
response = client.chat.completions.create(
    model="gpt-4o",
    presence_penalty=0.6,   # 鼓励引入新概念
    frequency_penalty=0.3,  # 轻度避免重复用词
    messages=[...]
)
```

| 场景 | presence_penalty | frequency_penalty |
|------|-----------------|-------------------|
| 创意写作 | `0.5 ~ 1.0` | `0.3 ~ 0.5` |
| 技术文档（用词精确） | `0` | `0` |
| 长文章（避免啰嗦） | `0.3` | `0.5` |

---

## 2. 量化格式（Quantization Formats）

量化（Quantization）是把模型权重从高精度浮点数压缩为低精度整数，大幅减少内存占用，代价是略微损失精度。

---

### 精度格式基础

```
FP32（32-bit Float）：
  每个参数占 4 字节，最高精度，训练时使用
  70B 模型 ≈ 280 GB 内存

FP16 / BF16（16-bit Float）：
  每个参数占 2 字节，精度损失极小
  70B 模型 ≈ 140 GB 内存

INT8（8-bit Integer）：
  每个参数占 1 字节，轻微精度损失
  70B 模型 ≈ 70 GB 内存

INT4（4-bit Integer）：
  每个参数占 0.5 字节，明显精度损失但可接受
  70B 模型 ≈ 35 GB 内存
```

---

### GGUF 量化命名规则（Ollama / llama.cpp 使用）

格式：`Q{位数}_{方法}_{变体}`

```
Q4_K_M
│  │ └── 变体（Variant）：S=Small, M=Medium, L=Large
│  └──── 方法（Method）：K=K-Quant（分组量化）
└─────── 位数：4-bit
```

#### 完整量化等级对比

| 格式 | 位数 | 内存占用（7B模型） | 质量损失 | 速度 | 推荐度 |
|------|------|-----------------|---------|------|--------|
| `F16` | 16-bit | ~14 GB | 无 | 慢 | 高配GPU基准 |
| `Q8_0` | 8-bit | ~7 GB | 极小 | 中 | 内存充裕首选 |
| `Q6_K` | 6-bit | ~5.5 GB | 很小 | 中快 | 质量与体积平衡 |
| **`Q5_K_M`** | 5-bit | ~5 GB | 小 | 快 | ⭐ 质量优先推荐 |
| **`Q4_K_M`** | 4-bit | ~4 GB | 中等 | 很快 | ⭐ 最佳综合选择 |
| `Q4_K_S` | 4-bit | ~3.8 GB | 中等偏高 | 很快 | 内存紧张备选 |
| `Q3_K_M` | 3-bit | ~3 GB | 明显 | 极快 | 低配机器 |
| `Q2_K` | 2-bit | ~2.5 GB | 严重 | 极快 | 不推荐（除非极端内存限制） |

#### 各格式详解

**`Q4_K_M`（最常用，默认推荐）**

```bash
# Ollama 拉取（默认即 Q4_K_M）
ollama pull qwen3:8b

# 明确指定
ollama pull qwen3:8b-instruct-q4_K_M
```

- 4-bit K-Quant Medium
- 7B 模型约占 4-5 GB 内存，RTX 3060（12GB）可跑 13B
- 质量损失约 1-3%，日常使用完全感知不到
- **适用场景：** 本地开发默认选择，RAG 问答，日常对话

**`Q5_K_M`（质量优先）**

- 5-bit，内存比 Q4_K_M 多约 20%
- 质量损失极小，接近 F16
- **适用场景：** 对质量敏感的生产任务（代码生成、技术文档），内存还有余量时

**`Q8_0`（高质量本地部署）**

- 8-bit，内存是 Q4_K_M 的约 1.75 倍
- 质量几乎与 F16 无差
- **适用场景：** 有足够 VRAM（24GB+），追求最高本地质量

**`Q4_0` vs `Q4_K_M`（重要区别）**

```
Q4_0：最早的 4-bit 格式，简单均匀量化，精度较差
Q4_K_M：K-Quant 分组量化，对"重要"层保留更高精度

→ 永远选 Q4_K_M 而不是 Q4_0
```

---

### 硬件与量化格式对照

| 硬件配置 | 适合运行的模型规模 | 推荐量化 |
|---------|-----------------|---------|
| 8GB RAM（CPU） | 3B 模型 | `Q4_K_M` |
| 16GB RAM（CPU / M系列Mac） | 7B 模型 | `Q4_K_M` ~ `Q5_K_M` |
| 16GB VRAM（RTX 4080） | 13B 模型 | `Q4_K_M` |
| 24GB VRAM（RTX 4090） | 34B 模型 | `Q4_K_M`，或 13B `Q8_0` |
| 48GB VRAM（双卡 3090） | 70B 模型 | `Q4_K_M` |
| 80GB VRAM（A100） | 70B `F16` | `F16` 或 `Q8_0` |

```bash
# 查看 Ollama 中模型的量化信息
ollama show qwen3:8b --modelinfo | grep quantization
```

---

### AWQ / GPTQ（GPU 专属量化格式）

主要用于 GPU 部署框架（vLLM、HuggingFace Transformers），不在 Ollama 中使用。

| 格式 | 特点 | 适用框架 |
|------|------|---------|
| **AWQ**（Activation-aware Weight Quantization） | 考虑激活值分布，精度更好 | vLLM、HF Transformers |
| **GPTQ**（Generative Pre-trained Transformer Quantization） | 最早成熟的 GPU 量化方案 | AutoGPTQ、vLLM |
| **BitsAndBytes（NF4）** | Unsloth/QLoRA 微调专用 | Transformers + PEFT |

```python
# 使用 BitsAndBytes 4-bit 加载（微调场景）
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NF4（Normal Float 4）量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 BF16
    bnb_4bit_use_double_quant=True,   # 双重量化，进一步节省内存
)
```

---

## 3. 上下文与内存参数（Context & Memory Parameters）

---

### `num_ctx`（上下文窗口大小，Context Window Size）

**含义：** 模型一次能"看到"的最大 token 数。包括输入的 Prompt + 检索到的文档 + 对话历史 + 输出。

**这是 Ollama 中最容易被忽视、最关键的参数。**

```
Ollama 默认 num_ctx = 2048（极小！）
Qwen3:8B 原生支持 = 128,000 tokens
→ 如果不手动设置，大量内容会被截断！
```

**取值范围：** `512 ~ 模型最大支持值`

| 场景 | 推荐 num_ctx | 原因 |
|------|------------|------|
| 简单对话（短问短答） | `4096` | 够用，节省内存 |
| RAG 问答（检索 5 个块） | `8192` | 5×512 tokens + prompt + 回答 |
| RAG 问答（检索 10 个块） | `16384` | 10×1024 tokens + 余量 |
| 长文档分析（整篇文章） | `32768 ~ 65536` | 长文本分析 |
| 长对话历史 | `16384` | 保留更多轮次 |

```python
# Ollama 设置
llm = ChatOllama(
    model="qwen3:8b",
    num_ctx=8192,     # ⚠️ RAG 场景必须设置！
)

# Modelfile 永久设置
# PARAMETER num_ctx 8192
```

**计算公式（RAG 场景）：**
```
所需 num_ctx ≈ (chunk_size × top_k) + prompt_tokens + max_output_tokens + 余量
示例：(512 × 5) + 500 + 512 + 500 = 4,072 → 设置 8192 保险
```

---

### `num_gpu` / `gpu_layers`（GPU 层数）

**含义：** 将模型的多少层加载到 GPU 显存中。层数越多，GPU 参与计算越多，速度越快；不足时溢出到 CPU/RAM（速度大幅下降）。

```bash
# Ollama 自动检测 GPU，通常无需手动设置
# 手动强制全部放 GPU
OLLAMA_NUM_GPU=999 ollama serve

# llama.cpp 手动设置
llama-cli -m model.gguf --n-gpu-layers 35   # 35层放GPU，其余CPU
```

| 场景 | 建议 |
|------|------|
| GPU 显存足够 | 设 `-1` 或 `999`（全部放 GPU） |
| 显存不足（混合推理） | 根据显存大小设合适层数 |
| 纯 CPU 推理 | 设 `0` |

---

### `num_thread`（CPU 线程数）

**含义：** CPU 推理时使用的线程数，影响纯 CPU 推理速度。

```bash
# Ollama 通常自动设置为 CPU 核心数
# 手动设置
llm = ChatOllama(model="qwen3:8b", num_thread=8)
```

**推荐：** 设为物理核心数（不是超线程数），过多线程反而因争抢内存带宽变慢。

---

### `num_batch`（批处理大小，Batch Size）

**含义（推理阶段）：** 预填充阶段（Prefill Phase）一次处理多少 token。更大的 batch 在有 GPU 时更快。

```python
llm = ChatOllama(model="qwen3:8b", num_batch=512)
```

| 场景 | 推荐值 |
|------|--------|
| GPU 推理（有余量） | `512 ~ 2048` |
| CPU 推理 | `128 ~ 256` |
| 内存紧张 | `64 ~ 128` |

---

## 4. Embedding 参数

---

### `embedding_dim`（嵌入维度，Embedding Dimensions）

**含义：** 每个文本被转换成的向量维度数。维度越高，表达能力越强，但存储和计算成本也越高。

```
nomic-embed-text：768 维
BGE-M3：1024 维
OpenAI text-embedding-3-large：3072 维（默认）
Qwen3-Embedding-8B：4096 维
```

**Matryoshka Representation Learning（MRL，套娃表示学习）：**

部分现代模型支持降维截断，不损失太多质量：

```python
# OpenAI 支持指定更小维度
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="你的文本",
    dimensions=512   # 从 3072 降到 512，节省存储
)
```

| 维度大小 | 存储成本（100万向量） | 检索速度 | 精度 |
|---------|-------------------|---------|------|
| 256 | ~1 GB | 极快 | 下降较多 |
| 512 | ~2 GB | 很快 | 轻微下降 |
| 768 | ~3 GB | 快 | 良好 |
| 1024 | ~4 GB | 中 | 很好 |
| 3072 | ~12 GB | 慢 | 最好 |

---

### `max_length` / `max_seq_length`（最大输入长度）

**含义：** Embedding 模型一次能处理的最大 token 数。超出会截断，导致信息丢失。

```
all-MiniLM-L6-v2：256 tokens（⚠️ 极短！）
BGE-base-en-v1.5：512 tokens
nomic-embed-text：8192 tokens
BGE-M3：8192 tokens
Jina v3：8192 tokens
Cohere embed-v4：128,000 tokens
```

**实践影响：** chunk_size 不能超过 embedding 模型的 max_length！

```python
# ❌ 错误：chunk_size 超过模型最大长度
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
embeddings = OllamaEmbeddings(model="all-MiniLM-L6-v2")  # 最大256 token！
# → 每个 chunk 会被截断，丢失后半部分

# ✅ 正确：chunk_size 与模型匹配
splitter = RecursiveCharacterTextSplitter(chunk_size=512)
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 支持 8192
```

---

### `normalize_embeddings`（向量归一化）

**含义：** 将向量归一化为单位长度（L2 Norm = 1），使余弦相似度（Cosine Similarity）等价于点积（Dot Product），加速计算。

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(texts, normalize_embeddings=True)  # 推荐开启
```

**几乎所有 RAG 场景都应开启。** 不开启时，向量长度不一，距离计算不准确。

---

### `batch_size`（Embedding 批处理大小）

**含义：** 一次送入 embedding 模型多少个文本片段并行处理。

```python
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(
    texts,
    batch_size=32,          # GPU 内存允许时可增大
    show_progress_bar=True
)
```

| 硬件 | 推荐 batch_size |
|------|---------------|
| CPU 推理 | `8 ~ 16` |
| GPU（8GB VRAM） | `32 ~ 64` |
| GPU（24GB VRAM） | `128 ~ 256` |

---

## 5. RAG 检索参数（Retrieval Parameters）

---

### `k`（检索数量，Top-K Retrieval）

**含义：** 向量相似度检索返回最相关的 k 个文档块。

**取值范围：** `1 ~ 20`（实践中 3-10 最常用）

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}   # 返回最相似的5个块
)
```

| 场景 | 推荐 k | 原因 |
|------|--------|------|
| 简单事实问答 | `3` | 问题聚焦，少量块就够 |
| 复杂多方面问题 | `5 ~ 8` | 需要多个角度的信息 |
| 长文档分析 | `8 ~ 10` | 答案分散在多处 |
| 用了 Reranker | `20`（粗检索）→ Reranker → Top-5 | 先广撒网再精排 |

**k 太小：** 召回率（Recall）低，漏掉关键信息  
**k 太大：** 精确率（Precision）低，填入大量无关内容，增加 LLM 噪音，浪费 token

---

### `score_threshold`（相似度阈值，Similarity Threshold）

**含义：** 只返回相似度分数高于该阈值的块，低于阈值的即使排名靠前也丢弃。

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.7   # 只要相似度 > 0.7 的块
    }
)
```

**取值范围：** `0.0 ~ 1.0`（余弦相似度）

| 阈值 | 含义 | 适用 |
|------|------|------|
| `0.5` | 宽松，召回率高 | 领域知识分散，宁可多 |
| `0.7` | 适中（推荐默认） | 一般 RAG 问答 |
| `0.8` | 严格，精确率高 | 精确知识库，要求准确 |
| `0.9` | 极严格 | 几乎只返回完全相关的块 |

---

### `search_type`（检索类型）

**含义：** 向量数据库的检索算法。

```python
# 1. 纯向量相似度（Cosine Similarity）
retriever = vectorstore.as_retriever(search_type="similarity")

# 2. MMR（Maximal Marginal Relevance，最大边际相关）
#    在相关性和多样性之间取平衡，避免检索到重复内容
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,       # 先检索20个，再从中选5个最多样的
        "lambda_mult": 0.5   # 0=最多样，1=最相关（默认0.5）
    }
)

# 3. 带阈值过滤
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)
```

| 检索类型 | 适用场景 |
|---------|---------|
| `similarity` | 默认，大多数场景 |
| `mmr` | 文档内容有大量重复（如政策文件多个相似段落） |
| `similarity_score_threshold` | 对质量要求严格，宁缺毋滥 |

---

### `fetch_k`（MMR 预检索数量）

**含义：** 使用 MMR 时，先从向量库检索 fetch_k 个候选，再从中选出 k 个最多样的。

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 30}
)
```

**规则：** `fetch_k` 必须 ≥ `k`，一般设为 `k` 的 4-6 倍。

---

### `lambda_mult`（MMR 多样性权重）

**含义：** 控制 MMR 中相关性与多样性的权重平衡。

```
lambda_mult = 1.0  → 纯相关性（等同于普通 similarity 检索）
lambda_mult = 0.5  → 相关性与多样性各半（推荐）
lambda_mult = 0.0  → 纯多样性（可能返回不相关但多样的内容）
```

---

### `chunk_size` + `chunk_overlap`（RAG 入库参数）

详见第2章，这里仅列应用场景速查：

| 文档类型 | chunk_size | chunk_overlap |
|---------|-----------|--------------|
| FAQ / 产品描述 | `256` | `20` |
| 技术文档 / 规范 | `512` | `50` |
| 法律合同 / 研究论文 | `1024` | `100` |
| 代码文件 | `500`（按函数切） | `0` |

---

## 6. 微调参数（Fine-tuning Parameters）

---

### `r`（LoRA 秩，LoRA Rank）

**含义：** LoRA（Low-Rank Adaptation）中低秩矩阵的维度，决定可训练参数量。r 越大，模型学习能力越强，但内存和训练时间也越多。

```
参数量 ≈ 2 × r × hidden_dim × num_layers

r=4：参数最少，约占原始模型 0.1%
r=8：常用轻量级微调（推荐起点）
r=16：标准推荐，平衡质量与效率
r=64：强学习能力，接近全参数微调效果
r=128：极大，通常不必要
```

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                          # LoRA 秩
    lora_alpha=32,                 # 缩放因子（通常 = 2×r）
    target_modules=["q_proj", "v_proj"],  # 注入的模块
    lora_dropout=0.05,             # Dropout 防过拟合
    bias="none",
)
```

| 场景 | 推荐 r | 原因 |
|------|--------|------|
| 风格调整（语气、格式） | `4 ~ 8` | 任务简单，小 r 够用 |
| 领域适配（专业术语） | `8 ~ 16` | 需要学新知识 |
| 任务专项（分类、提取） | `16 ~ 32` | 行为改变较大 |
| 复杂推理能力提升 | `32 ~ 64` | 需要强学习能力 |

---

### `lora_alpha`（LoRA 缩放因子，Scaling Factor）

**含义：** 控制 LoRA 矩阵的缩放比例，影响微调的"力度"。实际缩放比 = `lora_alpha / r`。

```
常见设置：lora_alpha = 2 × r

r=8,  lora_alpha=16  → 缩放比 = 2
r=16, lora_alpha=32  → 缩放比 = 2（推荐）
r=16, lora_alpha=16  → 缩放比 = 1（保守微调）
r=16, lora_alpha=64  → 缩放比 = 4（强微调）
```

---

### `lora_dropout`（LoRA Dropout）

**含义：** 训练时随机丢弃 LoRA 矩阵中的部分连接，防止过拟合（Overfitting）。

**取值范围：** `0.0 ~ 0.1`

```python
lora_dropout=0.0   # 数据量大时可设为0
lora_dropout=0.05  # 推荐默认
lora_dropout=0.1   # 数据量小、容易过拟合时
```

---

### `target_modules`（目标模块）

**含义：** 指定在哪些模块上注入 LoRA 适配器。注入越多模块，学习能力越强，但参数越多。

```python
# 最小配置（只注入 attention 的 Q 和 V）
target_modules=["q_proj", "v_proj"]

# 标准配置（Unsloth 推荐，注入全部 attention）
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# 最大配置（attention + MLP，几乎全模型）
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

| 配置 | 参数量 | 适用 |
|------|--------|------|
| Q+V only | 最少 | 简单任务，资源紧张 |
| All attention | 中等 | 推荐默认 |
| Attention + MLP | 最多 | 复杂任务，充足资源 |

---

### `learning_rate`（学习率，Learning Rate）

**含义：** 每步梯度更新的步长大小。太大会发散，太小会训练极慢或陷入局部最优。

**取值范围（LoRA 微调）：** `1e-5 ~ 3e-4`

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    learning_rate=2e-4,          # 推荐起点
    lr_scheduler_type="cosine",  # 余弦退火（Cosine Annealing）
    warmup_ratio=0.03,           # 前3%的步数做warmup
)
```

| 场景 | 推荐学习率 |
|------|-----------|
| 指令微调（SFT） | `2e-4` |
| 偏好对齐（DPO） | `5e-5 ~ 1e-4` |
| 持续预训练（Continued Pretraining） | `1e-5 ~ 5e-5` |
| 学习率太高的表现 | 训练 loss 震荡、不收敛 |
| 学习率太低的表现 | 训练 loss 下降极慢，最终效果差 |

---

### `num_train_epochs`（训练轮数，Training Epochs）

**含义：** 将整个训练数据集过几遍。

```
1 epoch：每条数据见一次
3 epochs：每条数据见三次（推荐默认）
5 epochs：较多，需监控过拟合
```

| 数据集大小 | 推荐 epochs |
|-----------|------------|
| <1,000 条 | `1 ~ 2`（小数据易过拟合） |
| 1,000 ~ 10,000 条 | `2 ~ 3` |
| >10,000 条 | `1 ~ 2`（数据够多不需要多轮） |

---

### `per_device_train_batch_size`（每设备训练批大小）

**含义：** 每张 GPU 每次处理多少条训练数据。

```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,   # 单卡批大小
    gradient_accumulation_steps=4,  # 梯度累积步数
    # 等效 batch_size = 2 × 4 = 8
)
```

**有效批大小（Effective Batch Size） = per_device_batch_size × gradient_accumulation_steps × num_GPUs**

| VRAM | 推荐 batch_size | gradient_accumulation |
|------|---------------|----------------------|
| 8GB | 1 | 8（等效8） |
| 16GB | 2 | 4（等效8） |
| 24GB | 4 | 2（等效8） |
| 80GB | 8 | 1（等效8） |

---

### `max_seq_length`（训练最大序列长度）

**含义：** 训练时每条数据最多处理多少 token，超出截断。

```python
# Unsloth 设置
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,   # 训练数据的最长长度
    load_in_4bit=True,
)
```

| 场景 | 推荐 max_seq_length |
|------|-------------------|
| 短对话（FAQ 风格） | `512 ~ 1024` |
| 通用指令微调 | `2048`（Unsloth 默认） |
| 长文档摘要 | `4096 ~ 8192` |
| RAG 上下文学习 | `4096` |

---

### `warmup_ratio`（学习率预热比例，Warmup Ratio）

**含义：** 训练开始时，学习率从 0 线性增加到目标值的过程占总步数的比例，防止初期参数剧烈震荡。

```python
warmup_ratio=0.03   # 前3%的步数做warmup（推荐）
warmup_steps=100    # 或者直接指定步数
```

---

### `weight_decay`（权重衰减，L2 正则化）

**含义：** 对模型权重施加 L2 正则化惩罚，防止过拟合。

```python
weight_decay=0.01   # 推荐默认
weight_decay=0.0    # 数据量大时可关闭
weight_decay=0.1    # 数据量小、过拟合严重时
```

---

## 7. 部署性能参数（Deployment Parameters）

---

### `tensor_parallel_size`（张量并行，Tensor Parallelism）

**含义（vLLM）：** 将模型的权重张量（Tensor）切分到多张 GPU 上。

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-72B-Instruct",
    tensor_parallel_size=4,   # 使用4张GPU
    dtype="bfloat16",
)
```

**规则：** tensor_parallel_size 必须是 2 的幂（1、2、4、8）。

---

### `gpu_memory_utilization`（GPU 内存利用率）

**含义（vLLM）：** 允许 vLLM 使用多少比例的 GPU 内存。剩余留给 KV Cache（键值缓存）。

```python
llm = LLM(
    model="qwen3:8b",
    gpu_memory_utilization=0.90,  # 使用 90% GPU 内存
)
```

| 取值 | 效果 |
|------|------|
| `0.95` | 最大化利用，KV Cache 小，并发低 |
| `0.90` | 推荐默认，平衡内存与并发 |
| `0.80` | 保守，稳定性高，适合生产 |

---

### `max_model_len`（最大模型长度，vLLM）

**含义：** 限制 vLLM 加载的最大序列长度，影响 KV Cache 大小和内存。

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    max_model_len=8192,   # 限制为 8192 而不是原始 128k
    # → 显著减少 KV Cache 内存占用
)
```

---

### `quantization`（vLLM 推理量化）

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    quantization="awq",       # 或 "gptq", "fp8", "int8"
    dtype="float16",
)
```

---

## 8. 参数速查表

### 推理参数（按场景）

| 场景 | temperature | top_p | top_k | repeat_penalty | max_tokens |
|------|------------|-------|-------|---------------|------------|
| **RAG 问答** | `0.1` | `0.9` | `40` | `1.1` | `512` |
| **代码生成** | `0.2` | `0.95` | `40` | `1.0` | `2048` |
| **数据提取/分类** | `0.0` | `1.0` | `1` | `1.0` | `100` |
| **日常对话** | `0.7` | `0.9` | `40` | `1.1` | `1024` |
| **创意写作** | `1.0` | `0.95` | `80` | `1.15` | `2048` |
| **摘要生成** | `0.3` | `0.9` | `40` | `1.1` | `512` |
| **JSON 输出** | `0.0` | `1.0` | `1` | `1.0` | `500` |

### 量化格式速查

| 量化 | 内存（7B） | 内存（13B） | 内存（70B） | 推荐场景 |
|------|-----------|-----------|-----------|---------|
| `F16` | 14 GB | 26 GB | 140 GB | 高端 GPU 基准 |
| `Q8_0` | 7 GB | 13 GB | 70 GB | 质量优先 |
| `Q5_K_M` | 5 GB | 9 GB | 50 GB | 质量与速度平衡 |
| `Q4_K_M` ⭐ | 4 GB | 7.5 GB | 40 GB | **默认推荐** |
| `Q3_K_M` | 3 GB | 5.5 GB | 30 GB | 极低内存 |

### RAG 参数速查

| 参数 | 推荐值 | 关键说明 |
|------|--------|---------|
| `num_ctx` | `8192` | RAG 场景必须手动设置！ |
| `temperature` | `0.1` | 低温减少幻觉 |
| `k`（检索数量） | `5` | 加 Reranker 时设 `20` |
| `chunk_size` | `512` | 绝大多数场景默认值 |
| `chunk_overlap` | `50` | chunk_size 的 10% |
| `score_threshold` | `0.7` | 过滤低质量检索结果 |

### LoRA 微调参数速查

| 参数 | 轻量微调 | 标准微调 | 强力微调 |
|------|---------|---------|---------|
| `r` | `8` | `16` | `64` |
| `lora_alpha` | `16` | `32` | `128` |
| `learning_rate` | `2e-4` | `2e-4` | `1e-4` |
| `epochs` | `1` | `3` | `5` |
| `batch_size` | `2` | `2` | `4` |
| `max_seq_length` | `1024` | `2048` | `4096` |

---

*文档版本：v1.0 | 2026年3月 | 参数基于 Ollama + LangChain + Unsloth + vLLM 生态*
