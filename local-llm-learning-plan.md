# 本地 LLM 学习计划：Ollama + Qwen + RAG + 微调 + 评估

> 目标：在本地搭建小模型环境，系统学习 RAG、Fine-tuning、Evaluation，并用 Claude Code 生成配套学习代码。
> 
> 技术栈：**Ollama · Qwen3 · LangChain · ChromaDB · Unsloth · RAGAS · Python**

---

## 一、环境准备（第 1 周）

### 1.1 硬件建议

| 配置 | 推荐模型 | 备注 |
|------|----------|------|
| CPU only / 低配 | Qwen3:0.6B / 1.7B (Q4) | 速度慢，仅用于学习 |
| 16GB RAM / M系列Mac | Qwen3:4B / 8B (Q4_K_M) | 流畅运行 RAG |
| RTX 4070+ / 16GB VRAM | Qwen3:14B (Q4) | 微调 + RAG 都顺畅 |

### 1.2 安装 Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取 Qwen3 模型（按自己硬件选择）
ollama pull qwen3:4b          # 轻量学习
ollama pull qwen3:8b          # 推荐日常使用
ollama pull nomic-embed-text  # Embedding 模型（RAG 必需）

# 验证安装
ollama list
ollama run qwen3:4b "你好，简单介绍一下自己"
```

### 1.3 Python 环境

```bash
python -m venv llm-learn
source llm-learn/bin/activate  # Windows: llm-learn\Scripts\activate

conda create -n llm-learn python=3.13 -y
conda activate llm-learn  # 如果使用 Conda


pip install ollama langchain langchain-community langchain-ollama \
            chromadb ragas unsloth datasets transformers \
            torch accelerate peft bitsandbytes fastapi uvicorn \
            pypdf sentence-transformers
```

### 1.4 项目目录结构

```
llm-workspace/
├── 01_basics/          # Ollama API 基础调用
├── 02_rag/             # RAG 管道实现
├── 03_rag_advanced/    # 高级 RAG：重排序、多查询
├── 04_evaluation/      # RAGAS 评估框架
├── 05_finetune/        # Unsloth 微调
├── 06_rag_finetune/    # RAG + 微调综合项目
└── data/               # 测试文档、数据集
```

---

## 二、阶段一：Ollama 基础 + 模型交互（第 1-2 周）

### 学习目标
- 掌握 Ollama API 调用方式（REST + Python Client）
- 理解 Chat vs Completion 模式
- 掌握 Qwen3 的 thinking / no-thinking 模式切换

### 核心概念
- **Context Window（上下文窗口）**：Qwen3:8B 支持 128k tokens，但本地运行需手动设置 `num_ctx`
- **Quantization（量化）**：Q4_K_M = 4-bit 量化，牺牲少量精度换取 75% 内存节省
- **Temperature**：控制输出随机性，RAG 任务建议 0.1-0.3，创意任务 0.7+

### 项目示例：`01_basics/`

**Claude Code 生成提示词：**
```
帮我创建一个 Python 文件 ollama_client.py，实现：
1. 使用 ollama Python 客户端调用 qwen3:8b 模型
2. 实现普通对话、流式输出两种模式
3. 实现多轮对话（维护 messages history）
4. 演示 /think 和 /no_think 模式切换
5. 实现一个简单的 CLI 聊天界面
```

### 学习资源
- [Ollama 官方文档](https://ollama.com/docs) — REST API 参考
- [Qwen3 官方博客](https://qwenlm.github.io/blog/qwen3/) — 模型能力详解
- [Codecademy: Qwen3 + Ollama](https://www.codecademy.com/article/qwen-3-ollama-setup-and-fine-tuning) — 上手教程

---

## 三、阶段二：RAG 基础管道（第 2-3 周）

### 学习目标
- 理解 RAG 完整流程：加载 → 分块 → Embedding → 存储 → 检索 → 生成
- 掌握 ChromaDB 向量数据库操作
- 实现 PDF 文档问答系统

### 核心概念

```
RAG 流程：

用户问题
    ↓
[Embedding 模型] → 问题向量
    ↓
[向量数据库 ChromaDB] → 相似度检索 → Top-K 文档块
    ↓
[Prompt 组装] → "基于以下上下文回答问题：{context}\n问题：{question}"
    ↓
[Qwen3 生成] → 最终答案
```

**关键参数说明：**
- `chunk_size`：每个文本块的 token 数，建议 500-1000
- `chunk_overlap`：块之间的重叠 token，建议 50-100（防止信息截断）
- `num_ctx`：给 Ollama 设置的上下文大小，RAG 任务建议设置 8192+
- `k`：检索返回的相似块数量，通常 3-5

### 项目示例：`02_rag/`

**Claude Code 生成提示词：**
```
帮我创建一个 PDF 问答 RAG 系统，包含以下文件：
- document_loader.py：加载 PDF，分块，生成 Embedding，存入 ChromaDB
- rag_chain.py：使用 LangChain + Ollama qwen3:8b 构建 RAG 链
- app.py：FastAPI 接口，支持上传 PDF 和问答
使用 nomic-embed-text 做 embedding，ChromaDB 做向量库，num_ctx=8192
```

### 学习资源
- [LangChain + Ollama RAG 教程](https://itsfoss.com/local-llm-rag-ollama-langchain/) — 完整 PDF RAG 实现
- [FreeCodeCamp: Build Local AI](https://www.freecodecamp.org/news/build-a-local-ai/) — Qwen3 + Ollama + RAG + Agent

---

## 四、阶段三：高级 RAG 技巧（第 3-4 周）

### 学习目标
- 掌握多查询检索（Multi-Query Retrieval）
- 掌握重排序（Reranking）提升检索质量
- 理解 HyDE（假设文档嵌入）技术
- 实现对话式 RAG（保留历史上下文）

### 核心概念

| 技术 | 原理 | 解决的问题 |
|------|------|-----------|
| **Multi-Query** | 用 LLM 将问题改写成多个变体，分别检索后合并 | 单一检索召回不足 |
| **Reranking** | 用 cross-encoder 模型对检索结果重新打分排序 | 检索结果相关性差 |
| **HyDE** | 先让 LLM 生成假设答案，再用答案做检索 | 问题与文档语义差距大 |
| **Contextual Compression** | 只保留检索块中与问题相关的部分 | 上下文 token 浪费 |

### 项目示例：`03_rag_advanced/`

**Claude Code 生成提示词：**
```
帮我基于 LangChain 实现高级 RAG，包含：
1. MultiQueryRetriever：用 qwen3:8b 生成3个查询变体
2. ContextualCompressionRetriever：压缩检索结果
3. 对话式 RAG：使用 ConversationBufferMemory 保留历史
4. 比较基础 RAG vs 高级 RAG 的检索质量对比脚本
向量库用 ChromaDB，embedding 用 nomic-embed-text
```

---

## 五、阶段四：RAG 评估框架（第 4-5 周）

### 学习目标
- 掌握 RAGAS 框架核心评估指标
- 实现自动化 RAG 评估流水线
- 理解 LLM-as-Judge 评估方法

### 核心评估指标

| 指标 | 含义 | 评估什么 |
|------|------|---------|
| **Faithfulness（忠实度）** | 答案是否忠于检索到的上下文 | 幻觉检测 |
| **Answer Relevancy（答案相关性）** | 答案是否真正回答了问题 | 答案质量 |
| **Context Recall（上下文召回）** | 检索到的内容是否覆盖了正确答案 | 检索质量 |
| **Context Precision（上下文精确率）** | 检索到的块有多少是真正有用的 | 检索噪音 |
| **Answer Correctness（答案正确性）** | 与参考答案的对比准确性 | 端到端质量 |

### 评估流程

```
1. 准备测试集：{question, ground_truth_answer, reference_contexts}
2. 运行 RAG 系统 → 得到 {answer, retrieved_contexts}
3. 送入 RAGAS → 得到各维度分数
4. 分析薄弱环节 → 优化 chunking / retrieval / prompt
```

### 项目示例：`04_evaluation/`

**Claude Code 生成提示词：**
```
帮我实现一个 RAGAS 评估脚本，要求：
1. 创建测试数据集（10个问答对）关于某个技术文档
2. 运行 RAG 管道获取答案和检索上下文
3. 使用 RAGAS 计算 faithfulness, answer_relevancy, context_recall, context_precision
4. 用 qwen3:8b（通过 Ollama）作为评估用的 LLM（LLM-as-Judge）
5. 输出评估报告到 CSV 和可视化图表
```

### 学习资源
- [RAGAS 官方文档](https://docs.ragas.io/en/stable/concepts/metrics/overview/) — 指标详解
- [RAGAS GitHub](https://github.com/vibrantlabsai/ragas) — 快速上手

---

## 六、阶段五：LLM 微调（Fine-tuning）（第 5-7 周）

### 学习目标
- 理解 LoRA / QLoRA 参数高效微调原理
- 掌握 Unsloth 框架进行本地微调
- 将微调后的模型导出为 GGUF，在 Ollama 中运行
- 构建微调数据集

### 核心概念

```
Full Fine-tuning（全参数微调）：
  更新全部 70B 参数 → 需要 8xA100 GPU，成本极高

LoRA（低秩适配）：
  冻结原始权重，只训练两个小矩阵 A 和 B
  可训练参数量 ≈ 1% → 普通 GPU 可跑

QLoRA（量化 LoRA）：
  在 LoRA 基础上，将原始模型压缩到 4-bit
  节省 75% 显存 → 16GB VRAM 可微调 7-13B 模型
```

**关键超参数：**
- `r`：LoRA rank，越大表达能力越强但参数越多，通常 8-64
- `lora_alpha`：缩放因子，通常 = 2*r
- `target_modules`：要注入 LoRA 的模块，通常是 attention 层
- `learning_rate`：微调学习率，建议 1e-4 到 2e-4

### 微调数据集格式（Alpaca 格式）

```json
{
  "instruction": "分析以下支付失败的原因",
  "input": "错误码：AM04，交易金额：5000 NZD",
  "output": "AM04 表示余额不足（Insufficient Funds）。建议提示用户检查账户余额后重试。"
}
```

### 项目示例：`05_finetune/`

**Claude Code 生成提示词：**
```
帮我使用 Unsloth 微调 Qwen2.5:7B 模型，完成以下任务：
1. prepare_dataset.py：生成一个包含100条金融/支付领域问答的训练数据集（Alpaca格式）
2. finetune.py：使用 QLoRA（4-bit）微调，r=16, lora_alpha=32
3. 训练完成后，保存 LoRA adapter 并转换为 GGUF 格式
4. create_modelfile.py：生成 Ollama Modelfile，用于本地运行微调模型
5. inference_compare.py：对比原始模型 vs 微调模型的回答质量
```

### 微调后导入 Ollama

```bash
# 转换为 GGUF（Unsloth 支持一行导出）
# 在 finetune.py 中已自动生成 model.gguf

# 创建 Modelfile
cat > Modelfile << 'EOF'
FROM ./model.gguf
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
SYSTEM "你是一个专业的支付系统助手，专注于 ISO 20022 消息格式和支付处理。"
EOF

# 注册到 Ollama
ollama create payment-assistant -f Modelfile

# 测试
ollama run payment-assistant "解释 AM04 错误码的含义和处理方式"
```

### 学习资源
- [Unsloth 官方文档](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide) — 最权威的 LoRA 微调指南
- [DataCamp Unsloth 教程](https://www.datacamp.com/tutorial/unsloth-guide-optimize-and-speed-up-llm-fine-tuning) — 实战案例
- [Hugging Face Fine-tune 指南](https://www.decodingai.com/p/playbook-to-fine-tune-and-deploy) — 从微调到部署全流程

---

## 七、阶段六：综合项目（第 7-8 周）

### 项目：`06_rag_finetune/` — 领域专属智能问答系统

结合 RAG + 微调 + 评估，构建一个面向特定领域的完整 AI 助手。

**项目架构：**

```
用户问题
    ↓
[微调后的 Qwen3（专业术语理解提升）]
    ↓
[RAG 检索（最新文档知识）]
    ↓
[RAGAS 自动评估]
    ↓
最终答案 + 置信度分数
```

**Claude Code 生成提示词：**
```
帮我构建一个完整的领域智能问答系统，包含以下模块：
1. ingest.py：文档入库（支持 PDF、Markdown、TXT）
2. rag_engine.py：高级 RAG 检索引擎（多查询 + 重排序）
3. llm_backend.py：支持切换原始 Qwen3 vs 微调版本
4. evaluator.py：RAGAS 自动评估，输出质量报告
5. api.py：FastAPI 接口，支持问答 + 评估触发
6. README.md：完整使用文档
使用 Ollama 作为推理后端，ChromaDB 作为向量库
```

---

## 八、整体学习时间线

```
Week 1    Week 2    Week 3    Week 4    Week 5    Week 6    Week 7    Week 8
  │         │         │         │         │         │         │         │
  ├── 环境搭建 + Ollama 基础 ──┤
              ├── RAG 基础管道 ──┤
                        ├── 高级 RAG ──┤
                                  ├── RAGAS 评估 ──┤
                                            ├──── 微调 LoRA/QLoRA ────┤
                                                              ├── 综合项目 ──┤
```

---

## 九、核心学习资源汇总

### 官方文档
| 资源 | 链接 | 用途 |
|------|------|------|
| Ollama 文档 | https://ollama.com/docs | API 参考 |
| Qwen3 博客 | https://qwenlm.github.io/blog/qwen3/ | 模型能力 |
| Unsloth 文档 | https://docs.unsloth.ai | 微调框架 |
| RAGAS 文档 | https://docs.ragas.io | 评估框架 |
| LangChain 文档 | https://python.langchain.com | RAG 管道 |

### 优质教程
| 教程 | 重点内容 |
|------|---------|
| FreeCodeCamp: Build Local AI | Qwen3 + Ollama + RAG + Agent 全栈实践 |
| itsfoss.com LangChain RAG | LangChain + ChromaDB PDF 问答 |
| Unsloth Colab Notebooks | 免费 GPU 微调（Kaggle/Colab） |
| DataCamp Unsloth Guide | QLoRA 微调全流程 |
| MachineLearning Mastery | LoRA/QLoRA 选择决策框架 |

### Hugging Face 数据集（微调用）
- `tatsu-lab/alpaca` — 通用指令数据集（52K 条）
- `HuggingFaceH4/ultrachat_200k` — 对话质量高
- `databricks/databricks-dolly-15k` — 任务多样
- 自建：围绕 Eduacan / 支付系统领域构建专业数据集

---

## 十、Claude Code 使用建议

每个阶段开始前，可以用以下方式与 Claude Code 协作：

```bash
# 1. 生成项目骨架
claude "在 02_rag 目录下创建 RAG 项目，使用 LangChain + ChromaDB + Ollama"

# 2. 迭代改进
claude "修改 rag_chain.py，添加多查询检索支持"

# 3. 调试问题
claude "这段代码报错 ChromaDB collection not found，帮我分析原因并修复"

# 4. 生成测试数据
claude "生成50条关于 ISO 20022 支付消息的问答对，保存为 JSONL 格式"
```

**提示：** 可以把你的 Westpac FTH 支付系统知识作为领域数据，在阶段六构建一个支付领域专属助手——这会让学习更有实际意义！

---

*计划版本：v1.0 | 2026年3月 | 基于 Qwen3 + Ollama + Unsloth + RAGAS*
