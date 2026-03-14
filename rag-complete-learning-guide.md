# RAG 完整学习指南：从 Chunk 到 GraphRAG

> 覆盖范围：基础 RAG → Chunking 策略 → Embedding 模型 → 高级 RAG → 企业级 RAG → GraphRAG
> 
> 每个阶段附 Claude Code 提示词，可直接用于生成学习代码。

---

## 目录

1. [RAG 全景图](#1-rag-全景图)
2. [Chunking：分块策略详解](#2-chunking分块策略详解)
3. [Embedding：向量化模型选择](#3-embedding向量化模型选择)
4. [基础 RAG 管道实现](#4-基础-rag-管道实现)
5. [高级 RAG 技术](#5-高级-rag-技术)
6. [企业级大规模 RAG](#6-企业级大规模-rag)
7. [GraphRAG：图增强检索](#7-graphrag图增强检索)
8. [RAG 评估体系](#8-rag-评估体系)
9. [学习路线总结](#9-学习路线总结)

---

## 1. RAG 全景图

### 1.1 RAG 是什么，为什么需要它

LLM（Large Language Model，大语言模型）的知识来自训练数据，存在三个根本限制：

- **知识截止日期（Knowledge Cutoff）**：训练完成后不再更新
- **私有数据盲区（Private Data Blindspot）**：无法获取企业内部、私密文档
- **幻觉问题（Hallucination）**：对不确定的知识倾向于"编造"

RAG（Retrieval-Augmented Generation，检索增强生成）的核心思路是**开卷考试**：让模型在回答时"翻阅"外部文档，而不是凭记忆作答。

```
                    ┌─────────────────────────────────────┐
                    │           RAG 系统架构               │
                    └─────────────────────────────────────┘

  [文档库]                [离线索引]                [在线查询]
  PDF / Word             ┌──────────┐              用户问题
  Markdown       ──────→ │ Chunking │                  │
  Database               └────┬─────┘                  ▼
                               │                  [Embed 问题]
                               ▼                        │
                         [Embedding]                    ▼
                               │              [向量相似度检索]
                               ▼                        │
                         [向量数据库]    ←───────────────┘
                         ChromaDB                       │
                         pgvector               [Top-K 文本块]
                         Qdrant                         │
                         Weaviate                       ▼
                                             [Prompt 组装 + LLM]
                                                        │
                                                        ▼
                                                   最终答案
```

### 1.2 RAG 的演进三阶段

| 阶段 | 名称 | 特点 |
|------|------|------|
| **Naive RAG** | 朴素 RAG | 直接检索 + 生成，无优化 |
| **Advanced RAG** | 高级 RAG | 预检索（Pre-Retrieval）优化 + 后检索（Post-Retrieval）优化 |
| **Modular RAG** | 模块化 RAG | 路由（Routing）、反馈循环（Feedback Loop）、Agentic 编排 |
| **GraphRAG** | 图 RAG | 知识图谱（Knowledge Graph）+ 关系推理（Relational Reasoning），超越向量检索 |

---

## 2. Chunking：分块策略详解

Chunking（分块）是 RAG 管道（Pipeline）的第一步，也是影响最终效果最大的单一因素。**错误的分块策略会成为整个系统的性能上限。**

### 2.1 为什么要 Chunking

1. **Embedding 模型有 Token 限制（Token Limit）**：大多数模型最多处理 512-8192 tokens
2. **上下文窗口限制（Context Window Limit）**：检索到的内容必须塞进 LLM 的 Prompt，不能太大
3. **检索精度（Retrieval Precision）**：块越大，语义越分散，相关性匹配越难

### 2.2 核心参数

```python
# 三个关键参数
chunk_size = 512        # 每块的 token 数
chunk_overlap = 50      # 相邻块重叠的 token 数（防止信息截断在边界）
separators = ["\n\n", "\n", ".", " "]  # 切割的优先分隔符顺序
```

**chunk_size 选择原则：**
- 问题答案简短具体（FAQ）→ 小块（128-256 tokens）
- 需要上下文理解的问题 → 中块（512-1024 tokens）
- 需要主题概览的问题 → 大块或层级块（Hierarchical Chunks，1024+ tokens）

**overlap（重叠）选择原则：**
- 通常设置为 chunk_size 的 10%-20%
- 2026年研究表明：对于 SPLADE（Sparse Lexical and Dense Retrieval）检索，overlap 对效果几乎无影响；但对 dense retrieval（稠密检索）有正向作用

### 2.3 分块策略全览

#### 策略一：Fixed-Size Chunking（固定大小分块）

最简单，按 token 数或字符数机械切割，不考虑语义。

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunks = splitter.split_text(text)
```

**适用场景：** 快速原型验证、同质化的短文档（新闻文章、FAQ）

**缺点：** 可能在句子中间截断，破坏语义完整性

---

#### 策略二：Recursive Character Splitting（递归字符分块）⭐ 默认首选

LangChain 的默认推荐策略，按优先级依次尝试多种分隔符，优先按段落切，其次按句子，最后按单词。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    # separators 从左到右优先级递减
)
chunks = splitter.split_documents(documents)
```

**2026年基准测试：** 在50篇学术论文测试中，512-token 递归分块达到 **69% 准确率**，排名第一。

**适用场景：** 绝大多数 RAG 应用的默认选择，通用性最强

---

#### 策略三：Semantic Chunking（语义分块）

不按字符数切，而是按**语义相似度（Semantic Similarity）**切。相邻句子如果语义相差超过阈值（Threshold），就在那里切分。

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",   # 或 "standard_deviation"
    breakpoint_threshold_amount=95,            # 语义差异超过95百分位才切割
)
chunks = splitter.split_text(text)
```

**原理：**

```
句子1 → embed → vec1 ─┐
句子2 → embed → vec2 ─┤→ cosine_similarity → 差异大？→ 在此切割
句子3 → embed → vec3 ─┘
```

**代价：** 每个句子都要做一次 Embedding，计算量大（10k词文档需200-300次embedding调用）

**召回率（Recall）提升：** 相比基础分块，召回率可提升最高 9 个百分点（Chroma 研究）

**适用场景：** 知识库、技术文档，对质量要求高、对速度要求不严苛

---

#### 策略四：Structure-Aware Chunking（结构感知分块）⭐ 推荐用于 Markdown/HTML

利用文档自身的结构（标题层级、代码块、表格）来切分，而不是盲目按字符数切。

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 按 Markdown 标题层级切分
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

# 每个 chunk 自动包含 metadata：{"h1": "...", "h2": "..."}
# 这个 metadata 在检索时可以用于过滤！
```

**适用场景：** Markdown 文档、API 文档、技术规范（如 ISO 20022 规范文档）

---

#### 策略五：Late Chunking（延迟分块）🆕 2024-2025 新技术

传统分块先切后 embed，每个块"不知道"自己在整篇文档的位置。Late Chunking 先对整篇文档做 Embedding，保留完整的上下文信息（Full Context），再在 embedding 空间（Embedding Space）中切分。

```
传统方式：
  [句子A] → embed → vec_A  （不知道句子B的存在）
  [句子B] → embed → vec_B  （不知道句子A的存在）

Late Chunking：
  [句子A | 句子B | 句子C | ...] → 长文档 embed → [vec_A' | vec_B' | vec_C' | ...]
                                                       ↑ 每个位置的向量都"看到了"全文
```

**效果：** 在代词和跨引用场景中显著改善，例如"他"、"该系统"等指代关系能被正确处理

**要求：** 需要支持长上下文的 Embedding 模型（如 Jina v3，支持 8192 tokens）

---

#### 策略六：Contextual Retrieval（上下文检索）🆕 Anthropic 提出

由 Anthropic 在 2024 年提出。在存入向量库（Vector Store）**之前**，先让 LLM 为每个 chunk 生成一段简短的"位置描述"，让每个 chunk 变得自包含（Self-Contained）。

```python
# 原始 chunk（缺乏上下文）：
"该系统在 2023 年 Q3 季度出现了 15% 的性能下降。"

# 添加上下文后的 chunk：
"""
本段来自 Westpac 年度技术报告第三章"性能监控"一节，
描述了 FTH 支付系统在 2023 年 Q3 的性能问题。
---
该系统在 2023 年 Q3 季度出现了 15% 的性能下降。
"""
```

**实现思路：**

```python
CONTEXT_PROMPT = """
文档内容：
<document>
{full_document}
</document>

以下是需要添加上下文的文本块：
<chunk>
{chunk}
</chunk>

请用1-2句话简洁描述这个文本块在整篇文档中的位置和作用，
直接输出描述，不要有任何前缀。
"""

def add_context(chunk, full_document, llm):
    context = llm.invoke(CONTEXT_PROMPT.format(
        full_document=full_document, chunk=chunk
    ))
    return f"{context}\n---\n{chunk}"
```

**成本：** 每个 chunk 需要一次 LLM 调用（可用小模型），但召回率提升显著

---

#### 策略七：Hierarchical / Parent-Child Chunking（层级分块）

双层结构：**父块（Parent Chunk）**用于生成最终答案（保持完整语义），**子块（Child Chunk）**用于检索（更精准匹配）。

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 父块：大块，用于生成答案
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# 子块：小块，用于精准检索
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(embedding_function=embeddings)
docstore = InMemoryByteStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**流程：**
1. 检索时（Retrieval）：用小块做精准匹配（找到相关位置）
2. 生成时（Generation）：返回对应的大块（提供完整上下文）

---

### 2.4 Chunking 策略选择决策树

```
你的文档是什么类型？
│
├── 有明确结构（Markdown、HTML）→ Structure-Aware Chunking
│
├── 短小均一（FAQ、新闻）→ Fixed-Size 或 Recursive
│
├── 长篇、多主题混合 → Semantic Chunking
│
├── 含大量代词和跨段引用 → Late Chunking
│
└── 对质量极度敏感（法律、医疗）→ Contextual Retrieval + Hierarchical
```

**2026 年实战默认推荐：**
1. 先用 `RecursiveCharacterTextSplitter(chunk_size=512, overlap=50)` 建立基线
2. 如文档有结构 → 换 `MarkdownHeaderTextSplitter`
3. 基线不满足 → 上 `SemanticChunker`
4. 还不够 → 加 Contextual Retrieval

---

### 2.5 Claude Code 提示词：Chunking 实验对比

```
创建一个 chunking_benchmark.py 脚本，对同一份 PDF 文档测试以下4种分块策略：
1. RecursiveCharacterTextSplitter（chunk_size=512）
2. SemanticChunker（使用 OllamaEmbeddings nomic-embed-text）
3. MarkdownHeaderTextSplitter（如文档包含 markdown）
4. ParentDocumentRetriever（父块2000，子块400）

对每种策略，用10个测试问题评估召回率（检索到的块是否包含正确答案）。
输出对比表格：策略名、平均块大小、块数量、召回率@3、耗时。
```

---

## 3. Embedding：向量化模型选择

Embedding（向量化）将文本转为高维向量（High-Dimensional Vector），是"语义理解（Semantic Understanding）"的核心。**选错 Embedding 模型，整个 RAG 系统的检索质量上限就被锁死了。**

### 3.1 Embedding 的本质

```
"支付失败" → [0.23, -0.87, 0.45, ..., 0.12]  (N维向量)
"交易被拒绝" → [0.25, -0.89, 0.43, ..., 0.11]  (语义相近 → 向量相近)
"今天天气不错" → [-0.54, 0.32, -0.71, ..., 0.88]  (语义不相关 → 向量远离)
```

相似度计算（Similarity Calculation）：`cosine_similarity(v1, v2) = dot(v1, v2) / (||v1|| × ||v2||)`

### 3.2 主流 Embedding 模型对比（2025-2026）

#### 本地部署（免费，数据不出本地）

| 模型 | 参数量 | 维度 | 最大 Token | MTEB 得分 | 特点 |
|------|--------|------|-----------|-----------|------|
| **nomic-embed-text** | 137M | 768 | 8192 | ~62 | Ollama 原生支持，最易本地部署 |
| **nomic-embed-text-v2** | 475M(MoE) | 768 | 8192 | ~65 | MoE架构，多语言，推荐 |
| **BGE-M3** | 568M | 1024 | 8192 | 63.0 | 支持 dense（稠密）+ sparse（稀疏）+ multi-vector（多向量），多语言 |
| **BGE-large-en-v1.5** | 335M | 1024 | 512 | 64.2 | 英文最强开源之一 |
| **E5-base-v2** | 109M | 768 | 512 | 63.4 | 速度快，集成简单 |
| **all-MiniLM-L6-v2** | 22M | 384 | 256 | 56 | 极轻量，速度最快，质量一般 |

#### 云端 API（高质量，按量付费）

| 模型 | 维度 | MTEB | 价格（/M tokens） | 特点 |
|------|------|------|------|------|
| **Qwen3-Embedding-8B** | 4096 | 70.58 🏆 | 按API计费 | 多语言最强，中文尤佳 |
| **OpenAI text-embedding-3-large** | 3072 | 64.6 | $0.13 | OpenAI生态最佳 |
| **OpenAI text-embedding-3-small** | 1536 | 62.3 | $0.02 | 性价比高 |
| **Cohere embed-v4** | 1536 | 65.2 🥇 | 按量计费 | 多模态（文本+图片），128k上下文 |

### 3.3 各模型详解

#### nomic-embed-text（本地学习首选）

```bash
# Ollama 一行拉取
ollama pull nomic-embed-text

# Python 使用
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 注意：nomic 是 instruction-aware 模型，需要加前缀
# 检索文档时：
doc_embedding = embeddings.embed_documents(["search_document: " + text])
# 检索查询时：
query_embedding = embeddings.embed_query("search_query: " + query)
```

**特点：** 完全本地运行，8192 tokens 长上下文，Apache 2.0 开源，MTEB 表现超过 OpenAI ada-002

---

#### BGE-M3（企业本地部署首选）

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# BGE-M3 独特能力：同时输出三种表示
output = model.encode(['query text'], 
    return_dense=True,    # 传统 dense vector（语义相似度）
    return_sparse=True,   # sparse vector（关键词匹配，类似BM25）
    return_colbert_vecs=True  # 多向量（更精准的late interaction）
)

dense_vectors = output['dense_vecs']   # shape: (1, 1024)
sparse_vectors = output['lexical_weights']  # dict of token→weight
```

**为什么强大：** 一个模型同时支持 dense retrieval（稠密检索）、sparse retrieval（稀疏检索）、multi-vector retrieval（多向量检索）三种检索方式，可以轻松构建混合检索（Hybrid Search），效果大幅超过纯向量检索

---

#### Qwen3-Embedding（中文场景最强）

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                             trust_remote_code=True)

# Qwen3 也是 instruction-aware
queries = ["Instruct: 检索相关文档\nQuery: 支付失败的原因"]
documents = ["Passage: AM04 错误码表示账户余额不足..."]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(documents, normalize_embeddings=True)
```

**MTEB 多语言排行榜得分 70.58，目前开源第一**

---

### 3.4 Embedding 选择决策框架

```
是否需要完全本地（数据隐私）？
│
├── 是 → 硬件配置？
│         ├── 低配（CPU/16GB RAM）→ nomic-embed-text（Ollama）
│         └── 中高配（GPU）→ BGE-M3（支持hybrid search）
│
└── 否 → 主要语言？
          ├── 中文为主 → Qwen3-Embedding
          ├── 多语言 → Cohere embed-v4
          └── 英文为主 → OpenAI text-embedding-3-large（性价比高）
```

**永远记住：在你自己的数据上做基准测试（Benchmarking），MTEB（Massive Text Embedding Benchmark）通用分数不代表你的领域表现。专业领域（法律、医疗、金融）微调（Fine-tuning）后可提升 10-30%。**

---

### 3.5 向量数据库选择

| 数据库 | 适用场景 | 特点 |
|--------|---------|------|
| **ChromaDB** | 本地学习、原型开发（Prototyping） | 零配置，Python 原生 |
| **pgvector** | 已有 PostgreSQL | 与关系数据（Relational Data）集成，<10M 向量 |
| **Qdrant** | 中大规模生产（Production） | 高性能，Rust 实现，支持 payload 过滤（Payload Filtering） |
| **Weaviate** | 企业多模态（Multimodal） | 内置 hybrid search，GraphQL 接口 |
| **Milvus** | 超大规模（亿级，Billion-scale） | 分布式（Distributed），GPU 加速 |

---

### 3.6 Claude Code 提示词：Embedding 对比实验

```
创建 embedding_benchmark.py，对同一组测试问答对（20条），
分别使用以下 embedding 模型测试检索召回率@5：
1. nomic-embed-text（通过 Ollama）
2. BAAI/bge-base-en-v1.5（通过 sentence-transformers）
3. intfloat/e5-base-v2（通过 sentence-transformers）

向量库统一用 ChromaDB，chunk_size 统一 512。
输出对比：模型名、向量维度、embedding 耗时、召回率@3、召回率@5。
并可视化每个模型的向量空间分布（用 UMAP 降维到 2D，matplotlib 绘图）。
```

---

## 4. 基础 RAG 管道实现

### 4.1 完整流程代码架构

```
project/
├── ingest.py          # 文档加载、分块、向量化、存库
├── retriever.py       # 检索逻辑封装
├── rag_chain.py       # LangChain RAG 链
└── app.py             # FastAPI 接口
```

### 4.2 ingest.py：文档入库

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

def ingest_documents(doc_path: str, collection_name: str = "rag_docs"):
    # 1. 加载文档
    loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"加载了 {len(documents)} 页文档")

    # 2. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        add_start_index=True,    # 记录每块在原文的位置
    )
    chunks = splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个文本块")

    # 3. 向量化并存库
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db",
    )
    print(f"已存入向量数据库，collection: {collection_name}")
    return vectorstore
```

### 4.3 rag_chain.py：RAG 问答链

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一个专业的文档问答助手。
请严格基于以下检索到的上下文内容回答问题。
如果上下文中没有相关信息，请直接说"根据文档，我找不到相关信息"，不要编造。

上下文：
{context}

问题：{question}

回答：
""")

def build_rag_chain(collection_name: str = "rag_docs"):
    # 加载向量库
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    # 检索器：返回 Top-5 相似块
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # LLM（设置大上下文窗口！这是关键）
    llm = ChatOllama(
        model="qwen3:8b",
        num_ctx=8192,        # ⚠️ 必须手动设置，否则默认2048会截断上下文
        temperature=0.1,     # RAG 任务低温度，减少幻觉
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )

    # LCEL 链（LangChain Expression Language）：检索 → 格式化 → Prompt → LLM → 解析
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
```

### 4.4 Claude Code 提示词：基础 RAG 项目

```
在 04_basic_rag/ 目录下创建完整的 PDF 问答 RAG 项目：
1. ingest.py：支持单 PDF 和目录批量入库，记录 metadata（文件名、页码）
2. rag_chain.py：LangChain LCEL 链，使用 qwen3:8b + nomic-embed-text + ChromaDB
3. app.py：FastAPI，提供 POST /ingest 和 POST /query 两个接口
4. test_rag.py：10个测试问题验证系统可用性
5. README.md：安装和使用说明

注意：num_ctx 必须设置为 8192，temperature 设 0.1
```

---

## 5. 高级 RAG 技术

高级 RAG 在基础 RAG 的 "检索→生成" 两端各加了优化层：

```
[用户问题]
    ↓
=== 预检索优化 (Pre-Retrieval) ===
• 查询重写（Query Rewriting）
• 查询扩展（HyDE）
• 问题分解（Query Decomposition）
    ↓
[向量检索]
    ↓
=== 后检索优化 (Post-Retrieval) ===
• 重排序（Reranking）
• 上下文压缩（Contextual Compression）
• 融合检索（Hybrid Search）
    ↓
[LLM 生成]
```

### 5.1 Multi-Query Retrieval（多查询检索）

单一查询措辞不同可能导致漏检。Multi-Query 让 LLM 把问题改写成多个角度，各自检索后合并去重。

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0.7)  # 用高温度鼓励多样性

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

# 内部会将 "ISO 20022 支付失败原因" 改写为多个变体：
# - "ISO 20022 消息中哪些错误码表示支付失败？"
# - "支付系统中常见的拒付原因有哪些？"
# - "AM04 错误码在 ISO 20022 中的含义是什么？"
# 每个变体各自检索，最终结果取并集
docs = multi_query_retriever.invoke("ISO 20022 支付失败原因")
```

---

### 5.2 HyDE（Hypothetical Document Embeddings）

**问题：** 用户的查询（问题形式）和文档内容（答案形式）在语义空间中可能相差较大

**解决：** 先让 LLM 生成一个"假设的答案"，用这个假设答案去做检索（与文档同形式，语义更接近）

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:8b", temperature=0.5)

# 构建 HyDE embedder
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    # 让 LLM 生成假设文档的 prompt
)

# 用 HyDE embeddings 构建检索器
vectorstore_hyde = Chroma.from_documents(
    chunks, embedding=hyde_embeddings
)
retriever = vectorstore_hyde.as_retriever(search_kwargs={"k": 5})
```

**适用场景：** 用户问题和文档内容在写法上差异很大（如用户问口语化问题，文档是专业术语）

---

### 5.3 Reranking（重排序）⭐ 效果提升最显著

向量检索返回的 Top-K 结果是基于向量相似度的粗排，准确度有限。Reranker（Cross-Encoder）对检索结果做精排，显著提高最终质量。

```
向量检索（Bi-Encoder，快）：
  Query ──embed──→ q_vec
  Doc1  ──embed──→ d1_vec  }  cosine similarity → 粗排 Top-20
  Doc2  ──embed──→ d2_vec  }

重排序（Cross-Encoder，慢但准）：
  [Query + Doc1] → Cross-Encoder → 精确相关性分数
  [Query + Doc2] → Cross-Encoder → 精确相关性分数
  → 根据精确分数重新排序 → 取 Top-5
```

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 加载 Cross-Encoder Reranker
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=5)

# 组合：先粗检索 20 个，再精排取 5 个
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

docs = compression_retriever.invoke("你的查询")
```

**推荐 Reranker 模型：**
- `BAAI/bge-reranker-v2-m3`：开源最强，多语言
- `cross-encoder/ms-marco-MiniLM-L-6-v2`：英文，速度快
- `Cohere Rerank`：云端 API，质量最高

---

### 5.4 Hybrid Search（混合检索）

纯语义向量检索对精确关键词匹配较弱（如特定型号、错误码）。混合检索结合**向量检索**（语义）和 **BM25 检索**（关键词），取长补短。

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 向量检索（语义）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 关键词检索
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 混合：60% 向量 + 40% BM25（权重可调）
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

docs = ensemble_retriever.invoke("AM04 错误码")
```

**企业场景中，Hybrid Search 几乎是标配。** 搜索 "AM04" 这种精确错误码，纯向量检索可能找到语义相近但不精确的内容，BM25 却能精确命中。

---

### 5.5 Conversational RAG（对话式 RAG）

支持多轮对话，将历史对话压缩为上下文传入检索。

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. 创建"历史感知"检索器（能根据对话历史重写查询）
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据对话历史，将用户的最新问题重写为独立的搜索查询。只输出查询，不解释。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

# 2. 问答链
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下上下文回答问题。\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 3. 完整对话 RAG 链
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 使用
chat_history = []
result = rag_chain.invoke({"input": "支付失败的常见原因是什么？", "chat_history": chat_history})
chat_history.extend([HumanMessage(content="支付失败..."), AIMessage(content=result["answer"])])

# 第二轮，会自动利用历史上下文
result2 = rag_chain.invoke({"input": "AM04 呢？", "chat_history": chat_history})
```

---

### 5.6 Claude Code 提示词：高级 RAG 全套

```
在 05_advanced_rag/ 目录下实现高级 RAG，包含以下模块：
1. multi_query.py：MultiQueryRetriever，展示3个改写变体和各自检索结果
2. reranking.py：使用 BAAI/bge-reranker-v2-m3 对基础检索结果重排序，
   对比重排前后 Top-5 结果的变化
3. hybrid_search.py：BM25 + 向量的 EnsembleRetriever，
   测试精确术语查询（如 "AM04"）和语义查询的效果差异
4. conversational_rag.py：支持5轮多轮对话的 RAG 链，
   保存对话历史并可视化历史对检索的影响
5. compare_all.py：用同10个问题对比所有策略的效果，生成对比表格
```

---

## 6. 企业级大规模 RAG

### 6.1 企业 RAG 与个人 RAG 的核心差异

| 维度 | 个人/原型 RAG | 企业级 RAG |
|------|-------------|-----------|
| 文档规模 | 几十个 PDF | 数百万文档 |
| 并发量 | 单人 | 千人同时访问 |
| 权限控制 | 无 | 细粒度 RBAC |
| 数据更新 | 手动重入库 | 实时流式更新 |
| 可观测性 | 无 | 全链路 tracing |
| 语言 | 单语言 | 多语言 |
| 合规 | 无 | GDPR、SOX、HIPAA |

### 6.2 企业 RAG 架构全图

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   企业 RAG 平台                      │
                    └─────────────────────────────────────────────────────┘

  数据源层                  处理层                    服务层
  ┌──────────┐          ┌──────────────┐          ┌──────────────────┐
  │ SharePoint│──────→  │  Document    │          │   API Gateway    │
  │ Confluence│          │  Pipeline   │          │  (认证/限流/路由) │
  │ S3/Blob  │          │             │          └────────┬─────────┘
  │ Database │          │ ┌──────────┐│                   │
  │ Email    │          │ │Chunking  ││          ┌────────▼─────────┐
  └──────────┘          │ └────┬─────┘│          │   RAG Engine     │
                        │      ↓      │          │                  │
  实时流数据              │ ┌──────────┐│          │ Query Router     │
  ┌──────────┐          │ │Embedding ││          │ ┌──────────────┐ │
  │  Kafka   │──────→  │ └────┬─────┘│          │ │Simple Query  │ │
  │  Queue   │          │      ↓      │          │ │→ Basic RAG   │ │
  └──────────┘          │ ┌──────────┐│          │ └──────────────┘ │
                        │ │Metadata  ││          │ ┌──────────────┐ │
                        │ │Enrichment││          │ │Complex Query │ │
                        │ └────┬─────┘│          │ │→ GraphRAG    │ │
                        │      ↓      │          │ └──────────────┘ │
                        └──────┼──────┘          └────────┬─────────┘
                               │                           │
                    ┌──────────▼──────────┐    ┌─────────▼──────────┐
                    │   向量数据库集群      │    │    LLM 集群         │
                    │   (Qdrant/Weaviate) │    │  (Qwen/GPT-4/等)   │
                    │   + 权限过滤        │    │  + LLM 负载均衡     │
                    └─────────────────────┘    └────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   可观测性平台        │
                    │   Langfuse/Phoenix  │
                    │   全链路 Trace      │
                    └─────────────────────┘
```

### 6.3 关键企业能力实现

#### 6.3.1 权限过滤（Access Control）

不同用户只能看到有权限的文档，在检索层直接过滤。

```python
# 存入时打上权限标签
vectorstore.add_documents(
    documents=chunks,
    metadatas=[{
        "source": "hr_policy.pdf",
        "department": "HR",           # 部门标签
        "sensitivity": "confidential", # 敏感级别
        "allowed_roles": ["hr_admin", "hr_manager"]  # 允许角色
    }]
)

# 检索时按用户权限过滤（Qdrant 过滤示例）
from qdrant_client.models import Filter, FieldCondition, MatchAny

user_roles = get_user_roles(user_id)  # 从认证系统获取用户角色

results = qdrant_client.search(
    collection_name="enterprise_docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="allowed_roles",
                match=MatchAny(any=user_roles)
            )
        ]
    ),
    limit=10
)
```

#### 6.3.2 实时文档更新（Incremental Indexing）

```python
# 监听文档变更，增量更新向量库
import hashlib

def upsert_document(doc_path: str, vectorstore):
    doc_hash = hashlib.md5(open(doc_path, 'rb').read()).hexdigest()

    # 检查是否已存在且未更改
    existing = vectorstore.get(where={"doc_hash": doc_hash})
    if existing['ids']:
        print(f"文档未变更，跳过：{doc_path}")
        return

    # 删除旧版本（如果存在）
    old = vectorstore.get(where={"source": doc_path})
    if old['ids']:
        vectorstore.delete(ids=old['ids'])
        print(f"删除旧版本：{len(old['ids'])} 个块")

    # 重新入库
    chunks = process_document(doc_path)
    for chunk in chunks:
        chunk.metadata['doc_hash'] = doc_hash
    vectorstore.add_documents(chunks)
    print(f"更新完成：{len(chunks)} 个块")
```

#### 6.3.3 可观测性（Observability）

使用 Langfuse 追踪每次 RAG 调用的全链路数据，支持后续分析和调试。

```python
from langfuse.callback import CallbackHandler

# 初始化 Langfuse（支持自托管）
langfuse_handler = CallbackHandler(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="http://localhost:3000"  # 自托管地址
)

# 在 RAG chain 中传入 callback
result = rag_chain.invoke(
    {"input": user_query, "chat_history": chat_history},
    config={"callbacks": [langfuse_handler]}
)
# Langfuse 自动记录：查询→检索→Prompt→LLM→输出的全链路耗时、token 数、成本
```

#### 6.3.4 缓存层（Semantic Cache）

相似的问题使用缓存结果，大幅降低 LLM 调用成本。

```python
from langchain.cache import InMemoryCache
from langchain_community.cache import RedisSemanticCache
import langchain

# 语义缓存：语义相似的查询复用缓存（不要求精确匹配）
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    score_threshold=0.95,  # 相似度超过 0.95 才命中缓存
)
```

### 6.4 推荐企业 RAG 技术栈（2025）

| 层级 | 推荐选型 | 备注 |
|------|---------|------|
| **文档解析** | Docling / Unstructured | 处理 PDF、Word、表格、图片 |
| **Chunking** | LangChain + Semantic | 按场景选择 |
| **Embedding** | BGE-M3 / Qwen3-Embedding | 本地 or 云端 |
| **向量库** | Qdrant / Weaviate | 生产级，支持过滤 |
| **编排框架** | LangGraph / Haystack | 复杂工作流 |
| **Reranker** | BGE-Reranker-v2-M3 | 开源最强 |
| **LLM** | Qwen3 / GPT-4o | 按需选择 |
| **缓存** | Redis Semantic Cache | 降低成本 |
| **可观测性** | Langfuse（自托管） | 全链路追踪 |
| **评估** | RAGAS + DeepEval | 自动化质量监控 |

---

## 7. GraphRAG：图增强检索

### 7.1 为什么需要 GraphRAG

传统向量 RAG 的根本局限：**把文档当孤岛，看不见文档之间的关系**。

```
传统 RAG 能回答：
✅ "AM04 错误码是什么意思？"（单点事实查询）
✅ "ISO 20022 消息格式是什么？"（文档内知识）

传统 RAG 难以回答：
❌ "我们的支付系统历史上最常见的故障模式是什么？"（跨文档主题）
❌ "张三、李四和王五三个团队在过去一年都参与了哪些项目？"（多跳关系推理）
❌ "整个技术文档库中，和性能优化相关的核心主题是什么？"（全局总结）
```

GraphRAG 的核心创新：**从"文档检索"升级为"知识图谱推理"**。

### 7.2 GraphRAG 工作原理

Microsoft GraphRAG 的处理流程：

```
第一阶段：知识图谱构建（离线，耗时耗资源）
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  文档                                                        │
│  ├── TextUnit（切分）                                        │
│  │     ↓ LLM 提取                                           │
│  ├── 实体（Entity）：人、组织、系统、概念                      │
│  ├── 关系（Relationship）：A 依赖 B、C 集成了 D              │
│  └── 关键声明（Claims）：重要结论和事实                       │
│         ↓ 聚类算法（Leiden）                                  │
│  社区（Communities）：相关实体的自然分组                       │
│         ↓ LLM 摘要                                           │
│  多层级社区摘要（Level 0/1/2/...）                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

第二阶段：查询（在线）
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  用户查询                                                    │
│       ↓                                                     │
│  Global Search：扫描所有社区摘要 → 全局主题问题               │
│  Local Search：实体向量 + 关系图遍历 → 具体实体问题           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 GraphRAG vs 传统 RAG 对比

| 维度 | 传统向量 RAG | GraphRAG |
|------|-------------|---------|
| **数据结构** | 向量块（孤立） | 实体-关系图 + 社区 |
| **检索方式** | 相似度匹配 | 图遍历 + 社区摘要 |
| **擅长问题** | 具体事实查询 | 主题分析、关系推理、全局总结 |
| **建索引成本** | 低（embedding） | 高（大量 LLM 调用提取实体） |
| **查询速度** | 快 | 较慢（图遍历） |
| **可解释性** | 低 | 高（可追踪实体和关系） |

### 7.4 Microsoft GraphRAG 快速上手

```bash
# 安装
pip install graphrag

# 初始化项目
mkdir graphrag_project && cd graphrag_project
graphrag init --root .

# 编辑配置（.env 和 settings.yaml）
# 设置 LLM（支持 OpenAI Compatible API，可用 Ollama）
```

```yaml
# settings.yaml 关键配置
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: qwen3:8b
  api_base: http://localhost:11434/v1  # Ollama API

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: nomic-embed-text
    api_base: http://localhost:11434/v1
```

```bash
# 将文档放入 ./input/ 目录
# 运行索引（会消耗较多 LLM 调用，先用小数据集测试！）
graphrag index --root .

# 查询
graphrag query --root . --method global "支付系统的主要技术挑战是什么？"
graphrag query --root . --method local "Westpac FTH 系统"
```

### 7.5 GraphRAG 的三种检索模式

#### Global Search（全局搜索）

适用于：**"整体上/宏观上…是什么？"** 类问题

扫描所有社区摘要，聚合全局知识，生成宏观分析。

```python
from graphrag.query.api import global_search

result = await global_search(
    config=graphrag_config,
    nodes=entities,
    entities=entities,
    community_reports=community_reports,
    query="整个技术文档库中，和性能优化相关的核心主题是什么？"
)
```

#### Local Search（本地搜索）

适用于：**"关于 X 的具体信息"** 类问题

从特定实体出发，沿关系图遍历，获取丰富上下文。

```python
from graphrag.query.api import local_search

result = await local_search(
    config=graphrag_config,
    nodes=entities,
    entities=entities,
    relationships=relationships,
    text_units=text_units,
    query="FTH 系统与 Finacle API 之间的集成方式"
)
```

#### Basic Search（基础搜索）

退回到标准向量检索，适用于简单的事实性查询。

### 7.6 GraphRAG 的实体-关系示例

对于支付系统文档，GraphRAG 会提取：

```
实体（Entities）：
  - FTH（Funds Transfer Hub）[系统]
  - ISO 20022 [标准]
  - Finacle API [系统]
  - AM04 [错误码]
  - Apache Camel [框架]

关系（Relationships）：
  FTH → 实现 → ISO 20022
  FTH → 集成 → Finacle API
  FTH → 使用 → Apache Camel
  AM04 → 属于 → ISO 20022 错误码体系
  Finacle API → 提供 → IMT支付服务

社区（Communities）：
  Community 1：FTH + Finacle API + IBM MQ（支付处理核心）
  Community 2：ISO 20022 + 错误码体系（消息标准）
  Community 3：Apache Camel + Spring + Java（技术框架）
```

### 7.7 GraphRAG 的轻量替代：LightRAG

Microsoft GraphRAG 建索引开销大（大量 LLM 调用提取实体）。LightRAG 提供更轻量的图 RAG 方案：

```bash
pip install lightrag-hku

# 使用
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

rag = LightRAG(
    working_dir="./lightrag_cache",
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen3:8b",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text")
    ),
)

# 插入文档
with open("your_document.txt") as f:
    rag.insert(f.read())

# 查询（支持多种模式）
result = rag.query("你的问题", param=QueryParam(mode="hybrid"))
# mode: "naive" | "local" | "global" | "hybrid"
```

### 7.8 何时选择 GraphRAG

**选 GraphRAG 的场景：**
- 文档之间有大量实体交叉引用（如技术架构文档、法律合同）
- 需要回答"整体趋势/主题"类问题
- 需要多跳推理（A→B→C 类型的关系链）
- 需要高度可解释的检索路径

**暂不选 GraphRAG 的场景：**
- 预算有限（建图成本高）
- 文档更新频繁（重建图代价大）
- 只需要简单的事实检索

---

### 7.9 Claude Code 提示词：GraphRAG 实践

```
在 07_graphrag/ 目录下创建 GraphRAG 学习项目：
1. setup_graphrag.py：初始化 Microsoft GraphRAG，
   配置使用 Ollama（qwen3:8b + nomic-embed-text）作为后端
2. build_index.py：对 ./sample_docs/ 下的 5 篇技术文档构建知识图谱
3. query_demo.py：分别演示 global search 和 local search，
   展示同一问题在两种模式下答案的差异
4. visualize_graph.py：用 networkx + matplotlib 可视化提取的实体关系图
5. compare_rag_vs_graphrag.py：用5个需要跨文档推理的问题，
   对比传统 RAG 和 GraphRAG 的答案质量
```

---

## 8. RAG 评估体系

没有评估就没有改进。以下是完整的 RAG 评估框架。

### 8.1 RAGAS 核心指标

```
RAG 评估指标分三层：

检索质量                 生成质量               端到端质量
─────────              ─────────              ─────────
Context Precision       Faithfulness           Answer Correctness
（精确率）              （忠实度）             （正确性）
Context Recall          Answer Relevancy
（召回率）              （相关性）
```

| 指标 | 含义 | 计算方式 | 目标 |
|------|------|---------|------|
| **Faithfulness** | 答案是否完全基于检索到的上下文（不幻觉） | LLM判断答案中每个陈述是否有上下文支撑 | >0.8 |
| **Answer Relevancy** | 答案是否真正回答了问题 | Embed答案，与问题的相似度 | >0.7 |
| **Context Precision** | 检索到的块中有多少是真正有用的 | 有用块数/总检索块数 | >0.7 |
| **Context Recall** | 正确答案中的信息有多少被检索到了 | 需要 ground truth answer | >0.8 |
| **Answer Correctness** | 与标准答案的一致程度 | 语义相似度 + 事实匹配 | >0.7 |

### 8.2 RAGAS 代码实现

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# 准备评估数据集
eval_dataset = Dataset.from_list([
    {
        "question": "AM04 错误码的含义是什么？",
        "answer": rag_chain.invoke("AM04 错误码的含义是什么？"),
        "contexts": [d.page_content for d in retriever.invoke("AM04 错误码")],
        "ground_truth": "AM04 错误码表示余额不足（Insufficient Funds），..."
    },
    # ... 更多测试案例
])

# 用本地 Ollama 模型做评估（LLM-as-Judge）
evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen3:8b", num_ctx=8192))
evaluator_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=evaluator_llm,
    embeddings=evaluator_emb,
)

print(results)
results.to_pandas().to_csv("ragas_results.csv")
```

### 8.3 RAG 问题诊断指南

```
评估分数低 → 诊断流程：

Faithfulness 低（<0.6）：
→ 模型在"编造"，检索到的内容和答案不一致
→ 解决：降低 temperature（0.1），加强 Prompt 约束，检查 num_ctx 是否太小

Context Precision 低（<0.5）：
→ 检索到了太多不相关的块
→ 解决：减少 k 值，加 Reranker，优化 chunk_size

Context Recall 低（<0.6）：
→ 检索漏掉了关键信息
→ 解决：增大 k 值，用 Hybrid Search，换更好的 Embedding 模型，
        尝试 Multi-Query 检索

Answer Relevancy 低（<0.6）：
→ 答案偏题，没有直接回答问题
→ 解决：优化 Prompt 模板，加强指令（"直接回答问题，不要绕弯"）
```

---

## 9. 学习路线总结

### 9.1 按能力等级划分

```
Level 1（入门，2周）：
  ✓ Ollama 本地部署 Qwen3
  ✓ 基础 RAG：PDF→Chunk→Embed→ChromaDB→问答
  ✓ 理解 num_ctx、temperature 等关键参数

Level 2（进阶，2-3周）：
  ✓ 掌握 5 种 Chunking 策略，知道何时选哪种
  ✓ 理解 Embedding 模型差异，会做基准测试
  ✓ 实现 Reranking + Hybrid Search
  ✓ 跑通 RAGAS 评估框架

Level 3（高级，3-4周）：
  ✓ Multi-Query + HyDE + 对话式 RAG
  ✓ 权限控制 + 增量更新 + 可观测性
  ✓ 了解企业 RAG 架构设计

Level 4（专家，持续）：
  ✓ GraphRAG：Microsoft GraphRAG + LightRAG
  ✓ 知识图谱构建和查询
  ✓ 针对特定领域微调 Embedding 模型
  ✓ Agentic RAG（LangGraph 编排）
```

### 9.2 配套项目建议

| 项目 | 技术点 | 与你工作的关联 |
|------|--------|--------------|
| ISO 20022 文档问答系统 | 基础RAG + Hybrid Search | Westpac FTH 规范查询 |
| 支付错误码知识库 | Chunking + Reranking | AM04/错误处理知识沉淀 |
| 技术文档跨文档分析 | GraphRAG | 架构关系理解 |
| Eduacan NZQA 题库 RAG | 对话式RAG + 评估 | 学生问答场景 |

### 9.3 核心参考资源

| 资源 | 链接 | 重点 |
|------|------|------|
| LangChain RAG 文档 | https://python.langchain.com/docs/use_cases/question_answering/ | 实现参考 |
| Microsoft GraphRAG | https://microsoft.github.io/graphrag/ | GraphRAG 官方 |
| LightRAG | https://github.com/HKUDS/LightRAG | 轻量图RAG |
| RAGAS 文档 | https://docs.ragas.io | 评估框架 |
| MTEB 排行榜 | https://huggingface.co/spaces/mteb/leaderboard | Embedding 选型 |
| Weaviate RAG 博客 | https://weaviate.io/blog | Chunking 最佳实践 |

---

*文档版本：v1.0 | 2026年3月 | 覆盖 Naive RAG → Advanced RAG → Enterprise RAG → GraphRAG*
