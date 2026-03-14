# 高级 RAG 技术深度指南
## 多查询检索（Multi-Query Retrieval）· 重排序（Reranking）· HyDE

> 阶段：Week 3-4 | 前置：基础 RAG 管道已跑通
>
> 这三项技术是从"能用"到"好用"的核心跨越，解决基础 RAG 的三个根本缺陷。

---

## 目录

1. [为什么基础 RAG 不够用](#1-为什么基础-rag-不够用)
2. [Multi-Query Retrieval：多查询检索](#2-multi-query-retrieval多查询检索)
3. [Reranking：重排序](#3-reranking重排序)
4. [HyDE：假设文档嵌入](#4-hyde假设文档嵌入)
5. [三者组合使用](#5-三者组合使用)
6. [评估与调试](#6-评估与调试)
7. [完整项目实战](#7-完整项目实战)

---

## 1. 为什么基础 RAG 不够用

基础 RAG 的流程看似完整，但在实际使用中有三个根本缺陷（Fundamental Flaws）：

### 缺陷一：单一视角的查询（Single Perspective Query）

用户提问的措辞是随机的。同一个意思，不同人会用完全不同的词表达：

```
同一个问题的不同表达：
  "ISO 20022 支付失败的原因是什么？"
  "为什么会出现支付拒绝？"
  "pacs.002 报文中错误码有哪些？"
  "FTH 系统里交易被 reject 的场景"

→ 这四个问题语义相似，但向量距离差距很大
→ 只用一个查询，极可能漏掉关键文档
```

**根本原因（Root Cause）：** 向量相似度（Vector Similarity）匹配的是措辞的相似性，不完全是语义的等价性（Semantic Equivalence）。用户的问法和文档的写法天然存在"语义鸿沟（Semantic Gap）"。

### 缺陷二：粗糙的相关性排序（Coarse Relevance Ranking）

向量检索（Vector Retrieval）用的是 Bi-Encoder（双编码器）——查询和文档分别独立编码为向量，然后计算余弦相似度（Cosine Similarity）。这个过程**快**，但精度（Precision）有限：

```
Bi-Encoder 的局限：
  Query: "支付超时的解决方案"
  → 变成向量 q
  
  Doc1: "支付超时会触发 AM04 错误" → 向量 d1，相似度 0.82
  Doc2: "超时处理机制：重试、回滚、告警" → 向量 d2，相似度 0.79
  Doc3: "网络延迟导致支付超时的根本原因" → 向量 d3，相似度 0.81

→ 按相似度排：Doc1 > Doc3 > Doc2
→ 但对"解决方案"这个需求，Doc2 才是最有用的！
→ 向量距离看不懂"解决方案"和文档内容的深层匹配
```

### 缺陷三：查询与文档的形式不对称（Query-Document Asymmetry）

```
用户查询（问题形式）：    "payment timeout causes?"
文档内容（答案形式）：    "The FTH system implements a 30-second timeout 
                          mechanism. When exceeded, the transaction is 
                          automatically rolled back and an AM04 error 
                          code is generated..."

→ 问题是简短、模糊、发散的
→ 答案是详细、具体、聚焦的
→ 两者在向量空间中距离较远，检索可能失败
```

**这三个缺陷，分别对应三种解决方案：**

| 缺陷 | 解决技术 | 核心思路 |
|------|---------|---------|
| 单一视角 | Multi-Query Retrieval | 生成多个查询角度，扩大覆盖 |
| 粗糙排序 | Reranking（Cross-Encoder） | 精细化重新打分 |
| 形式不对称 | HyDE | 先生成"答案形式"再检索 |

---

## 2. Multi-Query Retrieval：多查询检索

### 2.1 核心原理

让 LLM 把用户的一个问题**改写（Rewrite）成多个不同角度的查询**，每个查询独立检索，最后把结果合并去重（Deduplication）。

```
原始查询：
  "ISO 20022 支付失败的原因"

LLM 改写后：
  查询1: "ISO 20022 支付失败的原因"              ← 原问题
  查询2: "pacs.002 报文中的错误码有哪些类型"      ← 技术角度
  查询3: "支付系统中导致交易被拒绝的常见场景"     ← 业务角度
  查询4: "FTH 系统处理失败交易的机制是什么"       ← 系统角度

每个查询分别检索 Top-3：
  查询1 → Doc[A, B, C]
  查询2 → Doc[B, D, E]
  查询3 → Doc[A, F, G]
  查询4 → Doc[C, H, I]

去重合并 → Doc[A, B, C, D, E, F, G, H, I]（最多12个，去重后通常7-9个）
```

**召回率（Recall）提升原理：** 多个查询从不同"方向"接近同一知识点，就像用多束光照射一个空间，覆盖面（Coverage）更广，漏网之鱼更少。

### 2.2 基础实现

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import logging

# 开启日志，可以看到 LLM 生成了哪些查询（调试用）
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# 初始化组件
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 用高 temperature 鼓励多样化改写
llm = ChatOllama(model="qwen3:8b", temperature=0.7)

# 构建 Multi-Query Retriever
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

# 使用
docs = multi_retriever.invoke("ISO 20022 支付失败的原因是什么？")
print(f"检索到 {len(docs)} 个文档块")
```

**日志输出示例（开启 logging 后）：**
```
INFO: Generated queries: [
  'ISO 20022 消息标准中支付失败的触发条件有哪些？',
  '支付系统中 pacs.002 报文的错误码分类',
  'FTH 支付网关拒绝交易的原因和处理流程'
]
```

### 2.3 自定义查询生成 Prompt

默认的改写 prompt 是英文的，中文场景建议自定义：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

# 自定义输出解析器（Output Parser）：把 LLM 输出的多行文本解析成列表
class LineListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        lines = text.strip().split("\n")
        # 过滤空行和序号前缀（"1. ", "- " 等）
        return [
            line.strip().lstrip("0123456789.-）) ").strip()
            for line in lines
            if line.strip() and len(line.strip()) > 5
        ]

# 中文改写 Prompt（Query Rewriting Prompt）
QUERY_PROMPT = ChatPromptTemplate.from_template("""
你是一个 AI 助手，负责生成多个搜索查询来检索相关文档。

原始问题：{question}

请从 3 个不同角度改写这个问题，生成 3 个搜索查询：
1. 保留原始意图，但换一种措辞
2. 从技术/实现角度提问
3. 从业务/场景角度提问

每行输出一个查询，不要有序号或前缀，直接输出问题内容。
""")

# 组合成完整的查询生成链（Query Generation Chain）
query_chain = QUERY_PROMPT | llm | LineListOutputParser()

# 测试
queries = query_chain.invoke({"question": "支付超时如何处理？"})
for q in queries:
    print(f"  → {q}")
```

**输出示例：**
```
  → 支付交易超时后系统会触发什么操作？
  → timeout 错误在 pacs.002 报文中对应哪个错误码？
  → 用户支付等待时间过长时的业务处理流程是什么？
```

### 2.4 手动实现（更灵活的版本）

LangChain 的封装（Abstraction）有时不够灵活，手动实现可以精确控制每一步：

```python
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores.utils import maximal_marginal_relevance

def multi_query_retrieval(question: str, vectorstore, llm, n_queries=3, k_per_query=3):
    """
    手动实现 Multi-Query 检索
    
    Args:
        question: 原始用户问题
        vectorstore: 向量数据库
        llm: 用于生成改写查询的 LLM
        n_queries: 生成几个改写查询
        k_per_query: 每个查询检索几个文档
    
    Returns:
        去重后的文档列表
    """
    
    # Step 1: 生成改写查询（Query Generation）
    prompt = f"""将以下问题改写成 {n_queries} 个不同角度的搜索查询。
    
原始问题：{question}

要求：
- 每行一个查询
- 覆盖不同角度（措辞、技术、业务）
- 直接输出查询，不要序号

查询列表："""
    
    response = llm.invoke(prompt)
    generated_queries = [
        line.strip() for line in response.content.strip().split("\n")
        if line.strip() and len(line.strip()) > 3
    ][:n_queries]
    
    # 包含原始问题（Original Query）
    all_queries = [question] + generated_queries
    print(f"\n🔍 生成的查询（共 {len(all_queries)} 个）：")
    for q in all_queries:
        print(f"   - {q}")
    
    # Step 2: 每个查询独立检索（Independent Retrieval per Query）
    all_docs = []
    seen_ids = set()
    
    for query in all_queries:
        docs = vectorstore.similarity_search(query, k=k_per_query)
        for doc in docs:
            # 用内容哈希（Content Hash）去重（Deduplication）——避免同一文档块被多次返回
            doc_id = hash(doc.page_content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    
    print(f"\n📄 检索结果：{len(all_docs)} 个唯一文档块（去重后）")
    return all_docs


# 使用
docs = multi_query_retrieval(
    question="FTH 系统中支付失败的常见原因",
    vectorstore=vectorstore,
    llm=llm,
    n_queries=3,
    k_per_query=3
)
```

### 2.5 ⚠️ 关键注意事项

**`temperature` 要设高（0.7-0.9）**，不然 LLM 生成的查询高度相似，失去多样性（Diversity）的意义：

```python
# ❌ 错误：低 temperature 导致查询雷同
llm_low = ChatOllama(model="qwen3:8b", temperature=0.1)
# 可能生成：
# "ISO 20022 支付失败原因"
# "ISO 20022 支付失败的原因是什么"  ← 几乎一样！
# "ISO 20022 支付为什么失败"         ← 还是一样！

# ✅ 正确：高 temperature 产生多样化查询
llm_high = ChatOllama(model="qwen3:8b", temperature=0.7)
# 生成：
# "pacs.002 报文错误码分类"
# "支付网关拒绝交易的业务场景"
# "FTH 系统失败处理机制"
```

**每个查询的 `k`（检索数量）不要太大**，避免最终候选文档池（Candidate Pool）过大：

```python
# 推荐：3-4个查询，每个 k=3，最终约 7-10 个不重复文档
# 不推荐：5个查询 × k=10 = 潜在 50 个文档 → 塞满 context window
```

**LLM 调用次数（API Calls）= 1（生成查询）**，`n_queries` 次向量检索不消耗 LLM token，只是向量距离计算（Vector Distance Calculation）。

---

## 3. Reranking：重排序

### 3.1 Bi-Encoder vs Cross-Encoder

理解 Reranking（重排序）必须先理解两种编码架构（Encoding Architecture）的本质区别：

```
Bi-Encoder（双编码器）— 向量检索阶段：

  Query: "支付超时"    →  embed  →  [0.2, -0.8, 0.5, ...]
  Doc1:  "AM04错误..."  →  embed  →  [0.3, -0.7, 0.4, ...]
  Doc2:  "超时重试..."  →  embed  →  [0.1, -0.9, 0.6, ...]

  相似度 = cosine(q_vec, doc_vec)   ← 向量之间的距离
  
  优点：极快（O(1) 查询），可预计算文档向量（Pre-compute Document Vectors）
  缺点：Query 和 Doc 独立编码（Independent Encoding），无法捕捉两者之间的细微交互（Fine-grained Interaction）

───────────────────────────────────────────────────────────

Cross-Encoder（交叉编码器）— Reranking 阶段：

  [Query + Doc1] → 整体输入到模型 → 相关性分数 0.91
  [Query + Doc2] → 整体输入到模型 → 相关性分数 0.87
  [Query + Doc3] → 整体输入到模型 → 相关性分数 0.43

  模型同时看到 Query 和 Doc，能理解两者的交互关系
  
  优点：精度（Precision）远高于 Bi-Encoder
  缺点：每对 (query, doc) 都要单独推理（Individual Inference），不能预计算，速度慢
```

**这就是为什么用两阶段检索（Two-stage Retrieval）：**

```
第一阶段（召回阶段，Recall Stage）：Bi-Encoder 快速从百万文档中粗选 Top-20
第二阶段（精排阶段，Reranking Stage）：Cross-Encoder 对 Top-20 精细打分，选出 Top-5

总代价：20次 Cross-Encoder 推理（可接受）而不是百万次
```

### 3.2 Reranking 提升有多大？

以下是典型场景的数据对比（基于公开 benchmark）：

```
测试：10个技术文档查询，评估 Top-5 命中率

纯向量检索（k=5）：       62%
向量检索（k=20）：         71%  ← 多取了文档，但精度下降
向量检索 + Reranking：    84%  ← 先宽后精，显著提升

→ Reranking 是 RAG 质量提升最显著的单一优化手段
```

### 3.3 主流 Reranker 模型对比

| 模型 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| `BAAI/bge-reranker-v2-m3` | 本地，开源 | 多语言，中英文都强 | **首选推荐** |
| `BAAI/bge-reranker-base` | 本地，开源 | 轻量，速度快 | 低资源环境 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 本地，开源 | 英文最优，极快 | 纯英文场景 |
| `Cohere Rerank` | 云端 API | 质量最高 | 生产环境，预算充足 |
| `Jina Reranker v2` | 本地/云端 | 多语言，支持代码 | 代码文档场景 |

### 3.4 基础实现（LangChain 方式）

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 加载 Cross-Encoder 模型（首次会下载，约 500MB）
reranker_model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3"
)

# Reranker 压缩器：从候选文档中选出 top_n 个
compressor = CrossEncoderReranker(
    model=reranker_model,
    top_n=5   # 重排后保留 5 个
)

# 基础检索器（宽松，先检索更多候选）
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 20}  # 先取 20 个
)

# 组合：先粗检索，再精排
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 使用（和普通 retriever 完全相同的接口）
docs = reranking_retriever.invoke("支付超时的处理方案")
print(f"最终返回 {len(docs)} 个重排后的文档")
```

### 3.5 手动实现（带分数可视化）

LangChain 封装隐藏了分数，手动实现可以看到排序前后的对比，调试时很有帮助：

```python
from sentence_transformers import CrossEncoder
import numpy as np

class ManualReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)
        print(f"✅ Reranker 加载完成：{model_name}")
    
    def rerank(self, query: str, docs: list, top_n: int = 5, show_scores=True):
        """
        对检索结果重排序
        
        Args:
            query: 用户查询
            docs: 向量检索返回的文档列表
            top_n: 最终保留几个
            show_scores: 是否打印分数对比
        
        Returns:
            重排后的文档列表（按相关性降序）
        """
        if not docs:
            return []
        
        # 构造 [query, doc] 对
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Cross-Encoder 打分（返回相关性分数）
        scores = self.model.predict(pairs)
        
        # 组合文档和分数，排序
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        if show_scores:
            print(f"\n📊 Reranking 分数（查询：{query[:30]}...）")
            print(f"{'排名':<5} {'分数':<8} {'内容预览'}")
            print("-" * 60)
            for i, (doc, score) in enumerate(doc_score_pairs):
                marker = "✅" if i < top_n else "❌"
                preview = doc.page_content[:40].replace('\n', ' ')
                print(f"{marker} #{i+1:<3} {score:.4f}  {preview}...")
        
        # 返回 top_n 个文档
        return [doc for doc, _ in doc_score_pairs[:top_n]]


# 使用示例
reranker = ManualReranker()

# Step 1: 宽松检索（取更多候选）
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
candidate_docs = base_retriever.invoke("支付超时的处理方案")

# Step 2: 精排
final_docs = reranker.rerank(
    query="支付超时的处理方案",
    docs=candidate_docs,
    top_n=5,
    show_scores=True
)
```

**输出示例：**
```
📊 Reranking 分数（查询：支付超时的处理方案...）
────────────────────────────────────────────────────────────
排名  分数     内容预览
────────────────────────────────────────────────────────────
✅ #1   0.9823  超时处理机制：当支付请求超过30秒未收到响应...
✅ #2   0.9156  系统检测到超时后，自动触发回滚流程并生成告警...
✅ #3   0.8742  重试策略：第一次超时后等待5秒重试，最多3次...
✅ #4   0.7834  AM04 错误码在超时场景下的具体含义和处理...
✅ #5   0.6921  支付超时的监控指标和告警阈值设置...
❌ #6   0.3412  ISO 20022 消息格式规范第三章...
❌ #7   0.2891  Finacle API 接口文档...
...（15个被过滤掉的低分文档）
```

**清楚看到：** 排名 6 以后的文档相关性分数断崖式下降，Reranker 有效过滤了噪音。

### 3.6 与 Multi-Query 组合使用

Reranker 和 Multi-Query 是天然的搭档——Multi-Query 扩大召回，Reranker 保证精度：

```python
def multi_query_with_reranking(
    question: str,
    vectorstore,
    llm,
    reranker,
    n_queries=3,
    k_per_query=10,   # 注意：这里每个查询取更多候选
    final_top_n=5
):
    """
    Multi-Query + Reranking 组合检索
    """
    # Step 1: Multi-Query 扩大召回
    all_docs = multi_query_retrieval(
        question=question,
        vectorstore=vectorstore,
        llm=llm,
        n_queries=n_queries,
        k_per_query=k_per_query
    )
    
    print(f"\n🔄 进入 Reranking 阶段（候选文档：{len(all_docs)} 个）")
    
    # Step 2: Reranker 精排
    final_docs = reranker.rerank(
        query=question,
        docs=all_docs,
        top_n=final_top_n,
        show_scores=True
    )
    
    return final_docs
```

### 3.7 ⚠️ 关键注意事项

**`top_n` 不能大于候选文档数：**

```python
# ❌ 错误：base_retriever 只检索 5 个，却要 top_n=10
compressor = CrossEncoderReranker(model=model, top_n=10)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ 正确：候选数 >> top_n
compressor = CrossEncoderReranker(model=model, top_n=5)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
```

**Reranker 推理（Inference）有延迟（Latency）**，20 个文档约需 0.5-2 秒（CPU），生产环境（Production）注意超时设置。

**`bge-reranker-v2-m3` 有 512 token 输入限制（Input Length Limit）**，超长文档块会截断（Truncation），`chunk_size` 不要超过 400 token。

---

## 4. HyDE：假设文档嵌入

### 4.1 核心原理：以答找答

HyDE（Hypothetical Document Embeddings，假设文档嵌入）是解决"查询-文档形式不对称（Query-Document Asymmetry）"问题最优雅的方案。

**直觉理解：**

```
传统向量检索：
  用户问：  "payment timeout?"           → 短、模糊、问句形式
  文档内容："The system implements a 30-second timeout..."  → 长、具体、陈述形式
  
  向量距离：问句向量 ≠ 答案向量，检索可能失败

HyDE：
  用户问：  "payment timeout?"
       ↓ LLM 生成假设答案
  假设文档："Payment timeouts in FTH occur when the Finacle API 
            does not respond within 30 seconds. The system then 
            triggers an AM04 error code and initiates rollback..."
       ↓ Embed 假设文档
  假设文档向量 ≈ 真实文档向量（两者都是答案形式！）
  
  → 用假设答案去找真实答案，形式一致，距离更近
```

**为什么有效（Why It Works）：** LLM 生成的假设文档即使内容有误，但其**写作风格（Writing Style）和词汇选择（Vocabulary）**与真实文档高度相似，因此在向量空间（Vector Space）中更接近真实文档。

### 4.2 工作流程

```
[用户查询]
    │
    ▼
[LLM 生成假设文档]       ← 不在乎答案是否正确，关注格式和词汇
    │
    ▼
[Embed 假设文档]         ← 得到"答案形式"的向量
    │
    ▼
[用假设向量检索真实文档]  ← 与真实文档更匹配
    │
    ▼
[用真实文档 + 原始查询回答]  ← 用检索到的真实内容生成最终答案
```

**⚠️ 关键点（Key Point）：** 用于生成最终答案的是**真实检索到的文档（Retrieved Documents）**，不是 LLM 生成的假设文档（Hypothetical Document）。假设文档只用于向量检索（Vector Retrieval）这一步。

### 4.3 基础实现

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# 假设文档生成 Prompt（关键！）
HYDE_PROMPT = ChatPromptTemplate.from_template("""
请根据以下问题，写一段假设性的技术文档段落来回答这个问题。
要求：
- 使用专业技术文档的写作风格
- 200字左右
- 使用问题相关的专业术语
- 内容可以不完全正确，但风格要像真实文档
- 直接输出段落，不要有任何前言

问题：{question}

技术文档段落：
""")

def build_hyde_retriever(vectorstore, llm, k=5):
    """
    构建 HyDE 检索器
    """
    embeddings = vectorstore._embedding_function
    
    # 假设文档生成链（Hypothetical Document Generation Chain）
    hyde_chain = HYDE_PROMPT | llm | StrOutputParser()
    
    def hyde_retrieve(question: str):
        # Step 1: 生成假设文档
        hypothetical_doc = hyde_chain.invoke({"question": question})
        print(f"\n📝 生成的假设文档（前100字）：")
        print(f"   {hypothetical_doc[:100]}...")
        
        # Step 2: Embed 假设文档（注意：embed 的是假设文档 Hypothetical Doc，不是原始问题 Original Query）
        hyde_embedding = embeddings.embed_query(hypothetical_doc)
        
        # Step 3: 用假设文档的向量（Hypothetical Vector）检索真实文档
        docs = vectorstore.similarity_search_by_vector(
            hyde_embedding,
            k=k
        )
        return docs
    
    return hyde_retrieve


# 使用
hyde_retriever = build_hyde_retriever(vectorstore, llm)
docs = hyde_retriever("ISO 20022 支付失败有哪些错误码？")
```

### 4.4 标准 LangChain 实现（HypotheticalDocumentEmbedder）

LangChain 提供了内置的 HyDE 支持：

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

base_embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:8b", temperature=0.5)

# 构建 HyDE Embeddings（替换原始 embeddings）
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    # num_generations: 生成几个假设文档取平均向量（默认1）
    # 多个假设文档取平均可以更稳定
)

# 用 HyDE embeddings 构建向量库
# 注意：入库时用普通 embeddings（文档只入库一次）
# 查询时用 HyDE embeddings（自动在内部生成假设文档）
vectorstore_hyde = Chroma(
    collection_name="rag_docs",
    embedding_function=hyde_embeddings,  # ← 只在查询时用 HyDE
    persist_directory="./chroma_db"
)

retriever = vectorstore_hyde.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("支付失败的错误码有哪些？")
```

> ⚠️ 注意：文档入库时不要用 HyDE embeddings，否则每个 chunk 都会触发 LLM 调用，代价极高。文档用普通 embeddings 入库，查询时换 HyDE。

### 4.5 多假设文档平均（进阶）

生成多个假设文档（Multiple Hypothetical Documents），取向量平均值（Average Embedding），结果更稳定（More Robust）：

```python
import numpy as np

def multi_hyde_retrieve(question: str, vectorstore, llm, n_hypothetical=3, k=5):
    """
    生成多个假设文档，取平均向量检索（更稳定）
    """
    embeddings = vectorstore._embedding_function
    
    hyde_prompt = f"""请用技术文档风格写一段回答以下问题的段落（150字左右）：
{question}
直接输出段落："""
    
    # 生成 n 个假设文档
    hypothetical_docs = []
    for i in range(n_hypothetical):
        doc = llm.invoke(hyde_prompt).content
        hypothetical_docs.append(doc)
        print(f"假设文档 {i+1}: {doc[:50]}...")
    
    # 分别 embed，取平均向量
    all_embeddings = [
        embeddings.embed_query(doc) for doc in hypothetical_docs
    ]
    avg_embedding = np.mean(all_embeddings, axis=0).tolist()
    
    # 用平均向量检索
    docs = vectorstore.similarity_search_by_vector(avg_embedding, k=k)
    return docs
```

### 4.6 HyDE 的适用场景与局限

**适合用 HyDE 的场景（When to Use HyDE）：**

```
✅ 用户问口语化问题，文档是专业技术语言
   用户："为什么我的转账一直转不过去？"
   文档："pacs.002 报文 ReasonCode AM04 表示账户余额不足..."

✅ 跨语言检索（问题中文，文档英文）
   用户（中文）："支付超时如何配置？"
   文档（英文）："timeout_threshold: The maximum wait time..."

✅ 知识型问答（答案有固定格式的文档）
   如：规范文档、API 文档、学术论文
```

**不适合用 HyDE 的场景（When NOT to Use HyDE）：**

```
❌ 事实性精确查询（HyDE 可能产生误导性假设）
   用户："AM04 的 HTTP 状态码是多少？"
   → LLM 可能生成错误的假设文档，检索方向跑偏

❌ 最新信息查询（LLM 不知道最新内容，假设文档质量差）
   用户："2026年3月的最新汇率是多少？"

❌ 文档库很小（< 100 个 chunk）
   → 向量匹配本来就容易，不需要 HyDE 的开销
```

### 4.7 ⚠️ 关键注意事项

**HyDE 会额外消耗一次 LLM 调用（Extra LLM Call）**，每次查询（Query）都要先生成假设文档，延迟（Latency）增加 0.5-2 秒。

**假设文档的 `temperature` 建议 0.3-0.5**，既要有变化性（Diversity），又不能太随机（Random）导致假设文档偏离主题（Off-topic）：

```python
# 生成假设文档用的 LLM（中温度）
hyde_llm = ChatOllama(model="qwen3:8b", temperature=0.5)

# 最终回答用的 LLM（低温度）
answer_llm = ChatOllama(model="qwen3:8b", temperature=0.1)
```

---

## 5. 三者组合使用

### 5.1 完整的高级 RAG 管道

三种技术不互斥（Non-exclusive），可以自由组合（Combine Freely），效果叠加（Additive Effect）：

```
[用户查询]
    │
    ├─→ Multi-Query（扩大召回）
    │       ├─ 查询变体1 → 检索 Top-10
    │       ├─ 查询变体2 → 检索 Top-10
    │       └─ 查询变体3 → 检索 Top-10
    │                       ↓
    │               去重合并（约 20-25 个文档）
    │
    ├─→ HyDE（提升匹配质量）—— 可选，用于难以检索的查询
    │
    └─→ Reranking（精排）
            ↓
        Top-5 最相关文档
            ↓
        [LLM 生成最终答案]
```

### 5.2 组合实现代码

```python
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder

class AdvancedRAGPipeline:
    """
    整合 Multi-Query + Reranking + HyDE 的高级 RAG 管道
    可以按需开关各个组件
    """
    
    def __init__(
        self,
        vectorstore_path: str = "./chroma_db",
        llm_model: str = "qwen3:8b",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        use_multi_query: bool = True,   # 多查询（Multi-Query Retrieval）
        use_hyde: bool = False,         # 假设文档嵌入（HyDE），默认关闭，有延迟
        use_reranking: bool = True,     # 重排序（Reranking）
    ):
        self.use_multi_query = use_multi_query
        self.use_hyde = use_hyde
        self.use_reranking = use_reranking
        
        # 初始化组件
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(
            collection_name="rag_docs",
            embedding_function=self.embeddings,
            persist_directory=vectorstore_path
        )
        self.llm = ChatOllama(model=llm_model, num_ctx=8192, temperature=0.1)
        self.query_llm = ChatOllama(model=llm_model, temperature=0.7)  # 查询生成（Query Generation）用高 temperature
        
        if use_reranking:
            self.reranker = CrossEncoder(reranker_model)
            print(f"✅ Reranker 加载：{reranker_model}")
        
        # RAG 回答 Prompt
        self.rag_prompt = ChatPromptTemplate.from_template("""
你是一个专业的技术文档问答助手。
请严格基于以下检索到的文档内容回答问题。
如果文档中没有相关信息，请明确说明"文档中未找到相关信息"。
不要编造文档中没有的内容。

检索到的文档：
{context}

问题：{question}

回答：""")

    def _generate_queries(self, question: str, n=3) -> list[str]:
        """Multi-Query：生成查询变体"""
        prompt = f"""将以下问题改写成{n}个不同角度的搜索查询（每行一个，直接输出，不要序号）：

问题：{question}

查询列表："""
        response = self.query_llm.invoke(prompt)
        queries = [
            line.strip() for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 3
        ][:n]
        return [question] + queries  # 包含原始问题

    def _hyde_embed(self, question: str):
        """HyDE：生成假设文档并返回其向量"""
        prompt = f"""请用技术文档风格写一段回答以下问题的段落（150字左右，直接输出）：
{question}"""
        hypo_doc = self.query_llm.invoke(prompt).content
        return self.embeddings.embed_query(hypo_doc), hypo_doc

    def _retrieve(self, question: str, k_per_query: int = 10) -> list:
        """检索阶段：支持 Multi-Query 和 HyDE"""
        all_docs = []
        seen = set()

        def add_docs(docs):
            for doc in docs:
                h = hash(doc.page_content)
                if h not in seen:
                    seen.add(h)
                    all_docs.append(doc)

        if self.use_multi_query:
            queries = self._generate_queries(question)
            print(f"\n🔍 Multi-Query 生成 {len(queries)} 个查询")
            for q in queries:
                docs = self.vectorstore.similarity_search(q, k=k_per_query)
                add_docs(docs)
        
        if self.use_hyde:
            hyde_vec, hypo_doc = self._hyde_embed(question)
            print(f"\n📝 HyDE 假设文档：{hypo_doc[:60]}...")
            docs = self.vectorstore.similarity_search_by_vector(hyde_vec, k=k_per_query)
            add_docs(docs)
        
        if not self.use_multi_query and not self.use_hyde:
            # 都不用，回退到基础检索
            docs = self.vectorstore.similarity_search(question, k=k_per_query)
            add_docs(docs)
        
        print(f"\n📄 候选文档：{len(all_docs)} 个（去重后）")
        return all_docs

    def _rerank(self, question: str, docs: list, top_n: int = 5) -> list:
        """重排序阶段"""
        if not self.use_reranking or len(docs) <= top_n:
            return docs[:top_n]
        
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Reranking 完成，Top-{top_n} 分数：" +
              ", ".join([f"{s:.3f}" for _, s in ranked[:top_n]]))
        
        return [doc for doc, _ in ranked[:top_n]]

    def query(self, question: str, k_per_query: int = 10, final_top_n: int = 5) -> str:
        """完整查询流程"""
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"配置：Multi-Query={self.use_multi_query}, "
              f"HyDE={self.use_hyde}, Reranking={self.use_reranking}")
        
        # Step 1: 检索
        candidate_docs = self._retrieve(question, k_per_query)
        
        # Step 2: 重排序
        final_docs = self._rerank(question, candidate_docs, final_top_n)
        
        # Step 3: 生成答案
        context = "\n\n---\n\n".join([
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in final_docs
        ])
        
        chain = self.rag_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        return answer


# ====== 使用示例 ======

# 标准配置（Multi-Query + Reranking）
pipeline = AdvancedRAGPipeline(
    use_multi_query=True,
    use_hyde=False,
    use_reranking=True,
)
answer = pipeline.query("FTH 系统中 AM04 错误码的处理流程是什么？")
print(f"\n✅ 答案：{answer}")

# 困难查询（口语化问题）：加上 HyDE
pipeline_hyde = AdvancedRAGPipeline(
    use_multi_query=True,
    use_hyde=True,     # 开启 HyDE
    use_reranking=True,
)
answer = pipeline_hyde.query("为什么我的转账一直失败？")
```

### 5.3 什么时候开启哪些组件

```
基础情况（90%的查询都够用）：
  ✅ Multi-Query = True
  ✅ Reranking = True
  ❌ HyDE = False

加上 HyDE 的情况：
  ✅ 用户习惯口语化提问
  ✅ 中文问题 + 英文文档
  ✅ 用户不了解专业术语

关闭 Multi-Query 的情况：
  ❌ 查询本身已经很精确（如：直接输入错误码 "AM04"）
  ❌ Groq 免费 tier 配额紧张（Multi-Query 多消耗 LLM 调用）
  ❌ 对延迟要求极高
```

---

## 6. 评估与调试

### 6.1 用 RAGAS 量化提升幅度（Quantify Improvement）

```python
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness
from datasets import Dataset

# 准备测试数据集（Test Dataset，10-20条有 ground truth 的问答对）
test_cases = [
    {
        "question": "AM04 错误码是什么意思？",
        "ground_truth": "AM04 表示账户余额不足（Insufficient Funds）",
    },
    {
        "question": "ISO 20022 支付失败的常见原因有哪些？",
        "ground_truth": "常见原因包括余额不足(AM04)、账号错误(AC01)、...",
    },
    # ...更多测试案例
]

def evaluate_pipeline(pipeline, test_cases):
    """评估 RAG 管道质量"""
    results = []
    for case in test_cases:
        q = case["question"]
        
        # 获取检索结果
        candidate_docs = pipeline._retrieve(q)
        final_docs = pipeline._rerank(q, candidate_docs)
        
        results.append({
            "question": q,
            "answer": pipeline.query(q),
            "contexts": [doc.page_content for doc in final_docs],
            "ground_truth": case["ground_truth"],
        })
    
    dataset = Dataset.from_list(results)
    scores = evaluate(dataset, metrics=[context_recall, context_precision, faithfulness])
    return scores

# 对比不同配置
print("=== 基础 RAG ===")
basic = AdvancedRAGPipeline(use_multi_query=False, use_hyde=False, use_reranking=False)
print(evaluate_pipeline(basic, test_cases))

print("\n=== Multi-Query + Reranking ===")
advanced = AdvancedRAGPipeline(use_multi_query=True, use_reranking=True)
print(evaluate_pipeline(advanced, test_cases))
```

### 6.2 快速调试检查清单（Debugging Checklist）

**Multi-Query 问题排查（Multi-Query Debugging）：**

```python
# 检查生成的查询是否真的多样
queries = pipeline._generate_queries("你的测试问题")
for q in queries:
    print(q)

# 如果查询都很相似：
# → 提高 query_llm 的 temperature（试试 0.8-0.9）
# → 修改 prompt，明确要求"从不同角度"

# 检查多查询的去重是否正常
all_docs = pipeline._retrieve("你的测试问题", k_per_query=5)
print(f"去重后文档数：{len(all_docs)}，应该 < n_queries × k_per_query")
```

**Reranking 问题排查（Reranking Debugging）：**

```python
# 如果 Reranking 后效果反而差，检查分数分布
pairs = [["你的问题", doc.page_content] for doc in docs]
scores = pipeline.reranker.predict(pairs)
print(f"分数范围：{min(scores):.3f} ~ {max(scores):.3f}")

# 如果所有分数都很低（< 0.3）：
# → 文档可能与问题完全不相关，是检索问题而非 Reranking 问题
# → 检查 chunk_size 和 embedding 模型
```

**HyDE 问题排查（HyDE Debugging）：**

```python
# 查看生成的假设文档质量
hyde_vec, hypo_doc = pipeline._hyde_embed("你的问题")
print("假设文档：")
print(hypo_doc)

# 如果假设文档方向跑偏：
# → 修改 hyde prompt，加更多约束
# → 降低 temperature（试试 0.3）
# → 提供更多领域上下文（在 prompt 中说明文档类型）
```

### 6.3 性能（Performance）vs 质量（Quality）权衡（Trade-off）

```
配置方案（Configuration）    延迟 Latency  质量提升 Quality  API 消耗 Cost
──────────────────────────────────────────────────────────────────
基础 RAG（Naive RAG）         0.5s          基准 Baseline    1次LLM
+ Multi-Query                1.5s          +15%             2次LLM
+ Reranking                  2.5s          +25%             2次LLM
+ HyDE                       3.5s          +10%             3次LLM
全部开启（All Enabled）       4.5s          +35%             3次LLM

→ Multi-Query + Reranking 是性价比（Cost-Effectiveness）最高的组合
→ HyDE 在特定场景有效，但不是默认必选
```

---

## 7. 完整项目实战

### 7.1 项目结构

```
05_advanced_rag/
├── ingest.py              # 文档入库（与基础 RAG 相同）
├── advanced_pipeline.py   # 核心管道（本文的 AdvancedRAGPipeline）
├── compare.py             # 对比不同策略的效果
├── evaluate.py            # RAGAS 评估
└── app.py                 # FastAPI 接口
```

### 7.2 compare.py：策略效果对比

```python
"""
对同一组问题，对比4种配置的检索质量
"""
import time
from advanced_pipeline import AdvancedRAGPipeline

TEST_QUESTIONS = [
    "AM04 错误码的处理方式是什么？",
    "FTH 系统与 Finacle API 的集成方式",
    "支付超时后系统的自动处理流程",
    "ISO 20022 中 pacs.002 报文的结构",
    "如何排查支付失败的根本原因？",
]

CONFIGS = {
    "基础 RAG": dict(use_multi_query=False, use_hyde=False, use_reranking=False),
    "Multi-Query": dict(use_multi_query=True,  use_hyde=False, use_reranking=False),
    "Reranking":   dict(use_multi_query=False, use_hyde=False, use_reranking=True),
    "MQ + Rerank": dict(use_multi_query=True,  use_hyde=False, use_reranking=True),
    "全部开启":    dict(use_multi_query=True,  use_hyde=True,  use_reranking=True),
}

for config_name, config in CONFIGS.items():
    pipeline = AdvancedRAGPipeline(**config)
    
    total_docs_retrieved = 0
    total_time = 0
    
    for q in TEST_QUESTIONS:
        start = time.time()
        candidate_docs = pipeline._retrieve(q)
        elapsed = time.time() - start
        total_docs_retrieved += len(candidate_docs)
        total_time += elapsed
    
    avg_docs = total_docs_retrieved / len(TEST_QUESTIONS)
    avg_time = total_time / len(TEST_QUESTIONS)
    
    print(f"\n{config_name}:")
    print(f"  平均检索文档数：{avg_docs:.1f}")
    print(f"  平均检索耗时：{avg_time:.2f}s")
```

### 7.3 Claude Code 提示词

```
在 05_advanced_rag/ 目录创建完整的高级 RAG 项目：

1. advanced_pipeline.py
   - 实现 AdvancedRAGPipeline 类
   - 支持开关控制：use_multi_query, use_hyde, use_reranking
   - LLM 用 Groq（ChatGroq）或本地 Ollama（可配置）
   - Embedding 用本地 nomic-embed-text
   - Reranker 用 BAAI/bge-reranker-v2-m3

2. compare.py
   - 对10个问题测试5种配置组合
   - 输出对比表格：配置名、平均候选文档数、重排后文档数、耗时
   - 用 rich 库美化输出

3. evaluate.py
   - 准备20条有 ground_truth 的测试数据
   - 用 RAGAS 评估 Context Recall, Context Precision, Faithfulness
   - 对比"基础RAG" vs "MQ + Reranking" 的分数差异
   - 输出 CSV 和可视化图表

4. app.py
   - FastAPI，POST /query 接口
   - 请求体包含：question, use_multi_query, use_hyde, use_reranking
   - 响应包含：answer, retrieved_docs_count, latency_ms

注意：
- num_ctx 设置 8192
- Reranker 首次运行会下载模型，加提示信息
- 所有配置通过环境变量或 config.py 管理
```

---

## 总结

三项技术解决三个不同问题，相互独立（Independent）又可以叠加（Stackable）：

```
Multi-Query Retrieval（多查询检索）
  解决：单一查询（Single Query）的召回率（Recall）不足
  代价：多 1 次 LLM 调用（约 0.5-1s）
  效果：Context Recall +10~15%
  推荐：默认开启（Default ON）

Reranking（Cross-Encoder，重排序）
  解决：向量检索（Vector Retrieval）的排序精度（Ranking Precision）不足
  代价：本地 Cross-Encoder 推理（约 0.5-1s）
  效果：Context Precision +15~20%，是单一优化（Single Optimization）中效果最显著的
  推荐：默认开启（Default ON）

HyDE（Hypothetical Document Embeddings，假设文档嵌入）
  解决：查询与文档的形式不对称（Query-Document Asymmetry）
  代价：多 1 次 LLM 调用（约 0.5-1s）
  效果：特定场景（口语化查询 Colloquial Query、跨语言 Cross-lingual）+10%
  推荐：按需开启（ON as needed）
```

**学习路径建议（Recommended Learning Path）：**
1. 先单独实现（Implement Independently）每项技术，理解其原理
2. 用 `compare.py` 脚本量化（Quantify）每项技术的实际效果
3. 根据自己的数据和场景（Use Case），决定最终组合（Final Configuration）
4. 用 RAGAS 评估最终配置，建立基准分数（Baseline Score）

---

*文档版本：v1.0 | 2026年3月 | 对应学习计划 Week 3-4*

---

## 8. Claude Code Prompts：生成学习代码

> 以下是针对本章每个知识点的 Claude Code 提示词（Prompt）。  
> 使用方式：直接把提示词粘贴给 Claude Code，它会在你的项目目录生成完整可运行的代码。

---

### 8.1 Multi-Query Retrieval 学习代码

```
在 05_advanced_rag/multi_query/ 目录下创建 Multi-Query 检索学习项目。

文件结构：
  multi_query_basic.py       # LangChain 封装版
  multi_query_custom.py      # 自定义 Prompt 版（中文改写）
  multi_query_manual.py      # 纯手动实现版（带去重逻辑）
  demo.py                    # 对比演示：单查询 vs 多查询的召回差异

技术要求：
- LLM：优先使用 Groq（from langchain_groq import ChatGroq），
        fallback 到 Ollama（from langchain_ollama import ChatOllama）
- Embedding：本地 OllamaEmbeddings(model="nomic-embed-text")
- 向量库：ChromaDB，persist_directory="./chroma_db"
- num_ctx=8192（Ollama 时必须设置）
- 查询生成的 temperature=0.7，回答生成的 temperature=0.1

multi_query_basic.py 要求：
- 使用 MultiQueryRetriever.from_llm()
- 开启 logging 显示生成的查询变体
- 函数签名：run_multi_query(question: str, k: int = 3) -> list[Document]

multi_query_custom.py 要求：
- 自定义中文 QueryPrompt，要求 LLM 从"措辞/技术/业务"三角度改写
- 实现 LineListOutputParser 解析多行输出
- 函数签名：run_custom_multi_query(question: str, n_queries: int = 3) -> list[Document]

multi_query_manual.py 要求：
- 完全手动实现，不依赖 LangChain 的 MultiQueryRetriever
- 用 hash(doc.page_content) 去重
- 打印每个查询变体和各自检索到的文档数
- 函数签名：manual_multi_query(question: str, n_queries: int = 3, k_per_query: int = 3) -> list[Document]

demo.py 要求：
- 准备 5 个测试问题（从技术文档场景出发，如错误码、系统集成、流程等）
- 对比输出：
  * 单一查询检索到的文档数 vs Multi-Query 检索到的文档数
  * 多查询额外找到了哪些文档（用 set difference 展示）
- 打印清晰的对比表格（用 rich 库或手动格式化）

注意：首次运行需要先调用 ingest.py 把测试 PDF 入库。
如果没有 PDF，请生成 10 个测试用的技术文档文本片段存入 ChromaDB。
```

---

### 8.2 Reranking 学习代码

```
在 05_advanced_rag/reranking/ 目录下创建 Reranking 重排序学习项目。

文件结构：
  reranker_basic.py          # LangChain ContextualCompressionRetriever 版
  reranker_manual.py         # 手动实现版（带分数可视化）
  reranker_benchmark.py      # 对比 3 种 Reranker 模型的速度和质量
  demo.py                    # 核心演示：重排前后 Top-5 对比

技术要求：
- Reranker 模型：BAAI/bge-reranker-v2-m3（from sentence_transformers import CrossEncoder）
- 向量库：ChromaDB（复用 multi_query 目录的 chroma_db，或重新入库）
- 基础检索 k=20，Reranking 后 top_n=5

reranker_basic.py 要求：
- 使用 LangChain 的 ContextualCompressionRetriever + CrossEncoderReranker
- 函数签名：rerank_retrieve(question: str, top_n: int = 5) -> list[Document]
- 包含模型加载时间的计时（首次下载提示用户等待）

reranker_manual.py 要求：
- 实现 ManualReranker 类，包含 rerank(query, docs, top_n, show_scores) 方法
- show_scores=True 时打印完整的排名表格（含 ✅/❌ 标记、分数、内容预览）
- 额外实现 compare_before_after(question, vectorstore) 方法：
  打印 Reranking 前后 Top-5 文档的对比（哪些文档排名上升/下降/被过滤）

reranker_benchmark.py 要求：
- 测试以下 3 个 Reranker 模型：
  1. BAAI/bge-reranker-v2-m3（多语言）
  2. BAAI/bge-reranker-base（轻量）
  3. cross-encoder/ms-marco-MiniLM-L-6-v2（英文快速）
- 对 5 个测试问题分别测试
- 输出对比表格：模型名、加载时间、单次推理时间、Top-1 文档内容预览
- 保存结果到 benchmark_results.csv

demo.py 要求：
- 选 3 个精心设计的测试问题，展示 Reranking 最有价值的场景：
  * 问题1：精确术语查询（如错误码），向量检索可能排错
  * 问题2：语义接近但意图不同的查询
  * 问题3：长问题，向量检索排序不稳定
- 每个问题展示：原始 Top-5 → Reranking 后 Top-5 → 差异分析

注意：
- Reranker 模型首次下载约 500MB，加注释提醒用户
- bge-reranker-v2-m3 的 token 限制是 512，chunk 超长时自动截断并打印警告
```

---

### 8.3 HyDE 学习代码

```
在 05_advanced_rag/hyde/ 目录下创建 HyDE 假设文档嵌入学习项目。

文件结构：
  hyde_basic.py              # 手动实现版（最清晰）
  hyde_langchain.py          # LangChain HypotheticalDocumentEmbedder 版
  hyde_multi.py              # 多假设文档平均向量版（进阶）
  hyde_vs_standard.py        # 对比：标准检索 vs HyDE 检索
  demo.py                    # 重点演示 HyDE 最有效的场景

技术要求：
- LLM：Groq（ChatGroq）或 Ollama（ChatOllama），假设文档生成 temperature=0.5
- Embedding：OllamaEmbeddings(model="nomic-embed-text")
- 向量库：ChromaDB

hyde_basic.py 要求：
- 实现 build_hyde_retriever(vectorstore, llm, k=5) 函数
  返回一个可调用对象 hyde_retrieve(question: str) -> list[Document]
- 打印生成的假设文档（前150字）供调试
- 清晰注释：假设文档只用于向量检索，不用于最终回答

hyde_langchain.py 要求：
- 使用 HypotheticalDocumentEmbedder.from_llm()
- 关键注释：入库用普通 embeddings，查询时才用 HyDE embeddings
- 演示如何在不重新入库的情况下切换到 HyDE 检索

hyde_multi.py 要求：
- 实现 multi_hyde_retrieve(question, vectorstore, llm, n_hypothetical=3, k=5)
- 生成 n_hypothetical 个假设文档
- 分别 embed，用 numpy 取平均向量 (np.mean(embeddings, axis=0))
- 打印每个假设文档的前 50 字，展示多样性

hyde_vs_standard.py 要求：
- 实现 compare_retrieval(question, vectorstore, llm) 函数
- 返回字典：{"standard": [docs], "hyde": [docs], "overlap": [docs], "hyde_unique": [docs]}
- 展示 HyDE 独有检索到的文档（standard 找不到但 HyDE 找到的）
- 设计 3 类对比问题：
  * 口语化问题（HyDE 有明显优势）
  * 专业术语问题（HyDE 优势不大）
  * 跨语言问题（中文问英文文档）

demo.py 要求：
- 选 2 个最能展示 HyDE 价值的场景：
  * 场景1：用户用非技术语言提问，文档是专业术语写的
    例："为什么我的转账一直转不过去" vs 文档："pacs.002 ReasonCode AM04..."
  * 场景2：中文提问，英文文档
- 每个场景展示：
  1. 生成的假设文档内容
  2. 标准检索 Top-3 结果
  3. HyDE 检索 Top-3 结果
  4. 哪种结果更相关的分析

注意：
- 如果没有合适的测试文档，在 demo.py 中生成 mock 文档并存入 ChromaDB
- HyDE 温度建议 0.3-0.5，在注释中说明原因
```

---

### 8.4 三者组合：AdvancedRAGPipeline

```
在 05_advanced_rag/ 目录下创建整合所有技术的高级 RAG 管道。

文件结构：
  pipeline.py                # AdvancedRAGPipeline 核心类
  config.py                  # 配置管理（API keys、模型选择、参数）
  compare_configs.py         # 对比5种配置组合的效果
  evaluate.py                # RAGAS 自动化评估
  app.py                     # FastAPI 接口（可选）

pipeline.py 要求：
- 实现 AdvancedRAGPipeline 类，构造函数参数：
  * vectorstore_path: str = "./chroma_db"
  * llm_provider: str = "groq"  # "groq" 或 "ollama"
  * llm_model: str = "llama-3.3-70b-versatile"  # Groq 模型
  * reranker_model: str = "BAAI/bge-reranker-v2-m3"
  * use_multi_query: bool = True
  * use_hyde: bool = False
  * use_reranking: bool = True
  * n_queries: int = 3
  * k_per_query: int = 10
  * final_top_n: int = 5

- 关键方法：
  * query(question: str) -> dict：返回 {answer, docs, latency_ms, config}
  * retrieve(question: str) -> list[Document]：只做检索不生成答案
  * explain(question: str) -> dict：展示每步的中间结果（调试用）

- 内部流程打印清晰的步骤日志（可通过 verbose=True/False 控制）

config.py 要求：
- 用 python-dotenv 管理 API keys
- 提供 DEFAULT_CONFIG、QUALITY_CONFIG、SPEED_CONFIG 三套预设配置
- DEFAULT_CONFIG：Multi-Query=True, Reranking=True, HyDE=False
- QUALITY_CONFIG：全部开启
- SPEED_CONFIG：全部关闭（基础 RAG）

compare_configs.py 要求：
- 测试 5 种配置：Naive / MultiQuery / Reranking / MQ+Rerank / All
- 每种配置对 10 个测试问题运行
- 收集指标：平均延迟、候选文档数、最终文档数
- 输出结果到 results/comparison.csv 和 results/comparison.png（matplotlib 柱状图）

evaluate.py 要求：
- 准备 15 个有 ground_truth 的问答对（手动编写，贴近实际使用场景）
- 用 RAGAS 评估 4 项指标：
  * context_recall（召回率）
  * context_precision（精确率）
  * faithfulness（忠实度）
  * answer_relevancy（答案相关性）
- 对比 Naive RAG 和 MQ+Reranking 的评估分数
- 输出 results/ragas_report.md（Markdown 格式报告）

app.py 要求（FastAPI）：
- POST /query
  请求体：{"question": str, "use_multi_query": bool, "use_hyde": bool, "use_reranking": bool}
  响应：{"answer": str, "source_docs": [{"content": str, "source": str}], "latency_ms": int}
- GET /health：检查所有组件（ChromaDB、Reranker、LLM）状态
- 包含 CORS 设置，支持本地前端调试

注意：
- 所有 API keys 通过 .env 文件管理，.env.example 作为模板提交
- requirements.txt 包含所有依赖：
  langchain langchain-groq langchain-ollama langchain-chroma
  sentence-transformers ragas datasets fastapi uvicorn python-dotenv
  matplotlib rich numpy
- README.md 包含完整的运行步骤
```

---

### 8.5 快速验证单个技术（10分钟上手）

如果只是想快速验证某项技术是否工作，用这个精简版 prompt：

```
写一个单文件的 Python 脚本 quick_test_multi_query.py，要求：

1. 创建 5 个测试文档（mock data，不需要真实 PDF）直接存入内存版 ChromaDB
2. 用 OllamaEmbeddings(model="nomic-embed-text") 做向量化
3. 对比同一个问题：单查询 vs Multi-Query 的检索结果差异
4. 打印清晰的对比输出
5. 整个脚本可以直接 python quick_test_multi_query.py 运行

Mock 文档内容用支付系统相关的技术描述（5条，每条100字左右）。
使用 Groq 的 llama-3.3-70b-versatile 作为 LLM（需要设置 GROQ_API_KEY 环境变量）。
```

类似地可以换成 `quick_test_reranking.py` 或 `quick_test_hyde.py`。

---

### 8.6 调试辅助工具

```
创建 05_advanced_rag/utils/debug_tools.py，包含以下调试工具函数：

1. visualize_query_similarity(queries: list[str], embeddings) -> None
   - 将多个查询向量用 UMAP 降维到 2D 并用 matplotlib 可视化
   - 直观展示 Multi-Query 生成的查询在向量空间中的分散程度

2. show_reranking_impact(question: str, before_docs: list, after_docs: list) -> None
   - 用 rich 表格对比重排前后的文档顺序
   - 用颜色高亮：绿色=排名上升，红色=被过滤，黄色=排名下降

3. compare_hyde_vectors(question: str, hypothetical_doc: str, embeddings) -> dict
   - 计算原始查询向量 vs 假设文档向量 vs 真实文档向量的余弦相似度
   - 验证 HyDE 是否真的让查询更接近文档

4. token_count(text: str, model: str = "gpt-3.5-turbo") -> int
   - 用 tiktoken 估算文本的 token 数
   - 帮助判断 chunk 是否超过 Reranker 的 512 token 限制

所有工具函数打印清晰的输出，适合在 Jupyter Notebook 中使用。
```

---

*文档版本：v2.0 | 2026年3月 | 新增 Claude Code Prompts 章节（Section 8）*

---

## 7. 完整项目实战

### 7.1 项目结构（Project Structure）

```
05_advanced_rag/
├── config.py                       # 全局配置（Global Config）
├── ingest.py                       # 文档入库（Document Indexing）
│
├── multi_query/
│   ├── multi_query_retriever.py    # Multi-Query 实现
│   ├── compare_single_vs_multi.py  # 单查询 vs 多查询对比
│   └── test_query_diversity.py     # Query Diversity 测试
│
├── reranking/
│   ├── reranker.py                 # Cross-Encoder Reranker
│   ├── two_stage_retrieval.py      # 两阶段检索（Two-Stage）
│   └── benchmark_reranking.py      # RAGAS 效果评估
│
├── hyde/
│   ├── hyde_retriever.py           # HyDE 实现
│   ├── compare_standard_vs_hyde.py # 效果对比
│   └── hyde_debug.py               # UMAP 向量空间可视化
│
├── advanced_pipeline.py            # 组合管道（Combined Pipeline）
├── compare_configs.py              # 5种配置对比
├── latency_breakdown.py            # 延迟瀑布图（Latency Waterfall）
├── app.py                          # FastAPI 接口
│
├── evaluation/
│   ├── ragas_evaluator.py
│   ├── create_test_dataset.py
│   ├── benchmark_all_configs.py
│   └── debug_retrieval.py
│
└── requirements.txt
```

### 7.2 三项技术总结速查

```
Multi-Query Retrieval（多查询检索）
  解决问题：单一查询的 Recall（召回率）不足，Semantic Gap
  核心机制：Query Rewriting → Multiple Query Variants → Union + Dedup
  额外代价：+1 LLM Call，+0.5~1s Latency
  效果：Context Recall +10~15%
  推荐：默认开启（Default ON）

Reranking（重排序）—— Cross-Encoder
  解决问题：Bi-Encoder 向量检索的 Precision（精确率）不足
  核心机制：Two-Stage：Broad Retrieval（k=20）→ Fine-grained Scoring → Top-5
  额外代价：本地 Cross-Encoder 推理，+0.5~2s Latency（CPU）
  效果：Context Precision +15~20%，单一优化效果最显著
  推荐：默认开启（Default ON）

HyDE（Hypothetical Document Embeddings）
  解决问题：Query-Document Asymmetry（问题 vs 答案形式不对称）
  核心机制：LLM 生成假设文档 → Embed 假设文档 → 检索真实文档
  额外代价：+1 LLM Call，+0.5~1s Latency
  效果：口语化/跨语言查询 +10%，精确查询无效甚至负效
  推荐：按需开启（Conditional ON）
```

**最优默认配置（Recommended Default）：**

```python
pipeline = AdvancedRAGPipeline(
    use_multi_query=True,   # ✅ 默认开
    use_reranking=True,     # ✅ 默认开
    use_hyde=False,         # ❌ 按需开
)
```

---

*文档版本：v2.0（英文关键词版）| 2026年3月 | 对应学习计划 Week 3-4*
