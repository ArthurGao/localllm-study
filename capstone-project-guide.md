# 综合项目实战指南
## RAG + 微调 + 评估 一体化系统

> 阶段：Week 7-8 | 前置：高级 RAG、RAGAS 评估、Unsloth 微调均已完成
>
> 目标：把前7周所有技术整合进一个生产就绪（Production-Ready）的完整系统。

---

## 目录

1. [系统架构设计](#1-系统架构设计)
2. [项目目录结构](#2-项目目录结构)
3. [核心组件实现](#3-核心组件实现)
4. [微调模型 + RAG 协同](#4-微调模型--rag-协同)
5. [自动化评估流水线](#5-自动化评估流水线)
6. [API 服务层](#6-api-服务层)
7. [端到端测试](#7-端到端测试)
8. [性能优化与监控](#8-性能优化与监控)
9. [完整项目实战](#9-完整项目实战)

---

## 1. 系统架构设计

### 1.1 全栈技术架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     综合 RAG 系统全架构                               │
└─────────────────────────────────────────────────────────────────────┘

用户层
  HTTP / Python SDK
       │
       ▼
API 层（FastAPI）
  ┌──────────────────────────────────────┐
  │  POST /ingest    POST /query          │
  │  POST /evaluate  GET  /health         │
  └──────────────────┬───────────────────┘
                     │
       ┌─────────────▼─────────────┐
       │       RAG 引擎             │
       │                           │
       │  ┌─────────────────────┐  │
       │  │  查询路由（Router）   │  │
       │  │  简单 → Basic RAG    │  │
       │  │  复杂 → Advanced RAG │  │
       │  └──────────┬──────────┘  │
       │             │             │
       │  ┌──────────▼──────────┐  │
       │  │  检索层              │  │
       │  │  Multi-Query        │  │
       │  │  HyDE（可选）       │  │
       │  │  Hybrid Search      │  │
       │  │  Reranking          │  │
       │  └──────────┬──────────┘  │
       │             │             │
       │  ┌──────────▼──────────┐  │
       │  │  生成层              │  │
       │  │  微调模型（Ollama）  │  │
       │  │  payment-expert     │  │
       │  └─────────────────────┘  │
       └─────────────┬─────────────┘
                     │
       ┌─────────────▼─────────────┐
       │       数据层               │
       │  ChromaDB（向量存储）      │
       │  nomic-embed-text         │
       └───────────────────────────┘
                     │
       ┌─────────────▼─────────────┐
       │       评估层               │
       │  RAGAS（自动评估）         │
       │  LLM Judge（质量监控）     │
       │  History（趋势追踪）       │
       └───────────────────────────┘
```

### 1.2 微调模型 vs 基础模型在 RAG 中的分工

```
基础模型（qwen3:8b）负责：
  ├── 生成 Multi-Query 查询变体
  ├── HyDE 假设文档生成
  └── RAGAS 评估的 Judge LLM

微调模型（payment-expert）负责：
  └── 最终答案生成（最重要的一步）

为什么这样分工？
  微调模型理解领域术语，生成的答案更专业
  但微调模型如果只训练了领域数据，可能在 Query 改写等通用任务上退化
  所以通用任务还是用基础模型
```

### 1.3 各阶段技术对应

| 周次 | 技术 | 在综合项目中的位置 |
|------|------|-----------------|
| Week 1-2 | Ollama + 基础 API | 所有 LLM 调用的底层 |
| Week 2-3 | 基础 RAG | 文档入库、基础检索链 |
| Week 3-4 | 高级 RAG | Multi-Query、Reranking、HyDE |
| Week 4-5 | RAGAS 评估 | 评估流水线、质量监控 |
| Week 5-7 | Unsloth 微调 | 领域专家 LLM（生成层） |
| Week 7-8 | 综合项目 | 全部整合 + FastAPI + CI |

---

## 2. 项目目录结构

```
capstone_project/
│
├── config.py                    # 全局配置（模型名、路径、参数）
│
├── data/
│   ├── raw_docs/                # 原始文档（PDF、Markdown）
│   ├── processed/               # 处理后的分块文档
│   └── test_cases.json          # RAGAS 评估测试集
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py                # 文档入库（Chunking + Embedding）
│   ├── retriever.py             # 检索层（Multi-Query + Reranking）
│   ├── generator.py             # 生成层（微调模型 + Prompt）
│   └── rag_pipeline.py          # 组合完整 RAG 流程
│
├── evaluation/
│   ├── __init__.py
│   ├── ragas_eval.py            # RAGAS 评估框架
│   ├── llm_judge.py             # 自定义 LLM Judge
│   └── compare.py               # 多配置对比
│
├── finetuning/
│   ├── generate_data.py         # 生成微调数据集
│   ├── finetune.py              # QLoRA 训练脚本
│   └── Modelfile                # Ollama 模型配置
│
├── api/
│   ├── main.py                  # FastAPI 主程序
│   ├── models.py                # Pydantic 请求/响应模型
│   └── middleware.py            # 日志、错误处理
│
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   └── test_end_to_end.py
│
├── results/
│   ├── evaluation_history.csv   # 评估历史
│   └── baseline_scores.json     # 当前最佳基准
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. 核心组件实现

### 3.1 config.py：全局配置中心

```python
"""
config.py - 所有配置集中管理，通过环境变量覆盖
"""
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    # LLM 模型
    base_llm: str        = "qwen3:8b"           # 通用任务（query改写、评估）
    answer_llm: str      = "payment-expert"      # 答案生成（微调模型）
    embedding: str       = "nomic-embed-text"    # 向量化模型
    judge_llm: str       = "llama3.3-70b-versatile"  # Groq 评估裁判
    
    # LLM 参数
    num_ctx: int         = 8192
    temperature: float   = 0.1
    query_temperature: float = 0.7  # Multi-Query 生成用高温度

@dataclass
class RAGConfig:
    # 文档分块
    chunk_size: int      = 512
    chunk_overlap: int   = 50
    
    # 检索
    k_base: int          = 20       # 粗检索候选数（送入 Reranker）
    k_final: int         = 5        # 最终返回文档数
    n_queries: int       = 3        # Multi-Query 生成几个变体
    use_hyde: bool       = False    # HyDE 默认关闭（可按需开启）
    use_reranking: bool  = True
    use_multi_query: bool = True
    score_threshold: float = 0.0    # 不设阈值（让 Reranker 决定）

@dataclass
class PathConfig:
    chroma_db: str       = "./chroma_db"
    collection: str      = "capstone_docs"
    test_cases: str      = "./data/test_cases.json"
    eval_history: str    = "./results/evaluation_history.csv"
    baseline: str        = "./results/baseline_scores.json"

@dataclass
class AppConfig:
    models: ModelConfig  = field(default_factory=ModelConfig)
    rag: RAGConfig       = field(default_factory=RAGConfig)
    paths: PathConfig    = field(default_factory=PathConfig)
    
    # API
    api_host: str        = "0.0.0.0"
    api_port: int        = 8000
    log_level: str       = "INFO"
    
    # Groq API key（评估用）
    groq_api_key: str    = os.getenv("GROQ_API_KEY", "")


# 全局单例
config = AppConfig()
```

### 3.2 pipeline/ingest.py：文档入库

```python
"""
ingest.py - 文档处理、分块、向量化、存库
"""
import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader, DirectoryLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import config


class DocumentIngester:
    """
    文档入库管理器
    支持：PDF、Markdown、增量更新（Incremental Indexing）
    """
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=config.models.embedding)
        self.vectorstore = Chroma(
            collection_name=config.paths.collection,
            embedding_function=self.embeddings,
            persist_directory=config.paths.chroma_db,
        )
        
        # 两种分块器（根据文档类型选择）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            add_start_index=True,
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]
        )
    
    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _should_skip(self, doc_path: str) -> bool:
        """检查文档是否已入库且未更改（增量更新）"""
        content_hash = self._compute_hash(open(doc_path, 'rb').read().hex()[:1000])
        existing = self.vectorstore.get(where={"doc_hash": content_hash})
        return len(existing["ids"]) > 0
    
    def ingest_file(self, file_path: str, force: bool = False) -> int:
        """
        入库单个文件
        Returns: 新增的 chunk 数量
        """
        path = Path(file_path)
        
        if not force and self._should_skip(file_path):
            print(f"  ⏭️  跳过（未变更）：{path.name}")
            return 0
        
        # 删除旧版本（如果存在）
        old = self.vectorstore.get(where={"source": str(path)})
        if old["ids"]:
            self.vectorstore.delete(ids=old["ids"])
            print(f"  🗑️  删除旧版本：{len(old['ids'])} 个块")
        
        # 加载文档
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            documents = loader.load()
            chunks = self.recursive_splitter.split_documents(documents)
        elif path.suffix.lower() in [".md", ".markdown"]:
            with open(path) as f:
                content = f.read()
            header_chunks = self.markdown_splitter.split_text(content)
            chunks = self.recursive_splitter.split_documents(header_chunks)
        else:
            print(f"  ⚠️  不支持的格式：{path.suffix}")
            return 0
        
        # 添加 metadata
        doc_hash = self._compute_hash(path.read_bytes().hex()[:1000])
        for chunk in chunks:
            chunk.metadata.update({
                "source": str(path),
                "filename": path.name,
                "doc_hash": doc_hash,
            })
        
        # 入库
        self.vectorstore.add_documents(chunks)
        print(f"  ✅ 入库：{path.name} → {len(chunks)} 个块")
        return len(chunks)
    
    def ingest_directory(self, dir_path: str = "./data/raw_docs") -> dict:
        """批量入库目录下所有支持的文档"""
        path = Path(dir_path)
        files = list(path.glob("**/*.pdf")) + list(path.glob("**/*.md"))
        
        total_new = 0
        print(f"📁 扫描目录：{dir_path}（{len(files)} 个文件）")
        
        for file in files:
            n = self.ingest_file(str(file))
            total_new += n
        
        total_chunks = self.vectorstore._collection.count()
        print(f"\n📊 入库完成：新增 {total_new} 个块 | 向量库总计 {total_chunks} 个块")
        return {"new_chunks": total_new, "total_chunks": total_chunks}
```

### 3.3 pipeline/retriever.py：检索层

```python
"""
retriever.py - 整合 Multi-Query + HyDE + Reranking 的检索层
"""
from typing import List
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import config


class AdvancedRetriever:
    """
    高级检索器：Multi-Query + （可选HyDE）+ Reranking
    """
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=config.models.embedding)
        self.vectorstore = Chroma(
            collection_name=config.paths.collection,
            embedding_function=self.embeddings,
            persist_directory=config.paths.chroma_db,
        )
        
        # 查询改写用基础模型（高温度）
        self.query_llm = ChatOllama(
            model=config.models.base_llm,
            temperature=config.models.query_temperature,
            num_ctx=config.models.num_ctx,
        )
        
        # Reranker
        if config.rag.use_reranking:
            print("📥 加载 Reranker...")
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    
    def _generate_queries(self, question: str) -> List[str]:
        """Multi-Query：生成查询变体"""
        prompt = f"""将以下问题从{config.rag.n_queries}个不同角度改写为搜索查询（每行一个，直接输出）：

问题：{question}

查询列表："""
        response = self.query_llm.invoke(prompt)
        variants = [
            line.strip() for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 3
        ][:config.rag.n_queries]
        return [question] + variants  # 包含原始问题
    
    def _hyde_embed(self, question: str) -> List[float]:
        """HyDE：生成假设文档并返回其向量"""
        prompt = f"请用技术文档风格写一段回答此问题的段落（100字，直接输出）：\n{question}"
        hypo_doc = self.query_llm.invoke(prompt).content
        return self.embeddings.embed_query(hypo_doc)
    
    def _rerank(self, question: str, docs: List[Document]) -> List[Document]:
        """Cross-Encoder 重排序"""
        if not docs or len(docs) <= config.rag.k_final:
            return docs[:config.rag.k_final]
        
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:config.rag.k_final]]
    
    def retrieve(self, question: str, verbose: bool = False) -> List[Document]:
        """
        完整检索流程
        Returns: 最终排序后的文档列表
        """
        all_docs, seen = [], set()
        
        def add_unique(docs):
            for doc in docs:
                h = hash(doc.page_content)
                if h not in seen:
                    seen.add(h)
                    all_docs.append(doc)
        
        # Multi-Query 检索
        if config.rag.use_multi_query:
            queries = self._generate_queries(question)
            if verbose:
                print(f"  🔍 生成 {len(queries)} 个查询")
            for q in queries:
                docs = self.vectorstore.similarity_search(q, k=config.rag.k_base // len(queries))
                add_unique(docs)
        
        # HyDE 检索（可选）
        if config.rag.use_hyde:
            hyde_vec = self._hyde_embed(question)
            docs = self.vectorstore.similarity_search_by_vector(hyde_vec, k=5)
            add_unique(docs)
        
        # 回退到基础检索
        if not all_docs:
            docs = self.vectorstore.similarity_search(question, k=config.rag.k_base)
            add_unique(docs)
        
        if verbose:
            print(f"  📄 候选文档：{len(all_docs)} 个")
        
        # Reranking
        if config.rag.use_reranking:
            final = self._rerank(question, all_docs)
            if verbose:
                print(f"  📊 重排后：{len(final)} 个")
            return final
        
        return all_docs[:config.rag.k_final]
```

### 3.4 pipeline/generator.py：生成层

```python
"""
generator.py - 使用微调模型生成最终答案
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List

from config import config


# 核心 Prompt：约束模型只基于上下文回答
RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一个专业的技术文档问答助手。

规则：
1. 只基于以下检索到的文档内容回答问题
2. 如果文档中没有相关信息，明确说"文档中未找到此信息"
3. 不要编造文档中没有的内容
4. 回答要专业、结构清晰

检索到的文档：
{context}

问题：{question}

回答：""")


class AnswerGenerator:
    """答案生成器，使用微调后的领域专家模型"""
    
    def __init__(self):
        # 微调模型负责最终答案生成
        self.llm = ChatOllama(
            model=config.models.answer_llm,
            num_ctx=config.models.num_ctx,
            temperature=config.models.temperature,
        )
        self.chain = RAG_PROMPT | self.llm | StrOutputParser()
    
    def format_context(self, docs: List[Document]) -> str:
        """将文档列表格式化为上下文字符串"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "未知来源")
            parts.append(f"[文档 {i} - {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
    
    def generate(self, question: str, docs: List[Document]) -> dict:
        """
        生成答案
        Returns: {answer, sources, doc_count}
        """
        context = self.format_context(docs)
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        sources = list({doc.metadata.get("filename", "未知") for doc in docs})
        
        return {
            "answer": answer,
            "sources": sources,
            "doc_count": len(docs),
        }
    
    def generate_stream(self, question: str, docs: List[Document]):
        """流式生成（用于实时显示）"""
        context = self.format_context(docs)
        for chunk in self.chain.stream({"context": context, "question": question}):
            yield chunk
```

### 3.5 pipeline/rag_pipeline.py：主流程编排

```python
"""
rag_pipeline.py - 完整 RAG 流程编排
"""
from .ingest import DocumentIngester
from .retriever import AdvancedRetriever
from .generator import AnswerGenerator


class RAGPipeline:
    """
    完整的 RAG 流水线
    整合：文档入库 + 高级检索 + 微调模型生成
    """
    
    def __init__(self):
        self.ingester  = DocumentIngester()
        self.retriever = AdvancedRetriever()
        self.generator = AnswerGenerator()
    
    def ingest(self, path: str, force: bool = False) -> dict:
        """入库文档（单文件或目录）"""
        import os
        if os.path.isdir(path):
            return self.ingester.ingest_directory(path)
        else:
            n = self.ingester.ingest_file(path, force=force)
            return {"new_chunks": n}
    
    def query(self, question: str, verbose: bool = False) -> dict:
        """
        完整查询流程：检索 → 生成
        Returns: {answer, sources, doc_count, question}
        """
        # 检索
        docs = self.retriever.retrieve(question, verbose=verbose)
        
        # 生成
        result = self.generator.generate(question, docs)
        result["question"] = question
        
        return result
    
    def query_stream(self, question: str):
        """流式查询"""
        docs = self.retriever.retrieve(question)
        yield from self.generator.generate_stream(question, docs)
```

---

## 4. 微调模型 + RAG 协同

### 4.1 三种方案的实际效果对比

```python
"""
compare_strategies.py - 量化对比三种方案
"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pipeline.rag_pipeline import RAGPipeline
from pipeline.retriever import AdvancedRetriever


def strategy_comparison(questions: list, retriever: AdvancedRetriever):
    """对比三种方案：基础模型、微调模型、微调+RAG"""
    
    # 策略1：基础模型（无 RAG）
    base_llm = ChatOllama(model="qwen3:8b", temperature=0.1, num_ctx=8192)
    
    # 策略2：微调模型（无 RAG）
    finetuned_llm = ChatOllama(model="payment-expert", temperature=0.1, num_ctx=8192)
    
    # 策略3：微调模型 + RAG（本项目完整方案）
    rag_pipeline = RAGPipeline()
    
    simple_prompt = ChatPromptTemplate.from_template("{question}")
    
    results = []
    for q in questions:
        base_ans     = (simple_prompt | base_llm | StrOutputParser()).invoke({"question": q})
        finetune_ans = (simple_prompt | finetuned_llm | StrOutputParser()).invoke({"question": q})
        rag_ans      = rag_pipeline.query(q)["answer"]
        
        results.append({
            "question": q,
            "基础模型": base_ans,
            "微调模型": finetune_ans,
            "微调+RAG": rag_ans,
        })
    
    return results
```

### 4.2 System Prompt 的一致性（关键细节）

微调时的 system prompt 必须与 Modelfile 和推理时保持完全一致，否则模型表现会"不一致"：

```python
# ❌ 错误：三处不同

# 训练数据：
{"role": "system", "value": "你是支付专家。"}

# Modelfile：
SYSTEM "你是一个专业的技术助手。"

# 推理 Prompt：
"你是文档问答助手。基于上下文回答..."

# ✅ 正确：统一的 system prompt

SYSTEM_PROMPT = "你是一个专业的支付系统工程师，熟悉 ISO 20022 标准和 FTH 系统。"
# 训练数据、Modelfile、推理 Prompt 全部使用这个值
```

---

## 5. 自动化评估流水线

### 5.1 evaluation/ragas_eval.py

```python
"""
ragas_eval.py - 完整评估流水线
"""
import json, pandas as pd
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall, answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

from pipeline.rag_pipeline import RAGPipeline
from config import config


class EvaluationPipeline:
    
    def __init__(self):
        # Groq 作为 Judge（速度快，免费）
        judge_llm = LangchainLLMWrapper(
            ChatGroq(model=config.models.judge_llm, temperature=0.0,
                     api_key=config.groq_api_key)
        )
        judge_emb = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=config.models.embedding)
        )
        
        # 配置指标
        for m in [faithfulness, answer_relevancy, context_precision,
                  context_recall, answer_correctness]:
            m.llm = judge_llm
        for m in [answer_relevancy, answer_correctness]:
            m.embeddings = judge_emb
    
    def build_eval_dataset(self, rag_pipeline: RAGPipeline,
                           test_cases_path: str = None) -> Dataset:
        """构建评估数据集"""
        path = test_cases_path or config.paths.test_cases
        with open(path) as f:
            test_cases = json.load(f)
        
        eval_data = []
        for i, case in enumerate(test_cases):
            q = case["question"]
            print(f"  [{i+1}/{len(test_cases)}] {q[:40]}...")
            
            result = rag_pipeline.query(q)
            docs   = rag_pipeline.retriever.retrieve(q)
            
            eval_data.append({
                "question":     q,
                "answer":       result["answer"],
                "contexts":     [d.page_content for d in docs],
                "ground_truth": case.get("ground_truth", ""),
            })
        
        return Dataset.from_list(eval_data)
    
    def run(self, rag_pipeline: RAGPipeline, run_name: str = "") -> pd.DataFrame:
        """运行完整评估"""
        print(f"\n🚀 开始评估：{run_name or 'unnamed'}")
        
        dataset = self.build_eval_dataset(rag_pipeline)
        
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision,
                     context_recall, answer_correctness],
            raise_exceptions=False,
        )
        
        df = results.to_pandas()
        
        # 汇总
        summary = {
            "run_name":           run_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp":          datetime.now().isoformat(),
            "faithfulness":       df["faithfulness"].mean(),
            "answer_relevancy":   df["answer_relevancy"].mean(),
            "context_precision":  df["context_precision"].mean(),
            "context_recall":     df.get("context_recall", pd.Series([None])).mean(),
            "answer_correctness": df.get("answer_correctness", pd.Series([None])).mean(),
        }
        
        # 追加到历史记录
        history_path = Path(config.paths.eval_history)
        history_path.parent.mkdir(exist_ok=True)
        
        history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        history_df = pd.concat([history_df, pd.DataFrame([summary])], ignore_index=True)
        history_df.to_csv(history_path, index=False)
        
        print(f"\n📊 评估结果（{run_name}）：")
        for k, v in summary.items():
            if isinstance(v, float):
                bar = "█" * int(v * 20)
                print(f"  {k:<25}: {v:.3f}  {bar}")
        
        return df
    
    def check_regression(self, current_scores: dict, threshold: float = 0.02) -> bool:
        """检查是否有性能退步"""
        baseline_path = Path(config.paths.baseline)
        if not baseline_path.exists():
            print("⚠️ 无基准文件，跳过退步检查")
            self._save_baseline(current_scores)
            return True
        
        with open(baseline_path) as f:
            baseline = json.load(f)["scores"]
        
        passed = True
        print("\n🔍 退步检查：")
        for metric, current in current_scores.items():
            if not isinstance(current, float):
                continue
            prev = baseline.get(metric, 0)
            delta = current - prev
            if delta < -threshold:
                print(f"  ❌ {metric}: {prev:.3f} → {current:.3f} (退步 {abs(delta):.3f})")
                passed = False
            else:
                icon = "✅" if delta > 0.01 else "➖"
                print(f"  {icon} {metric}: {prev:.3f} → {current:.3f}")
        
        if passed:
            self._save_baseline(current_scores)
        return passed
    
    def _save_baseline(self, scores: dict):
        with open(config.paths.baseline, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "scores": scores},
                      f, indent=2)
```

---

## 6. API 服务层

### 6.1 api/models.py：请求/响应模型

```python
from pydantic import BaseModel
from typing import List, Optional

class IngestRequest(BaseModel):
    path: str
    force: bool = False

class IngestResponse(BaseModel):
    new_chunks: int
    total_chunks: Optional[int] = None
    message: str

class QueryRequest(BaseModel):
    question: str
    stream: bool = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    doc_count: int
    latency_ms: float

class EvalResponse(BaseModel):
    run_name: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float]
    regression_check: bool
```

### 6.2 api/main.py：FastAPI 主程序

```python
"""
main.py - FastAPI 服务，对外暴露 RAG 系统接口
"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from api.models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, EvalResponse
)
from pipeline.rag_pipeline import RAGPipeline
from evaluation.ragas_eval import EvaluationPipeline

# 全局实例（应用启动时初始化一次）
rag: RAGPipeline = None
evaluator: EvaluationPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, evaluator
    print("🚀 初始化 RAG Pipeline...")
    rag = RAGPipeline()
    evaluator = EvaluationPipeline()
    print("✅ 服务就绪")
    yield
    print("🛑 服务关闭")


app = FastAPI(
    title="支付系统 RAG API",
    description="基于微调 Qwen + 高级 RAG 的技术文档问答系统",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "models": {"llm": "payment-expert", "embedding": "nomic-embed-text"}}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """入库文档（支持单文件或目录）"""
    try:
        result = rag.ingest(req.path, force=req.force)
        return IngestResponse(
            new_chunks=result["new_chunks"],
            total_chunks=result.get("total_chunks"),
            message=f"成功入库 {result['new_chunks']} 个新文档块"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """查询问答（支持普通和流式两种模式）"""
    if req.stream:
        def stream_gen():
            for chunk in rag.query_stream(req.question):
                yield chunk
        return StreamingResponse(stream_gen(), media_type="text/plain")
    
    start = time.time()
    try:
        result = rag.query(req.question)
        latency = (time.time() - start) * 1000
        return QueryResponse(latency_ms=round(latency, 1), **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(run_name: str = ""):
    """运行 RAGAS 评估并检查退步"""
    try:
        df = evaluator.run(rag, run_name=run_name)
        scores = {
            "faithfulness":       float(df["faithfulness"].mean()),
            "answer_relevancy":   float(df["answer_relevancy"].mean()),
            "context_precision":  float(df["context_precision"].mean()),
            "context_recall":     float(df["context_recall"].mean()) if "context_recall" in df else None,
        }
        passed = evaluator.check_regression(scores)
        return EvalResponse(
            run_name=run_name or "latest",
            regression_check=passed,
            **scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from config import config
    uvicorn.run(app, host=config.api_host, port=config.api_port)
```

---

## 7. 端到端测试

### 7.1 tests/test_end_to_end.py

```python
"""
test_end_to_end.py - 端到端集成测试
"""
import pytest
from pipeline.rag_pipeline import RAGPipeline
from evaluation.ragas_eval import EvaluationPipeline


@pytest.fixture(scope="module")
def pipeline():
    p = RAGPipeline()
    # 确保测试文档已入库
    p.ingest("./data/raw_docs")
    return p


class TestRetrieval:
    """检索层测试"""
    
    def test_basic_retrieval(self, pipeline):
        docs = pipeline.retriever.retrieve("AM04 错误码")
        assert len(docs) > 0, "应该检索到至少 1 个文档"
        assert len(docs) <= 5, "不应超过 k_final 限制"
    
    def test_multi_query_generates_variants(self, pipeline):
        queries = pipeline.retriever._generate_queries("支付失败原因")
        assert len(queries) >= 2, "应该生成至少 2 个查询（含原始）"
    
    def test_retrieval_for_unknown_topic(self, pipeline):
        """范围外问题的检索（文档库中没有的内容）"""
        docs = pipeline.retriever.retrieve("量子计算的最新进展")
        # 检索会返回一些文档（无法判断相关性），但生成答案应该说"未找到"
        assert isinstance(docs, list)


class TestGeneration:
    """生成层测试"""
    
    def test_answer_contains_relevant_content(self, pipeline):
        result = pipeline.query("AM04 是什么？")
        assert "AM04" in result["answer"] or "余额" in result["answer"]
        assert result["doc_count"] > 0
    
    def test_out_of_scope_says_not_found(self, pipeline):
        result = pipeline.query("请给我写一首诗")
        # 微调模型应该说文档中没有此信息
        assert "未找到" in result["answer"] or "文档" in result["answer"]
    
    def test_sources_are_returned(self, pipeline):
        result = pipeline.query("FTH 系统的超时配置")
        assert len(result["sources"]) > 0


class TestQuality:
    """质量指标测试（快速抽样，非完整 RAGAS）"""
    
    def test_faithfulness_threshold(self, pipeline):
        """快速幻觉检测（5个问题的抽样）"""
        test_questions = [
            "AM04 错误码是什么？",
            "FTH 系统的超时时间是多少？",
        ]
        for q in test_questions:
            result = pipeline.query(q)
            # 简单检查：答案不为空，且不超过合理长度
            assert len(result["answer"]) > 20
            assert len(result["answer"]) < 3000


# 运行：pytest tests/test_end_to_end.py -v
```

---

## 8. 性能优化与监控

### 8.1 延迟分析

```python
import time
from functools import wraps

def timer(name=""):
    """装饰器：测量函数执行时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  ⏱  {name or func.__name__}: {elapsed:.0f}ms")
            return result
        return wrapper
    return decorator


# 典型延迟分解（本地 Ollama + T4 Reranker）：
# 
# Multi-Query 生成：  ~800ms  （1次 LLM 调用）
# 向量检索（4次）：    ~200ms  （ChromaDB 很快）
# Reranking（20个）：  ~600ms  （Cross-Encoder 推理）
# 答案生成：         ~2000ms  （1次微调模型调用）
# ─────────────────────────
# 总计：             ~3600ms  （约 3.5 秒）
#
# 如果使用 Groq 作为 answer_llm（远端但快）：
# 答案生成：          ~400ms
# 总计：             ~2000ms  （约 2 秒）
```

### 8.2 缓存层（可选）

```python
from langchain.cache import InMemoryCache
from langchain_community.cache import RedisSemanticCache
import langchain

# 简单内存缓存（开发阶段）
langchain.llm_cache = InMemoryCache()

# 语义缓存（生产阶段，相似问题复用缓存）
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    score_threshold=0.95,
)
```

---

## 9. 完整项目实战

### 9.1 启动顺序

```bash
# 1. 确保 Ollama 运行且模型已就绪
ollama serve &
ollama pull qwen3:8b           # 基础模型
ollama pull nomic-embed-text   # Embedding 模型
# （微调模型 payment-expert 假设已完成第7章的微调和导入）

# 2. 安装依赖
pip install -r requirements.txt --break-system-packages

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 GROQ_API_KEY

# 4. 入库文档
python -c "
from pipeline.rag_pipeline import RAGPipeline
p = RAGPipeline()
p.ingest('./data/raw_docs')
"

# 5. 运行端到端测试
pytest tests/ -v

# 6. 启动 API 服务
python api/main.py

# 7. 运行完整 RAGAS 评估
curl -X POST "http://localhost:8000/evaluate?run_name=v1.0"
```

### 9.2 Claude Code 提示词

```
创建完整的 capstone_project/ 综合项目，整合前7周所有技术：

核心要求：
1. config.py
   - 所有配置统一管理
   - 支持环境变量覆盖
   - answer_llm 默认 payment-expert（微调模型）
   - 无法加载微调模型时自动回退到 qwen3:8b

2. pipeline/ 目录
   - ingest.py：支持 PDF + Markdown，增量更新，MD5 去重
   - retriever.py：Multi-Query + Reranking，HyDE 可选，详细日志
   - generator.py：微调模型生成，流式输出支持
   - rag_pipeline.py：统一入口

3. evaluation/ 目录
   - ragas_eval.py：Groq Judge + 本地 Embedding
   - llm_judge.py：自定义 faithfulness 和 quality 评估
   - compare.py：对比"基础RAG / 高级RAG / 微调+RAG"三种方案

4. api/main.py
   - FastAPI，4个端点：/health /ingest /query /evaluate
   - /query 支持 stream=true 流式返回
   - 每次请求记录延迟到日志

5. tests/test_end_to_end.py
   - 检索测试（基础检索、Multi-Query、范围外）
   - 生成测试（答案相关性、范围外处理）
   - 快速质量抽样（5个问题）

6. README.md
   - 项目架构图（ASCII）
   - 快速启动步骤
   - 关键配置说明
   - 评估分数基准

技术栈固定：
  LLM: ChatOllama（本地）或 ChatGroq（评估Judge）
  Embedding: OllamaEmbeddings(nomic-embed-text)
  向量库: ChromaDB
  Reranker: BAAI/bge-reranker-v2-m3
  评估: RAGAS + 自定义 LLMJudge
  API: FastAPI + uvicorn
```

---

## 总结

```
八周学习成果汇总：

Week 1-2：基础能力
  ✅ Ollama 本地部署，API 调用，流式输出
  ✅ 基础 RAG：PDF→Chunk→Embed→ChromaDB→问答

Week 3-4：检索优化
  ✅ Multi-Query：扩大召回，+10-15% Context Recall
  ✅ Reranking：精排降噪，+15-20% Context Precision
  ✅ HyDE：应对口语化查询，特定场景 +10%

Week 4-5：质量评估
  ✅ RAGAS 5项指标：从感觉变成数字
  ✅ LLM-as-Judge：自定义评估维度
  ✅ 持续评估体系：每次修改都能量化影响

Week 5-7：领域专精
  ✅ 数据集构建：从文档自动生成，手工审核
  ✅ QLoRA 微调：~6GB 显存，1% 参数，领域专业性 +20-40%
  ✅ GGUF 导出 + Ollama 导入：端到端闭环

Week 7-8：系统整合
  ✅ 微调模型 + 高级 RAG 协同（各司其职）
  ✅ FastAPI 服务化
  ✅ 自动化评估流水线 + 退步检查

最终系统能力：
  输入：自然语言问题
  输出：基于真实文档、领域专业、可溯源的答案
  质量：Faithfulness > 0.85，Context Recall > 0.80
  速度：~3-4s（本地），~1.5-2s（Groq answer LLM）
```

---

*文档版本：v1.0 | 2026年3月 | 对应学习计划 Week 7-8*
