"""
文档加载与向量化
================
功能：
1. 加载 PDF / TXT / Markdown 文档
2. 文本分块（RecursiveCharacterTextSplitter）
3. 使用 nomic-embed-text 生成 Embedding
4. 存入 ChromaDB 向量数据库

使用方式：
    conda activate llm-learn
    python 02_rag/document_loader.py --file data/sample.pdf
"""

import os
import argparse
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings


# ============================================================
# 1. 文档加载 - 支持 PDF / TXT / Markdown
# ============================================================
def load_document(file_path: str):
    """
    根据文件扩展名自动选择合适的 Loader。

    核心概念：
    - LangChain 提供多种 DocumentLoader，每种处理一种文件格式
    - 返回的是 Document 对象列表，每个对象有 page_content 和 metadata
    - metadata 通常包含来源文件路径、页码等信息
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in (".md", ".markdown"):
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}（支持 .pdf / .txt / .md）")

    documents = loader.load()
    print(f"[加载完成] {file_path} -> {len(documents)} 个文档片段")
    return documents


# ============================================================
# 2. 文本分块 - 将长文档切分为适合检索的小块
# ============================================================
def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    将文档分割成更小的块，便于 Embedding 和检索。

    核心概念：
    - chunk_size: 每个块的最大字符数（不是 token 数），建议 500-1000
    - chunk_overlap: 相邻块的重叠字符数，防止关键信息被截断
    - RecursiveCharacterTextSplitter 会按优先级分割
      优先在段落边界切分，保持语义完整性
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],  # 中文友好的分隔符
    )

    chunks = text_splitter.split_documents(documents)
    print(f"[分块完成] {len(documents)} 个文档 -> {len(chunks)} 个文本块")
    print(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # 展示前3个块的预览
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"  块 {i}: {preview}...")

    return chunks


# ============================================================
# 3. Embedding + 存入 ChromaDB
# ============================================================
def create_vector_store(
    chunks,
    collection_name: str = "rag_docs",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "nomic-embed-text",
):
    """
    将文本块向量化并存入 ChromaDB。

    核心概念：
    - Embedding: 将文本转换为高维向量（例如 768 维），语义相近的文本向量距离更近
    - nomic-embed-text: Ollama 提供的 Embedding 模型，本地运行，无需 API Key
    - ChromaDB: 轻量级向量数据库，支持本地持久化存储
    - collection: ChromaDB 中的"表"，一个 collection 存一组相关文档
    """
    # 初始化 Embedding 模型（通过 Ollama 本地运行）
    embeddings = OllamaEmbeddings(model=embedding_model)

    # 初始化 ChromaDB 客户端（持久化到磁盘）
    client = chromadb.PersistentClient(path=persist_directory)

    # 如果 collection 已存在，先删除重建（方便重复实验）
    try:
        client.delete_collection(collection_name)
        print(f"[ChromaDB] 已删除旧 collection: {collection_name}")
    except ValueError:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )

    # 批量处理 Embedding（避免一次性处理太多导致 OOM）
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]
        ids = [f"doc_{i + j}" for j in range(len(batch))]

        # 生成 Embedding
        vectors = embeddings.embed_documents(texts)

        # 存入 ChromaDB
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  已入库: {i + len(batch)}/{len(chunks)} 个文本块")

    print(f"[向量化完成] {len(chunks)} 个块已存入 ChromaDB")
    print(f"  持久化路径: {persist_directory}")
    print(f"  Collection: {collection_name}")

    return collection


# ============================================================
# 4. 检索测试 - 验证向量库是否正常工作
# ============================================================
def test_retrieval(
    query: str,
    collection_name: str = "rag_docs",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "nomic-embed-text",
    top_k: int = 3,
):
    """
    用一个问题测试检索效果。

    核心概念：
    - 将查询文本也转换为向量
    - 在向量库中找到最相似的 top_k 个文本块
    - 返回的结果按相似度排序（距离越小越相似）
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)

    # 生成查询向量
    query_vector = embeddings.embed_query(query)

    # 检索
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    print(f"\n{'='*50}")
    print(f"[检索测试] 查询: {query}")
    print(f"{'='*50}")

    for i, (doc, distance) in enumerate(
        zip(results["documents"][0], results["distances"][0])
    ):
        print(f"\n--- 结果 {i+1} (距离: {distance:.4f}) ---")
        print(doc[:200])

    return results


# ============================================================
# 主入口 - 命令行工具
# ============================================================
def ingest_file(file_path: str, collection_name: str = "rag_docs",
                persist_directory: str = "./chroma_db",
                chunk_size: int = 500, chunk_overlap: int = 50):
    """完整的文档入库流程：加载 -> 分块 -> 向量化 -> 存储"""
    documents = load_document(file_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    collection = create_vector_store(chunks, collection_name, persist_directory)
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="文档加载与向量化工具")
    parser.add_argument("--file", type=str, required=True, help="文档路径（.pdf / .txt / .md）")
    parser.add_argument("--collection", type=str, default="rag_docs", help="ChromaDB collection 名称")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="ChromaDB 持久化路径")
    parser.add_argument("--chunk-size", type=int, default=500, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="分块重叠")
    parser.add_argument("--query", type=str, help="入库后进行检索测试的查询")

    args = parser.parse_args()

    # 入库
    ingest_file(args.file, args.collection, args.db_path, args.chunk_size, args.chunk_overlap)

    # 可选：检索测试
    if args.query:
        test_retrieval(args.query, args.collection, args.db_path)
