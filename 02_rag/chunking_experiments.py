"""
分块策略实验对比
================
功能：
1. 对同一文档测试 4 种不同的分块策略
2. 对比分块数量、平均长度、最大/最小长度
3. 用测试问题评估每种策略的检索召回率
4. 输出对比表格

学习重点：
- 理解不同分块策略的差异和适用场景
- chunk_size 和 chunk_overlap 对检索质量的影响
- 如何选择合适的分块策略

使用方式：
    conda activate llm-learn
    python 02_rag/chunking_experiments.py
"""

import sys
import os
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_embeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
import chromadb


# ============================================================
# 测试问题和期望答案关键词（用于评估召回率）
# ============================================================
TEST_CASES = [
    {
        "question": "RAG 解决了什么问题？",
        "expected_keywords": ["知识时效", "幻觉", "截止日期"],
    },
    {
        "question": "常见的向量数据库有哪些？",
        "expected_keywords": ["ChromaDB", "Pinecone", "Weaviate", "Milvus"],
    },
    {
        "question": "chunk_size 应该设置多大？",
        "expected_keywords": ["500", "1000", "分块大小"],
    },
    {
        "question": "什么是 Embedding？",
        "expected_keywords": ["向量", "语义", "维"],
    },
    {
        "question": "Naive RAG 有什么局限？",
        "expected_keywords": ["查询优化", "单次检索", "不相关"],
    },
    {
        "question": "余弦相似度是什么？",
        "expected_keywords": ["余弦", "夹角", "cosine"],
    },
    {
        "question": "Overlap 重叠的作用是什么？",
        "expected_keywords": ["截断", "重叠", "边界"],
    },
    {
        "question": "temperature 对 RAG 有什么影响？",
        "expected_keywords": ["temperature", "幻觉", "0.1"],
    },
]


# ============================================================
# 分块策略定义
# ============================================================
def strategy_fixed_small(documents):
    """策略1：固定大小 - 小块 (256字符)"""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=256,
        chunk_overlap=30,
        length_function=len,
    )
    return splitter.split_documents(documents)


def strategy_recursive_medium(documents):
    """策略2：递归字符分割 - 中块 (512字符) ⭐ 推荐默认"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def strategy_recursive_large(documents):
    """策略3：递归字符分割 - 大块 (1024字符)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def strategy_recursive_overlap_heavy(documents):
    """策略4：递归分割 - 高重叠 (512字符, 重叠200)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(documents)


STRATEGIES = [
    ("固定小块(256)", strategy_fixed_small),
    ("递归中块(512) ⭐", strategy_recursive_medium),
    ("递归大块(1024)", strategy_recursive_large),
    ("递归高重叠(512/200)", strategy_recursive_overlap_heavy),
]


# ============================================================
# 检索测试函数
# ============================================================
def test_retrieval_quality(collection, embeddings, test_cases, top_k=3):
    """
    用测试问题评估检索质量。

    对每个测试问题：
    1. 将问题转为向量
    2. 检索 top_k 个结果
    3. 检查结果中是否包含期望关键词
    4. 计算召回率
    """
    hits = 0

    for case in test_cases:
        query_vec = embeddings.embed_query(case["question"])
        results = collection.query(query_embeddings=[query_vec], n_results=top_k)
        retrieved_text = " ".join(results["documents"][0])

        # 检查是否命中至少一个关键词
        found = any(kw.lower() in retrieved_text.lower() for kw in case["expected_keywords"])
        if found:
            hits += 1

    recall = hits / len(test_cases)
    return recall


# ============================================================
# 主实验流程
# ============================================================
def run_experiments():
    """运行所有分块策略实验并输出对比结果。"""

    # 加载文档
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    doc_files = [
        os.path.join(data_dir, "rag_knowledge_base.txt"),
        os.path.join(data_dir, "sample_rag_test.txt"),
    ]

    documents = []
    for f in doc_files:
        if os.path.exists(f):
            loader = TextLoader(f, encoding="utf-8")
            documents.extend(loader.load())
            print(f"  已加载: {f}")

    if not documents:
        print("错误：没有找到测试文档。请确保 data/ 目录下有 .txt 文件。")
        return

    total_chars = sum(len(d.page_content) for d in documents)
    print(f"\n文档总量: {len(documents)} 个文件, {total_chars} 字符\n")

    # 初始化 Embedding
    embeddings = get_embeddings()

    # 临时 ChromaDB 路径
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chunking_exp_db")

    # 存储结果
    results = []

    for name, strategy_fn in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"测试策略: {name}")
        print(f"{'='*60}")

        # 1. 分块
        t0 = time.time()
        chunks = strategy_fn(documents)
        chunk_time = time.time() - t0

        # 统计
        lengths = [len(c.page_content) for c in chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0

        print(f"  块数量: {len(chunks)}")
        print(f"  平均长度: {avg_len:.0f} 字符")
        print(f"  最小/最大: {min_len}/{max_len} 字符")
        print(f"  分块耗时: {chunk_time:.2f}s")

        # 2. 存入 ChromaDB
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        client = chromadb.PersistentClient(path=db_path)
        collection = client.create_collection(
            name="exp", metadata={"hnsw:space": "cosine"}
        )

        t0 = time.time()
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.page_content for c in batch]
            vecs = embeddings.embed_documents(texts)
            collection.add(
                ids=[f"chunk_{i+j}" for j in range(len(batch))],
                embeddings=vecs,
                documents=texts,
            )
        embed_time = time.time() - t0
        print(f"  Embedding 耗时: {embed_time:.2f}s")

        # 3. 评估检索质量
        recall_3 = test_retrieval_quality(collection, embeddings, TEST_CASES, top_k=3)
        recall_5 = test_retrieval_quality(collection, embeddings, TEST_CASES, top_k=5)
        print(f"  召回率@3: {recall_3:.1%}")
        print(f"  召回率@5: {recall_5:.1%}")

        results.append({
            "name": name,
            "chunks": len(chunks),
            "avg_len": avg_len,
            "min_len": min_len,
            "max_len": max_len,
            "recall_3": recall_3,
            "recall_5": recall_5,
            "embed_time": embed_time,
        })

    # 清理临时数据库
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # 4. 输出对比表格
    print(f"\n\n{'='*80}")
    print("分块策略对比总结")
    print(f"{'='*80}")
    print(f"{'策略':<25} {'块数':>5} {'均长':>6} {'最小':>5} {'最大':>5} {'R@3':>6} {'R@5':>6} {'耗时':>6}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<25} {r['chunks']:>5} {r['avg_len']:>6.0f} "
            f"{r['min_len']:>5} {r['max_len']:>5} {r['recall_3']:>5.0%} "
            f"{r['recall_5']:>5.0%} {r['embed_time']:>5.1f}s"
        )

    print(f"\n{'='*80}")
    print("结论指南：")
    print("- 块太小 → 缺乏上下文，语义不完整")
    print("- 块太大 → 包含无关信息，检索精度下降")
    print("- 推荐起点：递归分割，chunk_size=512，overlap=50")
    print("- 如果召回率不够 → 增加 overlap 或尝试更小的块")
    print("- 如果答案不完整 → 尝试更大的块或增加 top_k")
    print(f"{'='*80}")


# ============================================================
# 额外实验：chunk_size 敏感度分析
# ============================================================
def chunk_size_sensitivity():
    """
    实验：固定策略（递归分割），只改变 chunk_size，观察对检索质量的影响。
    帮助理解 chunk_size 这个参数的敏感度。
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    loader = TextLoader(os.path.join(data_dir, "rag_knowledge_base.txt"), encoding="utf-8")
    documents = loader.load()

    embeddings = get_embeddings()
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chunk_size_db")

    print(f"\n{'='*60}")
    print("chunk_size 敏感度分析")
    print(f"{'='*60}")
    print(f"{'chunk_size':>12} {'块数':>6} {'均长':>6} {'R@3':>6} {'R@5':>6}")
    print("-" * 45)

    for size in [128, 256, 384, 512, 768, 1024, 1536, 2048]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=int(size * 0.1),
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.create_collection("exp", metadata={"hnsw:space": "cosine"})

        texts = [c.page_content for c in chunks]
        vecs = embeddings.embed_documents(texts)
        collection.add(
            ids=[f"c_{i}" for i in range(len(chunks))],
            embeddings=vecs,
            documents=texts,
        )

        avg_len = sum(len(t) for t in texts) / len(texts)
        r3 = test_retrieval_quality(collection, embeddings, TEST_CASES, top_k=3)
        r5 = test_retrieval_quality(collection, embeddings, TEST_CASES, top_k=5)

        print(f"{size:>12} {len(chunks):>6} {avg_len:>6.0f} {r3:>5.0%} {r5:>5.0%}")

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    print("\n提示：通常 384-768 范围是最佳区间")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分块策略实验对比")
    parser.add_argument(
        "--mode",
        choices=["compare", "sensitivity", "all"],
        default="all",
        help="实验模式: compare(策略对比), sensitivity(chunk_size敏感度), all(全部)",
    )
    args = parser.parse_args()

    if args.mode in ("compare", "all"):
        run_experiments()
    if args.mode in ("sensitivity", "all"):
        chunk_size_sensitivity()
