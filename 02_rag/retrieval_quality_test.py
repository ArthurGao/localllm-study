"""
RAG 检索质量系统测试
===================
功能：
1. 定义标准测试集（问题 + 标准答案 + 期望检索到的关键信息）
2. 自动测试检索质量：召回率、精确率、MRR
3. 诊断检索问题并给出优化建议
4. 对比不同 top_k 值的效果

学习重点：
- 如何系统地评估 RAG 检索质量
- 理解 Recall@K、Precision@K、MRR 等指标
- 掌握检索质量问题的诊断方法

使用方式：
    # 先入库文档
    python 02_rag/document_loader.py --file data/rag_knowledge_base.txt
    # 再运行测试
    python 02_rag/retrieval_quality_test.py

    # 也可以跳过入库，直接运行（自动入库）
    python 02_rag/retrieval_quality_test.py --auto-ingest
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_embeddings
from document_loader import ingest_file

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb


# ============================================================
# 标准测试集：问题 + 期望的关键信息
# ============================================================
TEST_SUITE = [
    {
        "id": "Q01",
        "question": "RAG 解决了大语言模型的什么问题？",
        "expected_in_context": ["知识时效性", "幻觉"],
        "ground_truth": "RAG 解决了知识时效性（训练数据有截止日期）和幻觉问题（模型编造内容）。",
        "difficulty": "easy",
    },
    {
        "id": "Q02",
        "question": "RAG 的三个主要阶段是什么？",
        "expected_in_context": ["索引", "检索", "生成"],
        "ground_truth": "索引阶段（Indexing）、检索阶段（Retrieval）、生成阶段（Generation）。",
        "difficulty": "easy",
    },
    {
        "id": "Q03",
        "question": "nomic-embed-text 的向量维度是多少？",
        "expected_in_context": ["768"],
        "ground_truth": "nomic-embed-text 的向量维度是 768。",
        "difficulty": "easy",
    },
    {
        "id": "Q04",
        "question": "ChromaDB 有什么特点和限制？",
        "expected_in_context": ["本地运行", "持久化", "不支持分布式"],
        "ground_truth": "ChromaDB 支持本地运行、持久化存储，但不支持分布式部署。",
        "difficulty": "medium",
    },
    {
        "id": "Q05",
        "question": "什么是 HyDE？",
        "expected_in_context": ["假设", "HyDE", "hypothetical"],
        "ground_truth": "HyDE 是假设文档嵌入技术，先让 LLM 生成假设答案，再用假设答案去检索。",
        "difficulty": "medium",
    },
    {
        "id": "Q06",
        "question": "Reranking 重排序的原理是什么？",
        "expected_in_context": ["Cross-Encoder", "重排序", "精排"],
        "ground_truth": "Reranking 使用 Cross-Encoder 对检索结果做精确打分重排，提高最终质量。",
        "difficulty": "medium",
    },
    {
        "id": "Q07",
        "question": "chunk_overlap 重叠的作用是什么？",
        "expected_in_context": ["截断", "重叠", "边界"],
        "ground_truth": "Overlap 确保关键信息不会在块边界被截断，相邻块共享部分内容。",
        "difficulty": "easy",
    },
    {
        "id": "Q08",
        "question": "什么是 Modular RAG？",
        "expected_in_context": ["模块化", "路由", "可插拔"],
        "ground_truth": "Modular RAG 将系统拆分为可插拔模块，包括路由、反馈循环、Agentic 编排。",
        "difficulty": "hard",
    },
    {
        "id": "Q09",
        "question": "RAG 中 temperature 应该设置多少？",
        "expected_in_context": ["0.1", "temperature", "幻觉"],
        "ground_truth": "RAG 场景建议 temperature=0.1，低温度减少幻觉。",
        "difficulty": "easy",
    },
    {
        "id": "Q10",
        "question": "Faithfulness 忠实度指标衡量什么？",
        "expected_in_context": ["忠实", "编造", "上下文"],
        "ground_truth": "Faithfulness 衡量答案中的陈述是否都有检索到的上下文支撑，低忠实度意味着模型在编造。",
        "difficulty": "medium",
    },
    {
        "id": "Q11",
        "question": "FAISS 是什么？有什么特点？",
        "expected_in_context": ["FAISS", "Meta", "GPU"],
        "ground_truth": "FAISS 是 Meta 开源的向量搜索库，支持 GPU 加速，适合超大规模向量搜索。",
        "difficulty": "medium",
    },
    {
        "id": "Q12",
        "question": "BGE-M3 模型有什么独特能力？",
        "expected_in_context": ["dense", "sparse", "multi-vector"],
        "ground_truth": "BGE-M3 支持 dense、sparse、multi-vector 三种检索方式，适合混合检索。",
        "difficulty": "hard",
    },
]


# ============================================================
# 检索质量评估
# ============================================================
class RetrievalEvaluator:
    """RAG 检索质量评估器。"""

    def __init__(self, collection, embeddings):
        self.collection = collection
        self.embeddings = embeddings

    def retrieve(self, query, top_k=5):
        """执行一次检索，返回结果。"""
        query_vec = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
        )
        return results

    def check_recall(self, retrieved_texts, expected_keywords):
        """
        检查检索结果是否包含期望的关键信息。
        返回命中的关键词列表和召回率。
        """
        combined = " ".join(retrieved_texts).lower()
        hits = [kw for kw in expected_keywords if kw.lower() in combined]
        recall = len(hits) / len(expected_keywords) if expected_keywords else 0
        return hits, recall

    def evaluate_test_suite(self, test_suite, top_k=5):
        """
        评估完整测试集。

        返回每个问题的详细结果和整体指标。
        """
        results = []

        for case in test_suite:
            retrieved = self.retrieve(case["question"], top_k=top_k)
            texts = retrieved["documents"][0]
            distances = retrieved["distances"][0]

            hits, recall = self.check_recall(texts, case["expected_in_context"])

            results.append({
                "id": case["id"],
                "question": case["question"],
                "difficulty": case["difficulty"],
                "expected_keywords": case["expected_in_context"],
                "found_keywords": hits,
                "recall": recall,
                "top_distance": distances[0] if distances else None,
                "avg_distance": sum(distances) / len(distances) if distances else None,
                "retrieved_preview": [t[:80] for t in texts[:3]],
            })

        return results

    def compute_metrics(self, eval_results):
        """计算整体评估指标。"""
        recalls = [r["recall"] for r in eval_results]
        full_recalls = [1 if r["recall"] == 1.0 else 0 for r in eval_results]

        metrics = {
            "avg_recall": sum(recalls) / len(recalls),
            "perfect_recall_rate": sum(full_recalls) / len(full_recalls),
            "total_questions": len(eval_results),
        }

        # 按难度分组
        for difficulty in ["easy", "medium", "hard"]:
            subset = [r for r in eval_results if r["difficulty"] == difficulty]
            if subset:
                metrics[f"recall_{difficulty}"] = sum(r["recall"] for r in subset) / len(subset)

        return metrics


# ============================================================
# 报告生成
# ============================================================
def print_report(eval_results, metrics, top_k):
    """打印详细的评估报告。"""
    print(f"\n{'='*70}")
    print(f"RAG 检索质量报告 (top_k={top_k})")
    print(f"{'='*70}")

    # 详细结果
    print(f"\n{'ID':<5} {'难度':<8} {'召回率':>8} {'关键词命中':>12} {'问题'}")
    print("-" * 70)

    for r in eval_results:
        status = "✓" if r["recall"] == 1.0 else "△" if r["recall"] > 0 else "✗"
        hits = f"{len(r['found_keywords'])}/{len(r['expected_keywords'])}"
        print(f"{r['id']:<5} {r['difficulty']:<8} {r['recall']:>7.0%} {hits:>12} {status} {r['question'][:35]}")

    # 汇总指标
    print(f"\n{'='*70}")
    print("汇总指标")
    print(f"{'='*70}")
    print(f"  平均召回率:        {metrics['avg_recall']:.1%}")
    print(f"  完美召回率:        {metrics['perfect_recall_rate']:.1%}")
    print(f"  测试问题数:        {metrics['total_questions']}")

    for difficulty in ["easy", "medium", "hard"]:
        key = f"recall_{difficulty}"
        if key in metrics:
            print(f"  {difficulty} 难度召回率:  {metrics[key]:.1%}")

    # 问题诊断
    print(f"\n{'='*70}")
    print("诊断建议")
    print(f"{'='*70}")

    failed = [r for r in eval_results if r["recall"] == 0]
    partial = [r for r in eval_results if 0 < r["recall"] < 1.0]

    if failed:
        print(f"\n完全未命中的问题 ({len(failed)} 个):")
        for r in failed:
            print(f"  ✗ {r['id']}: {r['question']}")
            print(f"    期望关键词: {r['expected_keywords']}")
            print(f"    检索结果预览: {r['retrieved_preview'][0][:60]}...")
        print("\n  → 建议: 增加 top_k、检查文档是否包含相关内容、尝试不同的分块策略")

    if partial:
        print(f"\n部分命中的问题 ({len(partial)} 个):")
        for r in partial:
            missed = set(r["expected_keywords"]) - set(r["found_keywords"])
            print(f"  △ {r['id']}: {r['question']}")
            print(f"    命中: {r['found_keywords']}, 缺失: {list(missed)}")
        print("\n  → 建议: 增加 chunk_overlap、尝试 Multi-Query 检索")

    if not failed and not partial:
        print("\n  ✓ 所有测试问题均完美命中！检索质量优秀。")

    if metrics["avg_recall"] < 0.6:
        print("\n⚠️ 整体召回率偏低，可能的原因：")
        print("  1. chunk_size 太大，关键信息被稀释")
        print("  2. Embedding 模型对当前文档领域效果不佳")
        print("  3. 文档中缺少测试问题对应的内容")


# ============================================================
# top_k 对比实验
# ============================================================
def top_k_comparison(evaluator, test_suite):
    """对比不同 top_k 值对检索质量的影响。"""
    print(f"\n{'='*70}")
    print("top_k 对比实验")
    print(f"{'='*70}")

    print(f"\n{'top_k':>8} {'平均召回率':>12} {'完美召回率':>12}")
    print("-" * 35)

    for k in [1, 2, 3, 5, 8, 10]:
        results = evaluator.evaluate_test_suite(test_suite, top_k=k)
        metrics = evaluator.compute_metrics(results)
        print(f"{k:>8} {metrics['avg_recall']:>11.1%} {metrics['perfect_recall_rate']:>11.1%}")

    print("\n提示：")
    print("  - top_k 太小：容易漏掉相关信息")
    print("  - top_k 太大：引入无关内容，增加 LLM 处理负担")
    print("  - 推荐配合 Reranker：先检索 top_20，再重排取 top_5")


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索质量测试")
    parser.add_argument("--auto-ingest", action="store_true", help="自动入库测试文档")
    parser.add_argument("--collection", default="rag_docs", help="ChromaDB collection 名称")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB 路径")
    parser.add_argument("--top-k", type=int, default=5, help="检索结果数量")
    parser.add_argument("--compare-k", action="store_true", help="运行 top_k 对比实验")
    args = parser.parse_args()

    # 自动入库
    if args.auto_ingest:
        print("自动入库测试文档...")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        for fname in ["rag_knowledge_base.txt", "sample_rag_test.txt"]:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                ingest_file(fpath, args.collection, args.db_path)
                print(f"  已入库: {fname}")

    # 初始化
    embeddings = get_embeddings()
    client = chromadb.PersistentClient(path=args.db_path)

    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(f"\n错误：找不到 collection '{args.collection}'")
        print("请先运行入库命令，或使用 --auto-ingest 参数：")
        print(f"  python 02_rag/document_loader.py --file data/rag_knowledge_base.txt")
        print(f"  python 02_rag/retrieval_quality_test.py --auto-ingest")
        return

    print(f"\nCollection: {args.collection}, 文档数: {collection.count()}")

    evaluator = RetrievalEvaluator(collection, embeddings)

    # 运行评估
    results = evaluator.evaluate_test_suite(TEST_SUITE, top_k=args.top_k)
    metrics = evaluator.compute_metrics(results)
    print_report(results, metrics, args.top_k)

    # top_k 对比
    if args.compare_k:
        top_k_comparison(evaluator, TEST_SUITE)


if __name__ == "__main__":
    main()
