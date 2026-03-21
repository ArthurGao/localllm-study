"""
RAG vs 无 RAG 对比实验
=====================
功能：
1. 同一问题分别用 RAG 和无 RAG 方式回答
2. 测试事实性问题、文档特定问题、开放性问题
3. 展示 RAG 的优势和局限
4. 帮助理解 RAG 何时有用、何时多余

学习重点：
- RAG 在哪些场景下明显优于直接问 LLM
- RAG 的局限性（文档中没有的知识无法回答）
- 理解"开卷考试"vs"闭卷考试"的区别

使用方式：
    # 先入库文档（如果还没有）
    python 02_rag/document_loader.py --file data/rag_knowledge_base.txt
    # 运行对比
    python 02_rag/rag_comparison.py

    # 自动入库并运行
    python 02_rag/rag_comparison.py --auto-ingest
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import chat, get_llm_config, get_embeddings
from document_loader import ingest_file
from rag_chain import get_retriever, create_rag_chain


# ============================================================
# 测试问题分类
# ============================================================
TEST_QUESTIONS = {
    "文档内事实题（RAG 应该更好）": [
        "nomic-embed-text 的向量维度是多少？最大 Token 限制是多少？",
        "ChromaDB 支持哪些距离度量方式？",
        "RAG 的三个主要阶段分别是什么？每个阶段的作用是什么？",
        "什么是 Reranking？它用的是什么模型？",
    ],
    "文档外通用题（LLM 本身就知道）": [
        "Python 中 list 和 tuple 的区别是什么？",
        "什么是梯度下降算法？",
    ],
    "文档中没有的问题（都应该说不知道）": [
        "Westpac 银行的 FTH 系统使用什么编程语言？",
        "2025年最新的 GPT-5 模型有什么特点？",
    ],
}


# ============================================================
# 对比运行
# ============================================================
def run_comparison(chain, question):
    """对一个问题分别运行无 RAG 和有 RAG，返回对比结果。"""
    cfg = get_llm_config()

    # 1. 无 RAG：直接问 LLM
    t0 = time.time()
    no_rag_answer = chat(
        [{"role": "user", "content": question}],
        temperature=0.1,
    )
    no_rag_time = time.time() - t0

    # 2. 有 RAG：检索后问 LLM
    t0 = time.time()
    rag_result = chain.invoke({"query": question})
    rag_answer = rag_result["result"]
    rag_time = time.time() - t0

    # 3. 提取检索到的来源
    sources = []
    for doc in rag_result.get("source_documents", []):
        sources.append(doc.page_content[:100])

    return {
        "question": question,
        "no_rag_answer": no_rag_answer,
        "no_rag_time": no_rag_time,
        "rag_answer": rag_answer,
        "rag_time": rag_time,
        "sources": sources,
    }


def print_comparison(result, idx):
    """美化输出单个对比结果。"""
    print(f"\n{'='*70}")
    print(f"问题 {idx}: {result['question']}")
    print(f"{'='*70}")

    print(f"\n--- 无 RAG（直接问 LLM）| 耗时 {result['no_rag_time']:.1f}s ---")
    # 截取前 300 字符
    answer = result["no_rag_answer"]
    if len(answer) > 400:
        answer = answer[:400] + "...(截断)"
    print(answer)

    print(f"\n--- 有 RAG（检索后问 LLM）| 耗时 {result['rag_time']:.1f}s ---")
    answer = result["rag_answer"]
    if len(answer) > 400:
        answer = answer[:400] + "...(截断)"
    print(answer)

    if result["sources"]:
        print(f"\n--- 检索到的参考来源 ({len(result['sources'])} 个) ---")
        for i, src in enumerate(result["sources"][:3]):
            print(f"  [{i+1}] {src}...")


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG vs 无 RAG 对比实验")
    parser.add_argument("--auto-ingest", action="store_true", help="自动入库测试文档")
    parser.add_argument("--collection", default="rag_docs", help="ChromaDB collection")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB 路径")
    parser.add_argument("--category", type=int, help="只测试指定分类 (1-3)")
    args = parser.parse_args()

    cfg = get_llm_config()
    print(f"当前后端: {cfg['backend']}, 模型: {cfg['model']}")

    # 自动入库
    if args.auto_ingest:
        print("\n自动入库测试文档...")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        for fname in ["rag_knowledge_base.txt", "sample_rag_test.txt"]:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                ingest_file(fpath, args.collection, args.db_path)

    # 构建 RAG Chain
    try:
        retriever = get_retriever(args.db_path, args.collection)
        chain = create_rag_chain(retriever)
    except Exception as e:
        print(f"\n错误：无法创建 RAG Chain: {e}")
        print("请先入库文档或使用 --auto-ingest")
        return

    # 运行对比
    categories = list(TEST_QUESTIONS.items())
    if args.category:
        categories = [categories[args.category - 1]]

    total_idx = 0
    for cat_name, questions in categories:
        print(f"\n\n{'#'*70}")
        print(f"# 分类: {cat_name}")
        print(f"{'#'*70}")

        for q in questions:
            total_idx += 1
            result = run_comparison(chain, q)
            print_comparison(result, total_idx)

    # 总结
    print(f"\n\n{'='*70}")
    print("实验总结")
    print(f"{'='*70}")
    print("""
观察要点：
1. 文档内事实题：RAG 应该给出更准确、有依据的答案
   - 无 RAG 可能编造数字或给出过时信息
   - 有 RAG 的答案应与文档内容一致

2. 文档外通用题：RAG 可能反而表现更差
   - LLM 本身就知道的常识，加入不相关的检索结果反而干扰
   - 这说明 RAG 不是万能的，需要合理的路由策略

3. 文档中没有的问题：
   - 好的 RAG 系统应该说"根据文档无法回答"
   - 如果还是编造答案，说明 Prompt 约束不够或 temperature 太高

结论：
- RAG 最适合：文档特定的事实性查询
- RAG 不适合：通用知识、需要推理的问题
- 进阶方案：Query Router 自动判断是否需要 RAG
""")


if __name__ == "__main__":
    main()
