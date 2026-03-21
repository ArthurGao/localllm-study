"""
Embedding 向量探索器
===================
功能：
1. 理解 Embedding 向量的本质 - 维度、数值范围、归一化
2. 语义相似度实验 - 相似文本 vs 不相关文本
3. 中英文混合相似度测试
4. 同义词和近义词的向量距离
5. 向量运算：加法、平均、类比

学习重点：
- 直观理解 Embedding 是什么
- 理解余弦相似度的含义
- 掌握 Embedding 在 RAG 中的作用

使用方式：
    conda activate llm-learn
    python 02_rag/embedding_explorer.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_embeddings
import numpy as np


# ============================================================
# 工具函数
# ============================================================
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度。"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_distance(a, b):
    """余弦距离 = 1 - 余弦相似度。"""
    return 1 - cosine_similarity(a, b)


def print_similarity_matrix(texts, vectors):
    """打印文本间的相似度矩阵。"""
    n = len(texts)
    # 打印表头
    short_names = [t[:12] + "..." if len(t) > 15 else t for t in texts]
    header = f"{'':>16}" + "".join(f"{s:>16}" for s in short_names)
    print(header)
    print("-" * len(header))

    for i in range(n):
        row = f"{short_names[i]:>16}"
        for j in range(n):
            sim = cosine_similarity(vectors[i], vectors[j])
            row += f"{sim:>16.4f}"
        print(row)


# ============================================================
# 实验 1：Embedding 基础属性
# ============================================================
def experiment_1_basics(embeddings):
    """
    实验1：了解 Embedding 向量的基础属性。
    - 向量维度是什么意思
    - 数值范围和分布
    - 归一化特性
    """
    print(f"\n{'='*60}")
    print("实验 1：Embedding 基础属性")
    print(f"{'='*60}")

    text = "RAG 是检索增强生成技术"
    vector = embeddings.embed_query(text)
    v = np.array(vector)

    print(f"\n输入文本: '{text}'")
    print(f"向量维度: {len(vector)}")
    print(f"数值范围: [{v.min():.4f}, {v.max():.4f}]")
    print(f"平均值: {v.mean():.6f}")
    print(f"标准差: {v.std():.4f}")
    print(f"L2 范数: {np.linalg.norm(v):.4f}")
    print(f"前 10 个值: {v[:10].round(4).tolist()}")

    # 检查是否已归一化
    norm = np.linalg.norm(v)
    is_normalized = abs(norm - 1.0) < 0.01
    print(f"\n是否已归一化: {'是' if is_normalized else '否'} (范数={norm:.4f})")

    if is_normalized:
        print("  → 归一化后，余弦相似度 = 向量内积（dot product）")
    else:
        print("  → 未归一化，计算相似度时需要除以范数")


# ============================================================
# 实验 2：语义相似度
# ============================================================
def experiment_2_similarity(embeddings):
    """
    实验2：测试语义相似的文本向量是否更接近。
    核心问题：Embedding 模型真的能理解语义吗？
    """
    print(f"\n{'='*60}")
    print("实验 2：语义相似度测试")
    print(f"{'='*60}")

    # 准备测试文本对
    test_pairs = [
        ("语义相似对", [
            ("RAG 是检索增强生成技术", "Retrieval-Augmented Generation 是一种 AI 方法"),
            ("机器学习模型需要大量训练数据", "训练 AI 系统需要准备很多数据"),
            ("向量数据库用于存储 Embedding", "ChromaDB 可以保存向量化后的文本"),
        ]),
        ("语义不相关对", [
            ("RAG 是检索增强生成技术", "今天的天气非常好"),
            ("机器学习模型需要大量训练数据", "这家餐厅的菜很好吃"),
            ("向量数据库用于存储 Embedding", "明天有一场足球比赛"),
        ]),
    ]

    for group_name, pairs in test_pairs:
        print(f"\n--- {group_name} ---")
        for text1, text2 in pairs:
            v1 = embeddings.embed_query(text1)
            v2 = embeddings.embed_query(text2)
            sim = cosine_similarity(v1, v2)
            print(f"  相似度 {sim:.4f} | '{text1[:20]}...' vs '{text2[:20]}...'")


# ============================================================
# 实验 3：同义词和近义词
# ============================================================
def experiment_3_synonyms(embeddings):
    """
    实验3：测试同义词、近义词、反义词在向量空间中的距离。
    理解 Embedding 对词汇语义的捕捉能力。
    """
    print(f"\n{'='*60}")
    print("实验 3：同义词 / 近义词 / 反义词")
    print(f"{'='*60}")

    base_text = "大语言模型"
    comparisons = [
        ("LLM", "英文缩写"),
        ("Large Language Model", "英文全称"),
        ("ChatGPT", "具体产品"),
        ("人工智能", "上位概念"),
        ("深度学习", "相关技术"),
        ("机器学习", "更广义的领域"),
        ("数据库", "不太相关的技术"),
        ("做饭", "完全不相关"),
    ]

    base_vec = embeddings.embed_query(base_text)

    print(f"\n基准文本: '{base_text}'")
    print(f"{'比较文本':<25} {'关系':<12} {'相似度':>8}")
    print("-" * 50)

    for text, relation in comparisons:
        vec = embeddings.embed_query(text)
        sim = cosine_similarity(base_vec, vec)
        # 相似度柱状图
        bar = "█" * int(sim * 30)
        print(f"  {text:<23} {relation:<12} {sim:>.4f} {bar}")


# ============================================================
# 实验 4：查询 vs 文档 的语义匹配
# ============================================================
def experiment_4_query_vs_docs(embeddings):
    """
    实验4：模拟 RAG 检索场景。
    测试查询（问题形式）能否匹配到文档（陈述形式）。
    这是 RAG 的核心挑战之一。
    """
    print(f"\n{'='*60}")
    print("实验 4：查询 vs 文档匹配（RAG 核心场景）")
    print(f"{'='*60}")

    # 模拟文档块
    doc_chunks = [
        "RAG 解决了大语言模型的两个关键问题：知识时效性和幻觉问题。",
        "ChromaDB 是一个轻量级的开源向量数据库，支持本地运行和持久化存储。",
        "文本分块是 RAG 中至关重要的一步，分块质量直接影响检索效果。",
        "Embedding 模型负责将文本转换为固定维度的向量表示。",
        "temperature=0.1 是 RAG 场景推荐的设置，可以减少幻觉。",
    ]

    # 用户查询
    queries = [
        "RAG 能解决什么问题？",
        "有什么好用的向量数据库？",
        "怎么做文本分块？",
        "什么是 Embedding？",
        "RAG 应该用多高的 temperature？",
    ]

    # 计算所有文档的向量
    doc_vecs = embeddings.embed_documents(doc_chunks)

    print("\n查询与文档的匹配矩阵（越高越匹配）：")
    print(f"\n{'查询':<32}", end="")
    for i in range(len(doc_chunks)):
        print(f"  文档{i+1}", end="")
    print("  最佳匹配")
    print("-" * (32 + 8 * len(doc_chunks) + 10))

    for q_idx, query in enumerate(queries):
        q_vec = embeddings.embed_query(query)
        sims = [cosine_similarity(q_vec, dv) for dv in doc_vecs]
        best_idx = np.argmax(sims)

        print(f"  {query:<30}", end="")
        for i, sim in enumerate(sims):
            marker = " ★" if i == best_idx else "  "
            print(f" {sim:.3f}{marker}", end="")
        match = "✓" if best_idx == q_idx else "✗"
        print(f"  doc{best_idx+1} {match}")

    # 统计正确匹配率
    correct = 0
    for q_idx, query in enumerate(queries):
        q_vec = embeddings.embed_query(query)
        sims = [cosine_similarity(q_vec, dv) for dv in doc_vecs]
        if np.argmax(sims) == q_idx:
            correct += 1

    print(f"\n正确匹配率: {correct}/{len(queries)} = {correct/len(queries):.0%}")
    print("(每个查询是否能正确找到对应的文档块)")


# ============================================================
# 实验 5：向量运算（加法和类比）
# ============================================================
def experiment_5_vector_arithmetic(embeddings):
    """
    实验5：Embedding 向量的算术运算。
    经典实验：king - man + woman ≈ queen
    测试中文场景下类似的关系是否成立。
    """
    print(f"\n{'='*60}")
    print("实验 5：向量运算和语义类比")
    print(f"{'='*60}")

    # 准备候选词
    candidates = [
        "国王", "王后", "男人", "女人", "王子", "公主",
        "医生", "护士", "学生", "老师",
        "Python", "Java", "编程", "数据库",
    ]
    candidate_vecs = {w: np.array(embeddings.embed_query(w)) for w in candidates}

    # 类比实验
    analogies = [
        ("国王", "男人", "女人", "期望: 王后"),
        ("老师", "学生", "医生", "期望: 护士或病人"),
        ("Python", "编程", "数据库", "期望: 某种数据库技术"),
    ]

    for a, b, c, expected in analogies:
        va = candidate_vecs[a]
        vb = candidate_vecs[b]
        vc = candidate_vecs[c]

        # a - b + c ≈ ?
        result_vec = va - vb + vc

        # 找最接近的候选词（排除 a, b, c 自身）
        best_sim = -1
        best_word = ""
        for word, wvec in candidate_vecs.items():
            if word in (a, b, c):
                continue
            sim = cosine_similarity(result_vec, wvec)
            if sim > best_sim:
                best_sim = sim
                best_word = word

        print(f"\n  {a} - {b} + {c} = ?")
        print(f"  最接近: {best_word} (相似度: {best_sim:.4f})")
        print(f"  {expected}")


# ============================================================
# 实验 6：批量文本的相似度矩阵
# ============================================================
def experiment_6_similarity_matrix(embeddings):
    """
    实验6：对一组文本计算完整的相似度矩阵。
    帮助理解不同主题之间的语义距离。
    """
    print(f"\n{'='*60}")
    print("实验 6：相似度矩阵")
    print(f"{'='*60}")

    texts = [
        "向量数据库存储",
        "ChromaDB 使用",
        "文本分块策略",
        "chunk_size 参数",
        "LLM 幻觉问题",
        "今天天气不错",
    ]

    vecs = [embeddings.embed_query(t) for t in texts]
    print("\n相似度矩阵（对角线为1.0，越接近1.0越相似）：\n")
    print_similarity_matrix(texts, vecs)

    print("\n观察：")
    print("  - '向量数据库存储' 和 'ChromaDB 使用' 应该最相似（同主题）")
    print("  - '文本分块策略' 和 'chunk_size 参数' 应该较相似（相关话题）")
    print("  - '今天天气不错' 和其他所有文本都应该不太相似（无关话题）")


# ============================================================
# 主入口
# ============================================================
def main():
    print("Embedding 向量探索器")
    print("=" * 60)

    embeddings = get_embeddings()

    experiments = [
        ("基础属性", experiment_1_basics),
        ("语义相似度", experiment_2_similarity),
        ("同义词/近义词", experiment_3_synonyms),
        ("查询vs文档匹配", experiment_4_query_vs_docs),
        ("向量运算", experiment_5_vector_arithmetic),
        ("相似度矩阵", experiment_6_similarity_matrix),
    ]

    import argparse
    parser = argparse.ArgumentParser(description="Embedding 向量探索器")
    parser.add_argument(
        "--exp",
        type=int,
        choices=range(1, len(experiments) + 1),
        help=f"运行指定实验 (1-{len(experiments)}), 不指定则运行全部",
    )
    args = parser.parse_args()

    if args.exp:
        name, fn = experiments[args.exp - 1]
        print(f"\n运行实验 {args.exp}: {name}")
        fn(embeddings)
    else:
        for i, (name, fn) in enumerate(experiments, 1):
            print(f"\n>>> 实验 {i}/{len(experiments)}: {name}")
            fn(embeddings)

    print(f"\n{'='*60}")
    print("全部实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
