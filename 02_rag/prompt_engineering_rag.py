"""
RAG Prompt 工程实验
==================
功能：
1. 对比不同 Prompt 模板对 RAG 答案质量的影响
2. 测试严格约束型、引用来源型、推理分析型三种 Prompt
3. 测试不同 temperature 对 RAG 的影响
4. 展示好 Prompt 和坏 Prompt 的差异

学习重点：
- Prompt 设计对 RAG 输出质量的决定性影响
- 如何让模型"只根据上下文回答"
- temperature 在 RAG 场景下的最佳实践

使用方式：
    # 先入库文档
    python 02_rag/document_loader.py --file data/rag_knowledge_base.txt
    # 运行实验
    python 02_rag/prompt_engineering_rag.py

    # 自动入库
    python 02_rag/prompt_engineering_rag.py --auto-ingest
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_llm_config, get_embeddings
from document_loader import ingest_file
from rag_chain import get_retriever, get_langchain_llm

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb


# ============================================================
# Prompt 模板定义
# ============================================================
PROMPTS = {
    "最简单（无约束）": PromptTemplate(
        template="""参考资料：
{context}

问题：{question}

回答：""",
        input_variables=["context", "question"],
    ),
    "严格约束型": PromptTemplate(
        template="""你是一个专业的文档问答助手。请严格遵守以下规则：

规则：
1. 只根据【参考资料】中的内容回答，不要添加自己的知识
2. 如果参考资料中找不到答案，必须回答"根据提供的资料，我无法回答这个问题"
3. 不要编造任何信息
4. 用中文简洁回答

参考资料：
{context}

问题：{question}

回答（严格基于参考资料）：""",
        input_variables=["context", "question"],
    ),
    "引用来源型": PromptTemplate(
        template="""你是一个文档问答助手，回答时必须标注信息来源。

规则：
1. 回答基于参考资料
2. 每个要点后用 [来源X] 标注来自哪段参考资料
3. 如果无法回答，说明原因

参考资料：
{context}

问题：{question}

回答（请标注来源）：""",
        input_variables=["context", "question"],
    ),
    "分步推理型": PromptTemplate(
        template="""你是一个分析型问答助手。请按以下步骤回答：

步骤：
1. 先列出参考资料中与问题相关的关键信息
2. 基于这些信息给出回答
3. 如果参考资料不足，明确指出哪些方面缺少信息

参考资料：
{context}

问题：{question}

分析过程：""",
        input_variables=["context", "question"],
    ),
    "坏的 Prompt（反面教材）": PromptTemplate(
        template="""{context}

{question}""",
        input_variables=["context", "question"],
    ),
}


# ============================================================
# 测试问题
# ============================================================
PROMPT_TEST_QUESTIONS = [
    # 文档中有答案的问题
    "ChromaDB 有什么特点？支持哪些距离度量？",
    # 需要综合多段信息的问题
    "RAG 中 Embedding 和向量数据库分别起什么作用？",
    # 文档中没有答案的问题（测试约束力）
    "GPT-4 和 Claude 哪个更好？",
]


# ============================================================
# Prompt 对比实验
# ============================================================
def compare_prompts(retriever):
    """对比不同 Prompt 模板的效果。"""
    llm = get_langchain_llm(temperature=0.1)

    for question in PROMPT_TEST_QUESTIONS:
        print(f"\n{'#'*70}")
        print(f"# 问题: {question}")
        print(f"{'#'*70}")

        for prompt_name, prompt_template in PROMPTS.items():
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template},
            )

            result = chain.invoke({"query": question})
            answer = result["result"]

            # 截断过长的回答
            if len(answer) > 300:
                answer = answer[:300] + "...(截断)"

            print(f"\n--- [{prompt_name}] ---")
            print(answer)

        print(f"\n{'~'*70}")
        print("对比观察：")
        print("  - '最简单' 和 '坏的 Prompt'：可能编造信息，无约束")
        print("  - '严格约束型'：对文档外的问题应该说'无法回答'")
        print("  - '引用来源型'：应该标注 [来源X]")
        print("  - '分步推理型'：应该先列出相关信息再回答")


# ============================================================
# Temperature 对比实验
# ============================================================
def compare_temperatures(retriever):
    """
    对比不同 temperature 对 RAG 回答的影响。

    RAG 关键洞察：
    - temperature 太低：回答死板，可能逐字复述文档
    - temperature 适中（0.1-0.3）：在忠实于文档的前提下有一定组织能力
    - temperature 太高（0.7+）：容易产生幻觉，偏离文档内容
    """
    print(f"\n{'='*70}")
    print("Temperature 对比实验")
    print(f"{'='*70}")

    question = "RAG 的核心流程是什么？每个阶段做什么？"
    print(f"\n问题: {question}")

    prompt = PROMPTS["严格约束型"]

    for temp in [0.0, 0.1, 0.3, 0.7, 1.0]:
        llm = get_langchain_llm(temperature=temp)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
        )

        result = chain.invoke({"query": question})
        answer = result["result"]
        if len(answer) > 250:
            answer = answer[:250] + "...(截断)"

        print(f"\n--- temperature={temp} ---")
        print(answer)

    print(f"\n{'='*70}")
    print("Temperature 选择建议（RAG 场景）：")
    print("  temperature=0.0 ~ 0.1  适合事实性问答（推荐默认值）")
    print("  temperature=0.2 ~ 0.3  适合需要总结归纳的问题")
    print("  temperature=0.5+       不推荐用于 RAG（幻觉风险高）")
    print(f"{'='*70}")


# ============================================================
# 主入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Prompt 工程实验")
    parser.add_argument("--auto-ingest", action="store_true", help="自动入库测试文档")
    parser.add_argument("--collection", default="rag_docs", help="ChromaDB collection")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB 路径")
    parser.add_argument(
        "--mode",
        choices=["prompts", "temperature", "all"],
        default="all",
        help="实验模式",
    )
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

    # 构建 Retriever
    try:
        retriever = get_retriever(args.db_path, args.collection)
    except Exception as e:
        print(f"\n错误：{e}")
        print("请先入库文档或使用 --auto-ingest")
        return

    if args.mode in ("prompts", "all"):
        compare_prompts(retriever)

    if args.mode in ("temperature", "all"):
        compare_temperatures(retriever)

    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
