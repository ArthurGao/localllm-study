"""
RAG 检索增强生成链
==================
功能：
1. 使用 LangChain 构建 RAG 管道
2. 支持 Ollama（本地）和 Groq（云端）两种 LLM 后端
3. ChromaDB 检索 + LLM 生成
4. 支持自定义 Prompt Template
5. 支持流式输出

使用方式：
    conda activate llm-learn
    python 02_rag/rag_chain.py --query "你的问题"

后端切换：
    修改 .env 文件中的 LLM_BACKEND=ollama 或 LLM_BACKEND=groq
"""

import sys
import os
import argparse

# 让 import config 能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_llm_config, get_embeddings

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# ============================================================
# 1. 初始化向量检索器
# ============================================================
def get_retriever(
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_docs",
    embedding_model: str = "nomic-embed-text",
    top_k: int = 3,
):
    """
    从已有的 ChromaDB 创建检索器。

    核心概念：
    - Retriever: LangChain 的检索抽象，封装了"问题 → 相关文档"的过程
    - search_kwargs["k"]: 返回的文档数量，越多上下文越丰富但也越冗余
    - Embedding 根据后端自动选择: Ollama 用 nomic-embed-text, Groq 用 sentence-transformers
    """
    embeddings = get_embeddings(model=embedding_model)

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    print(f"[检索器就绪] collection={collection_name}, top_k={top_k}")
    return retriever


# ============================================================
# 2. 构建 Prompt Template
# ============================================================
RAG_PROMPT_TEMPLATE = """你是一个专业的问答助手。请根据以下参考资料回答用户问题。

规则：
1. 只根据提供的参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请明确说明"根据提供的资料，我无法回答这个问题"
3. 回答时引用参考资料的关键内容
4. 用中文回答，简洁准确

参考资料：
{context}

用户问题：{question}

回答："""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ============================================================
# 3. 获取 LangChain LLM（支持 Ollama / Groq）
# ============================================================
def get_langchain_llm(
    temperature: float = 0.1,
    num_ctx: int = 8192,
    streaming: bool = False,
):
    """
    根据 .env 配置返回对应的 LangChain LLM 实例。

    - Ollama: 使用 langchain_ollama.OllamaLLM
    - Groq: 使用 langchain_groq.ChatGroq
    """
    cfg = get_llm_config()
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

    if cfg["backend"] == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=cfg["model"],
            api_key=cfg["groq_api_key"],
            temperature=temperature,
            callbacks=callbacks,
        )
        print(f"[LLM 就绪] Groq: {cfg['model']}, temperature={temperature}")
    else:
        llm = OllamaLLM(
            model=cfg["model"],
            temperature=temperature,
            num_ctx=num_ctx,
            callbacks=callbacks,
        )
        print(f"[LLM 就绪] Ollama: {cfg['model']}, temperature={temperature}, num_ctx={num_ctx}")

    return llm


# ============================================================
# 4. 构建 RAG Chain
# ============================================================
def create_rag_chain(
    retriever,
    model: str = None,
    temperature: float = 0.1,
    num_ctx: int = 8192,
    streaming: bool = False,
):
    """
    组装完整的 RAG 链: 检索 → 组装 Prompt → LLM 生成。

    核心概念：
    - RetrievalQA: LangChain 提供的 RAG 链，自动完成检索 + 生成
    - chain_type="stuff": 最简单的策略，把所有检索到的文档塞进 prompt
    - return_source_documents=True: 返回检索到的原始文档，便于追溯
    - temperature=0.1: RAG 任务用低温度，减少幻觉

    注意：model 参数已废弃，模型由 .env 中的配置决定。
    """
    llm = get_langchain_llm(temperature=temperature, num_ctx=num_ctx, streaming=streaming)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将所有检索文档拼接到 prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt},
    )

    return chain


# ============================================================
# 5. 执行 RAG 问答
# ============================================================
def ask(chain, question: str) -> dict:
    """
    执行一次 RAG 问答，返回答案和来源文档。

    返回格式：
    {
        "query": "用户问题",
        "result": "生成的答案",
        "source_documents": [Document, ...]
    }
    """
    result = chain.invoke({"query": question})

    print(f"\n{'='*50}")
    print(f"问题: {question}")
    print(f"{'='*50}")
    print(f"\n答案:\n{result['result']}")

    if result.get("source_documents"):
        print(f"\n--- 参考来源 ({len(result['source_documents'])} 个) ---")
        for i, doc in enumerate(result["source_documents"]):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i+1}] {source} (p.{page}): {preview}...")

    return result


# ============================================================
# 6. 交互式 RAG 问答
# ============================================================
def interactive_rag(chain):
    """命令行交互式 RAG 问答。"""
    cfg = get_llm_config()
    print(f"\n{'='*50}")
    print(f"RAG 问答系统 - 后端: {cfg['backend']}, 模型: {cfg['model']}")
    print("输入 /quit 退出")
    print(f"{'='*50}\n")

    while True:
        try:
            question = input("你的问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not question:
            continue
        if question == "/quit":
            print("再见！")
            break

        ask(chain, question)
        print()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索增强生成")
    parser.add_argument("--query", type=str, help="单次提问（不提供则进入交互模式）")
    parser.add_argument("--collection", type=str, default="rag_docs", help="ChromaDB collection")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="ChromaDB 路径")
    parser.add_argument("--top-k", type=int, default=3, help="检索文档数量")
    parser.add_argument("--stream", action="store_true", help="启用流式输出")

    args = parser.parse_args()

    retriever = get_retriever(args.db_path, args.collection, top_k=args.top_k)
    chain = create_rag_chain(retriever, streaming=args.stream)

    if args.query:
        ask(chain, args.query)
    else:
        interactive_rag(chain)
