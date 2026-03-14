"""
RAG FastAPI 服务
================
功能：
1. 上传 PDF/TXT/MD 文档并入库
2. 问答接口（基于已入库文档）
3. 查看已入库文档列表

使用方式：
    conda activate llm-learn
    uvicorn 02_rag.app:app --reload --port 8000

API 文档：
    http://localhost:8000/docs
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from document_loader import ingest_file, test_retrieval
from rag_chain import get_retriever, create_rag_chain, ask


# ============================================================
# 配置
# ============================================================
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_docs"
UPLOAD_DIR = "./data/uploads"
MODEL = "qwen3:8b"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="RAG 问答系统",
    description="基于 Ollama + ChromaDB + LangChain 的本地 RAG 系统",
    version="0.1.0",
)


# ============================================================
# 请求/响应模型
# ============================================================
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    filename: str
    message: str
    chunks: int


# ============================================================
# API 路由
# ============================================================
@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并自动入库（支持 .pdf / .txt / .md）。

    流程：保存文件 → 加载 → 分块 → Embedding → 存入 ChromaDB
    """
    # 验证文件类型
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(allowed_extensions)}",
        )

    # 保存上传文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 入库
    try:
        collection = ingest_file(
            file_path,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH,
        )
        count = collection.count()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {str(e)}")

    return IngestResponse(
        filename=file.filename,
        message="文档已成功入库",
        chunks=count,
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """
    基于已入库文档进行问答。

    流程：问题 → Embedding → 检索相似文档 → Qwen3 生成答案
    """
    try:
        retriever = get_retriever(
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            top_k=req.top_k,
        )
        chain = create_rag_chain(retriever, model=MODEL)
        result = chain.invoke({"query": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
        })

    return AnswerResponse(
        question=req.question,
        answer=result["result"],
        sources=sources,
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model": MODEL, "collection": COLLECTION_NAME}


# ============================================================
# 启动提示
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("启动 RAG 服务...")
    print("API 文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
