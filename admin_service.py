"""
管理端 Web 服务 - 文档管理 / 配置 / 索引 / 问答测试 / Webhook / 健康检查

端点：
  GET  /                    后台管理页面
  GET  /health              健康检查
  GET  /webhook/douyin      抖音 Webhook 验证
  POST /webhook/douyin      抖音 Webhook 消息事件
  POST /upload              文档上传
  DELETE /documents/{name}  删除文档
  GET  /config              获取配置
  POST /config              更新配置
  GET  /stats               索引统计
  POST /rebuild-index       重建索引
  POST /query               问答测试
  GET  /sources             文档列表
"""
import os
import json
import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from dotenv import load_dotenv, set_key
from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vector_index import VectorIndex
from rag_engine import RAGEngine, RAGConfig

load_dotenv()

app = FastAPI(title="Laike RAG - 管理端", version="1.0.0")

_index: Optional[VectorIndex] = None
_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(config=RAGConfig.from_env())
    return _engine


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index


class QueryRequest(BaseModel):
    question: str

class ConfigUpdate(BaseModel):
    key: str
    value: str


ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".csv"}

CONFIG_KEYS = {
    "DEEPSEEK_MODEL": "大模型名称",
    "LLM_TEMPERATURE": "模型温度 (0-2)",
    "CHUNK_SIZE": "分片大小",
    "CHUNK_OVERLAP": "分片重叠",
    "RECALL_TOP_N": "召回数量",
    "RERANK_TOP_K": "重排数量",
    "MAX_CONTEXT_CHARS": "最大上下文字符数",
    "EMBEDDING_PROVIDER": "Embedding 方案 (tfidf/api/local)",
}


# ══════════════════════════════════════════════════════════════
#  健康检查
# ══════════════════════════════════════════════════════════════

@app.get("/health")
@app.post("/health")
async def health():
    index_loaded = False
    try:
        get_index().load()
        index_loaded = True
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "laike-rag-admin",
        "timestamp": datetime.now().isoformat(),
        "index_loaded": index_loaded,
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


# ══════════════════════════════════════════════════════════════
#  抖音 Webhook
# ══════════════════════════════════════════════════════════════

@app.get("/webhook/douyin")
async def webhook_verify(challenge: str = Query(default="")):
    if not challenge:
        raise HTTPException(status_code=400, detail="缺少 challenge 参数")
    print(f"[Webhook] 验证请求: challenge={challenge}")
    return JSONResponse(content={"challenge": challenge})


@app.post("/webhook/douyin")
async def webhook_event(request: Request):
    t0 = time.time()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体")

    print(f"[Webhook] 收到消息: {json.dumps(body, ensure_ascii=False, default=str)[:500]}")

    content = body.get("content") or body.get("text") or body.get("message") or body.get("msg") or ""

    if not content:
        return JSONResponse(content={"code": 0, "msg": "消息内容为空", "reply": ""})

    engine = get_engine()
    try:
        result = engine.ask(content, verbose=False)
    except Exception as e:
        print(f"[Webhook] RAG 引擎异常: {e}")
        return JSONResponse(content={"code": 1, "msg": str(e), "reply": "抱歉，暂时无法处理。"})

    elapsed = round((time.time() - t0) * 1000)
    print(f"[Webhook] 回答: {result['answer'][:100]}... 耗时 {elapsed}ms")

    return {
        "code": 0, "msg": "success",
        "reply": result["answer"],
        "sources": result.get("sources", []),
        "elapsed_ms": elapsed,
    }


# ══════════════════════════════════════════════════════════════
#  文档管理
# ══════════════════════════════════════════════════════════════

@app.post("/upload")
async def admin_upload(files: List[UploadFile] = File(...)):
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    uploaded, skipped = [], []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            skipped.append({"name": f.filename, "reason": f"不支持的类型 {suffix}"})
            continue

        file_path = docs_dir / f.filename
        content = await f.read()
        file_path.write_bytes(content)
        uploaded.append({"name": f.filename, "size": len(content), "type": suffix})

    if uploaded:
        rebuild_result = await admin_rebuild_index()
        return {"status": "ok", "uploaded": uploaded, "skipped": skipped, "total": len(uploaded), "rebuild": rebuild_result}

    return {"status": "ok", "uploaded": uploaded, "skipped": skipped, "total": len(uploaded)}


@app.delete("/documents/{name}")
async def admin_delete_document(name: str):
    file_path = Path("docs") / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {name}")
    file_path.unlink()
    rebuild_result = await admin_rebuild_index()
    return {"status": "ok", "msg": f"已删除: {name}", "rebuild": rebuild_result}


@app.get("/sources")
async def admin_sources():
    docs_dir = Path("docs")
    files = []
    if docs_dir.exists():
        for f in sorted(docs_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                stat = f.stat()
                files.append({
                    "name": f.name, "path": str(f), "size": stat.st_size,
                    "type": f.suffix, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
    return {"status": "ok", "count": len(files), "files": files}


# ══════════════════════════════════════════════════════════════
#  配置管理
# ══════════════════════════════════════════════════════════════

@app.get("/config")
async def admin_get_config():
    items = {}
    for key, desc in CONFIG_KEYS.items():
        items[key] = {"value": os.getenv(key, ""), "description": desc}
    return {"status": "ok", "config": items}


@app.post("/config")
async def admin_update_config(req: ConfigUpdate):
    if req.key not in CONFIG_KEYS:
        raise HTTPException(status_code=400, detail=f"未知配置项: {req.key}")

    env_path = ".env"
    set_key(env_path, req.key, req.value)
    os.environ[req.key] = req.value

    return {"status": "ok", "msg": f"已更新 {req.key} = {req.value}"}


# ══════════════════════════════════════════════════════════════
#  索引与问答
# ══════════════════════════════════════════════════════════════

@app.get("/stats")
async def admin_stats():
    idx = get_index()
    try:
        idx.load()
        stats = idx.stats()
    except FileNotFoundError:
        stats = {"error": "向量库不存在，请先构建索引"}

    return {
        "status": "ok",
        "index": stats,
        "config": {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "tfidf"),
        },
    }


@app.post("/rebuild-index")
async def admin_rebuild_index():
    from document_splitter import DocumentSplitter, SplitterConfig

    t0 = time.time()

    splitter = DocumentSplitter(config=SplitterConfig.from_env())
    chunks = splitter.load_and_split()

    if not chunks:
        return JSONResponse(content={
            "status": "ok", "msg": "docs/ 目录无文档",
            "chunks": 0, "elapsed_ms": round((time.time() - t0) * 1000),
        })

    idx = get_index()
    # 清除 TF-IDF 缓存，确保全新拟合
    cache_path = Path("knowledge_base/tfidf_vectorizer.pkl")
    if cache_path.exists():
        cache_path.unlink()
    idx.build_index(chunks, clear_existing=True)
    stats = idx.stats()

    return {
        "status": "ok", "msg": "索引重建完成",
        "chunks": len(chunks), "index": stats,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


@app.post("/query")
async def admin_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    engine = get_engine()
    result = engine.ask(req.question, verbose=False)

    return {
        "status": "ok",
        "question": result["question"],
        "answer": result["answer"],
        "stages": result["stages"],
        "sources": result["sources"],
    }


# ══════════════════════════════════════════════════════════════
#  前端页面
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def admin_page():
    return FileResponse("static/admin.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════════════════════
#  启动入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ADMIN_PORT", 8001))
    host = os.getenv("WEB_HOST", "0.0.0.0")

    print(f"\n  Laike RAG 管理端启动")
    print(f"  管理页: http://{host}:{port}")
    print(f"  API 文档: http://{host}:{port}/docs\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
