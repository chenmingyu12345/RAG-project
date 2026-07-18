"""
用户端 Web 服务 - 仅 AI 问答

端点：
  GET  /            用户聊天页面
  POST /api/chat    用户聊天接口
"""
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import RAGEngine, RAGConfig

load_dotenv()

app = FastAPI(title="Laike RAG - 用户端", version="1.0.0")

_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(config=RAGConfig.from_env())
    return _engine


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # [{"role":"user","content":"..."}, ...]


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    engine = get_engine()
    result = engine.ask(req.message, history=req.history, verbose=False)

    return {
        "code": 0,
        "reply": result["answer"],
        "question": result["question"],
        "sources": [
            {"file": s["file"], "preview": s["preview"][:120], "score": s["rerank_score"]}
            for s in result.get("sources", [])
        ],
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("WEB_PORT", 8000))
    host = os.getenv("WEB_HOST", "0.0.0.0")

    print(f"\n  Laike RAG 用户端启动")
    print(f"  AI 问答: http://{host}:{port}\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
