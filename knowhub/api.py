"""Single-workspace operator API. Bind localhost unless secured by an external gateway."""

import io
import os
import secrets
import threading
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    ticket_id: str | None = Field(default=None, max_length=50)
    mode: Literal["agent", "rag"] = "agent"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool


def create_app(store, agent, token):
    if not token:
        raise RuntimeError("KNOWHUB_API_TOKEN must be configured")
    app = FastAPI(title="KnowHub", version="0.2.0", docs_url=None, redoc_url=None, openapi_url=None)
    # Bound local concurrency: model calls can be slow, so fail clearly rather than grow a queue.
    capacity = threading.BoundedSemaphore(2)

    def authorize(authorization: str | None = Header(default=None)):
        expected = "Bearer " + token
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(401, "需要有效的操作员访问令牌")

    protected = [Depends(authorize)]

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": agent.model.name}

    @app.get("/api/tickets", dependencies=protected)
    def tickets():
        return store.tickets()

    @app.get("/api/documents", dependencies=protected)
    def documents():
        return store.documents()

    @app.post("/api/documents/upload", dependencies=protected)
    async def upload(file: Annotated[UploadFile, File()]):
        name = Path((file.filename or "document").replace("\\", "/")).name[:160]
        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".md", ".txt"}:
            raise HTTPException(415, "仅支持 PDF、Markdown 和 TXT")
        try:
            raw = await file.read(5 * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "文件超过5MB限制")

        def ingest():
            try:
                pages = None
                if suffix == ".pdf":
                    reader = PdfReader(io.BytesIO(raw))
                    if reader.is_encrypted or len(reader.pages) > 200:
                        raise ValueError("Encrypted or oversized PDF")
                    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
                    text = "\n".join(t for _, t in pages)
                else:
                    text = raw.decode("utf-8-sig")
                if len(text) > 500_000:
                    raise ValueError("Too much extracted text")
                return agent.retriever.ingest(name, text, "uploads/" + name, pages=pages)
            except Exception:
                raise HTTPException(422, "无法索引文件：请检查格式、可提取文字和向量模型服务") from None

        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "当前正在处理其他任务，请稍后重试")
        try:
            return await run_in_threadpool(ingest)
        finally:
            capacity.release()

    @app.post("/api/runs", dependencies=protected)
    def run(body: RunRequest):
        if not body.question.strip():
            raise HTTPException(422, "问题不能为空")
        if body.ticket_id and not store.ticket(body.ticket_id):
            raise HTTPException(404, "工单不存在")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "当前正在处理其他任务，请稍后重试")
        try:
            return agent.run(body.question, body.ticket_id, body.mode)
        finally:
            capacity.release()

    @app.get("/api/runs", dependencies=protected)
    def runs():
        return store.runs()

    @app.get("/api/runs/{run_id}", dependencies=protected)
    def get_run(run_id: str):
        result = store.run(run_id)
        if not result:
            raise HTTPException(404, "运行记录不存在")
        return result

    @app.post("/api/runs/{run_id}/decision", dependencies=protected)
    def decide(run_id: str, body: Decision):
        try:
            return store.approve_run(run_id, body.approved)
        except KeyError:
            raise HTTPException(404, "运行记录不存在") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    dist = Path(__file__).resolve().parent.parent / "frontend/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="workspace")
    return app


def build_app():
    from dotenv import load_dotenv

    from knowhub.agent import Agent
    from knowhub.model import Model
    from knowhub.retrieval import Embedder, Retriever
    from knowhub.seed import seed
    from knowhub.store import Store

    load_dotenv()
    token = os.environ.get("KNOWHUB_API_TOKEN", "")
    if len(token) < 24:
        raise RuntimeError("Set a random KNOWHUB_API_TOKEN with at least24 characters")
    store = Store(
        os.environ.get(
            "KNOWHUB_DATABASE_URL", "postgresql://knowhub:knowhub-local-only@localhost:55439/knowhub"
        )
    )
    store.initialize()
    embedder = Embedder(os.environ.get("KNOWHUB_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    retriever = Retriever(store, embedder)
    model = Model(session_limit=int(os.environ.get("KNOWHUB_SESSION_MODEL_LIMIT", "40")))
    seed(store, retriever)
    return create_app(store, Agent(store, retriever, model), token)
