"""
server.py — FastAPI backend for multimodal multi-agent chat.

Endpoints:
    POST /chat            → JSON response (text + optional images)
    POST /chat/stream     → SSE streaming response
    POST /chat/json       → JSON-only (images as base64 in body)
    GET  /sessions/{id}   → Session stats
    DELETE /sessions/{id} → Clear session
    GET  /health          → Health check

Usage:
    uvicorn school_agents.server:app --host 0.0.0.0 --port 8000

    # Text only
    curl -X POST http://localhost:8000/chat \
         -F 'query=Tin tức hôm nay' -F 'session_id=abc123'

    # With images (multipart)
    curl -X POST http://localhost:8000/chat \
         -F 'query=Thuốc gì đây?' -F 'session_id=abc123' \
         -F 'images=@thuoc.jpg' -F 'images=@label.png'

    # JSON body (images as base64)
    curl -X POST http://localhost:8000/chat/json \
         -H 'Content-Type: application/json' \
         -d '{"query":"Thuốc gì?","session_id":"abc123","images":[{"b64":"...","mime":"image/jpeg"}]}'

    # Streaming
    curl -N -X POST http://localhost:8000/chat/stream \
         -F 'query=Thời tiết hôm nay' -F 'session_id=abc123'
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .config import load_config, AppConfig
from .crew_runner import (
    bootstrap, route, run_crew_with_memory,
    make_openai_client, get_llm_extra_body,
)
from .conversation_memory import ConversationMemory
from .context_compressor import ContextCompressor
from .memory_bank import MemoryDB

log = logging.getLogger("school_agents.server")

# ── App init ──────────────────────────────────────────────────────────

app = FastAPI(title="BrighTO Multi-Agent Chat", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CFG: AppConfig = load_config(Path(__file__).resolve().parent / "config")
bootstrap(CFG)

# Shared memory bank (thread-safe JSONL backend)
_BANK = MemoryDB(str(Path(CFG.memory.bank_path)))

# Compressor + LLM client (shared, stateless)
_oai = make_openai_client(CFG)
_extra = get_llm_extra_body(CFG)
_model = CFG.llm.model
if _model.startswith("openai/"):
    _model = _model[len("openai/"):]

_COMPRESSOR = ContextCompressor(
    strategy=CFG.memory.compressor,
    openai_client=_oai,
    model=_model,
    extra_body=_extra,
    max_tokens=CFG.llm.structured_max_tokens,
)


def _get_memory(session_id: str) -> ConversationMemory:
    mc = CFG.memory
    return ConversationMemory(
        session_id=session_id,
        bank=_BANK,
        compressor=_COMPRESSOR,
        max_recent_turns=mc.max_recent_turns,
        max_context_tokens=mc.max_context_tokens,
        enable_facts=mc.enable_facts,
    )


# ── Image helpers ─────────────────────────────────────────────────────

async def _encode_uploads(files: list[UploadFile]) -> list[dict]:
    images = []
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            continue
        data = await f.read()
        # Convert to JPEG for VLM backend compatibility (webp etc. may not be supported)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        images.append({"b64": base64.b64encode(buf.getvalue()).decode(), "mime": "image/jpeg"})
    return images


# ── Query expansion (auto mode — no user confirmation) ────────────────

def _expand_and_search(query: str) -> str:
    mc = CFG.memory
    if not mc.expand_enabled:
        return ""
    try:
        from .tools.web_tools import expand_queries_only, expand_and_search

        expanded = expand_queries_only(
            query, _oai, _model, extra_body=_extra,
            max_tokens=CFG.llm.structured_max_tokens,
        )
        if len(expanded) <= 1:
            return ""

        merged = expand_and_search(
            user_query=query, openai_client=_oai, model=_model,
            max_results_per_query=mc.expand_max_results_per_query,
            selected_queries=expanded, extra_body=_extra,
        )
        results = merged.get("results", [])
        if not results:
            return ""

        lines = [f"Found {len(results)} results from {len(expanded)} queries:"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "")[:300]
            lines.append(f"[S{i}] {title}\n    URL: {url}\n    {content}")
        return "\n\n".join(lines)
    except Exception as exc:
        log.warning("[server:expand] Failed: %s", exc)
        return ""


# ── Core turn execution (blocking, runs in thread) ───────────────────

def _execute_turn(
    session_id: str,
    query: str,
    images: list[dict] | None = None,
    student_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    stream_callback: Any = None,
) -> dict:
    t0 = time.perf_counter()
    memory = _get_memory(session_id)

    # Record user turn
    turn_text = query + (f" [📎 {len(images)} image(s)]" if images else "")
    memory.add_user_turn(turn_text)

    # Route
    context = memory.build_context(current_query=query)
    routing = route(CFG, query, student_id, from_date, to_date,
                    conversation_context=context)
    routes = routing.get("routes")
    if routes is None:
        routes = ["web"]

    # Query expansion (auto)
    expanded_ctx = ""
    if "web" in routes:
        expanded_ctx = _expand_and_search(query)

    # Build inputs
    inputs = {
        "user_query": query,
        "student_id": student_id,
        "from_date": from_date,
        "to_date": to_date,
        "policy_domain": routing.get("policy_domain", "other"),
    }
    if expanded_ctx:
        inputs["user_query"] = f"{query}\n\n[Pre-searched results]\n{expanded_ctx}"

    # Run crew
    answer = run_crew_with_memory(
        CFG, routes, inputs, memory,
        stream_callback=stream_callback,
        images=images or None,
    )

    _BANK.flush()
    elapsed = time.perf_counter() - t0
    return {
        "answer": answer,
        "session_id": session_id,
        "routes": routes,
        "elapsed_seconds": round(elapsed, 1),
        "turn_count": memory.turn_count,
    }


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(
    query: str = Form(...),
    session_id: str = Form(default=None),
    student_id: str = Form(default=None),
    from_date: str = Form(default=None),
    to_date: str = Form(default=None),
    images: list[UploadFile] = File(default=[]),
):
    """Multimodal chat. Accepts text + images via multipart form."""
    sid = session_id or str(uuid.uuid4())[:8]
    encoded = await _encode_uploads(images) if images else None
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _execute_turn(sid, query, encoded, student_id, from_date, to_date),
    )


@app.post("/chat/stream")
async def chat_stream(
    query: str = Form(...),
    session_id: str = Form(default=None),
    student_id: str = Form(default=None),
    from_date: str = Form(default=None),
    to_date: str = Form(default=None),
    images: list[UploadFile] = File(default=[]),
):
    """SSE streaming chat. Events: chunk, done, error."""
    sid = session_id or str(uuid.uuid4())[:8]
    encoded = await _encode_uploads(images) if images else None

    async def event_gen():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _cb(chunk):
            content = getattr(chunk, "content", None) or ""
            agent = getattr(chunk, "agent_role", None) or ""
            if content:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("chunk", {"content": content, "agent": agent}),
                )

        def _blocking():
            try:
                result = _execute_turn(
                    sid, query, encoded, student_id, from_date, to_date,
                    stream_callback=_cb,
                )
                loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", {"error": str(exc)}))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _blocking)

        while True:
            item = await queue.get()
            if item is None:
                break
            evt, data = item
            yield {"event": evt, "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(event_gen())


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    student_id: str | None = None
    from_date: str | None = None
    to_date: str | None = None
    images: list[dict] | None = None  # [{"b64": "...", "mime": "image/jpeg"}]


@app.post("/chat/json")
async def chat_json(req: ChatRequest):
    """JSON-only chat. Images as base64 in body (for mobile/programmatic clients)."""
    sid = req.session_id or str(uuid.uuid4())[:8]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _execute_turn(
            sid, req.query, req.images, req.student_id, req.from_date, req.to_date,
        ),
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    return _get_memory(session_id).get_stats()


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    memory = _get_memory(session_id)
    memory.clear()
    _BANK.flush()
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok", "model": CFG.llm.model}
