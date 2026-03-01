from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .config import load_config
from .crew_runner import bootstrap, route, run_crew

app = FastAPI(title="Simple Multi-Agent (CrewAI)")

CFG = load_config(Path(__file__).resolve().parent / "config")
bootstrap(CFG)


class RunRequest(BaseModel):
    query: str
    student_id: str | None = None
    from_date: str | None = None
    to_date: str | None = None


def _route_and_inputs(req: RunRequest) -> tuple[list[str], dict]:
    routing = route(CFG, req.query, req.student_id, req.from_date, req.to_date)
    routes = routing.get("routes") or ["web"]
    inputs = {
        "user_query": req.query,
        "student_id": req.student_id,
        "from_date": req.from_date,
        "to_date": req.to_date,
        "policy_domain": routing.get("policy_domain", "other"),
    }
    return routes, inputs


# ── sync endpoint (runs blocking crew in threadpool) ───────────────────

@app.post("/run")
async def run_endpoint(req: RunRequest):
    routes, inputs = _route_and_inputs(req)

    # Run blocking kickoff in a thread so we don't block the event loop
    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(
        None, lambda: run_crew(CFG, routes, inputs, stream=False)
    )
    return {"routes": routes, "result": out.raw}


# ── SSE streaming endpoint ────────────────────────────────────────────

@app.post("/run/stream")
async def run_stream(req: RunRequest):
    routes, inputs = _route_and_inputs(req)

    async def event_gen():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _blocking_stream():
            """Run in a thread: iterate sync streaming, push to async queue."""
            try:
                streaming = run_crew(CFG, routes, inputs, stream=True)
                for chunk in streaming:
                    # Build event data defensively
                    data = {
                        "content": getattr(chunk, "content", str(chunk)),
                        "task": getattr(chunk, "task_name", None),
                        "agent": getattr(chunk, "agent_role", None),
                        "type": getattr(chunk, "chunk_type", None),
                    }
                    tool_call = getattr(chunk, "tool_call", None)
                    if tool_call is not None:
                        data["tool_call"] = (
                            tool_call.model_dump()
                            if hasattr(tool_call, "model_dump")
                            else str(tool_call)
                        )
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", data))

                # Final result
                final_raw = streaming.result.raw if hasattr(streaming, "result") else ""
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("final", {"result": final_raw})
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", {"error": str(exc)})
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        # Start blocking iteration in a background thread
        loop.run_in_executor(None, _blocking_stream)

        # Yield SSE events from the async queue
        while True:
            item = await queue.get()
            if item is None:
                break
            event_type, data = item
            yield {
                "event": event_type,
                "data": json.dumps(data, ensure_ascii=False),
            }

    return EventSourceResponse(event_gen())
