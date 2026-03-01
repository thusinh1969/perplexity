from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import json_repair

from crewai import Agent, Crew, Task, LLM

from .config import AppConfig
from .tool_context import set_tool_config
from .llm_utils import strip_think_tags
from .tools.web_tools import web_search_deep, web_crawl_url, web_search_then_crawl, web_search_expanded
from .tools.student_tools import student_get_profile, student_get_grades, student_get_attendance
from .tools.rag_tools import rag_query_policy
from .tools.datetime_tools import get_datetime, date_add_delta_days

log = logging.getLogger("school_agents.crew_runner")


# ── helpers ────────────────────────────────────────────────────────────

def _make_llm(cfg: AppConfig) -> LLM:
    log.info(
        "[LLM] Creating: model=%s base_url=%s temp=%.1f top_p=%.2f top_k=%d rep_pen=%.2f max_tokens=%d",
        cfg.llm.model, cfg.llm.base_url, cfg.llm.temperature,
        cfg.llm.top_p, cfg.llm.top_k, cfg.llm.repetition_penalty,
        cfg.llm.max_tokens,
    )
    llm = LLM(
        model=cfg.llm.model,
        provider="openai",
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        max_tokens=cfg.llm.max_tokens,
        # top_k and repetition_penalty are not standard OpenAI params
        # LiteLLM passes them via extra_body to LMStudio/vLLM/etc.
        extra_body={
            "top_k": cfg.llm.top_k,
            "repetition_penalty": cfg.llm.repetition_penalty,
        },
    )
    log.info("[LLM] Created OK: %s", type(llm).__name__)
    return llm


def _make_agents(cfg: AppConfig, llm: LLM) -> Dict[str, Agent]:
    a = cfg.agents

    # ── Inject current datetime into ALL agent backstories ──
    # This ensures every agent knows today's date without needing a tool call.
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    _now = _dt.now(_ZI("Asia/Ho_Chi_Minh"))
    _weekdays_vi = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]
    _weekday = _weekdays_vi[_now.weekday()]
    _datetime_line = (
        f"[SYSTEM TIME] Hôm nay là {_weekday}, ngày {_now:%d/%m/%Y}, "
        f"giờ {_now:%H:%M:%S} (múi giờ Asia/Ho_Chi_Minh, UTC+7). "
        f"English: {_now:%A, %B %d, %Y at %H:%M:%S %Z}."
    )

    def _inject_datetime(agent_cfg: dict) -> dict:
        """Prepend datetime to backstory."""
        cfg_copy = dict(agent_cfg)
        original = cfg_copy.get("backstory", "")
        cfg_copy["backstory"] = f"{_datetime_line}\n\n{original}"
        return cfg_copy

    router = Agent(llm=llm, verbose=False, tools=[get_datetime, date_add_delta_days],
                   **_inject_datetime(a["router"]))
    web = Agent(
        llm=llm,
        verbose=True,
        tools=[web_search_deep, web_search_expanded, web_crawl_url, web_search_then_crawl],
        max_iter=3,
        **_inject_datetime(a["web_researcher"]),
    )
    student = Agent(
        llm=llm,
        verbose=True,
        tools=[student_get_profile, student_get_grades, student_get_attendance],
        max_iter=3,
        **_inject_datetime(a["student_data"]),
    )
    rag = Agent(
        llm=llm,
        verbose=True,
        tools=[rag_query_policy],
        max_iter=3,
        **_inject_datetime(a["policy_rag"]),
    )
    # Composer: NO tools, max_iter=1 (one-shot, no retry)
    composer = Agent(llm=llm, verbose=True, max_iter=1,
                     **_inject_datetime(a["composer"]))

    return {
        "router": router,
        "web_researcher": web,
        "student_data": student,
        "policy_rag": rag,
        "composer": composer,
    }


def _make_task(cfg: AppConfig, name: str, agents: Dict[str, Agent]) -> Task:
    t = cfg.tasks[name]
    agent_key = t["agent"]
    return Task(
        description=t["description"],
        expected_output=t["expected_output"],
        agent=agents[agent_key],
    )


# ── public API ─────────────────────────────────────────────────────────

def route(
    cfg: AppConfig,
    user_query: str,
    student_id: str | None,
    from_date: str | None,
    to_date: str | None,
    conversation_context: str = "",
) -> Dict[str, Any]:
    """Use the router agent to decide which sub-crews to invoke."""
    log.info("[route] query=%r student_id=%s context=%d chars",
             user_query, student_id, len(conversation_context))
    llm = _make_llm(cfg)
    agents = _make_agents(cfg, llm)
    route_task = _make_task(cfg, "route_task", agents)

    # Inject current datetime into context so router can understand time references
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    _now = _dt.now(_ZI("Asia/Ho_Chi_Minh"))
    _weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    _datetime_str = f"{_weekdays[_now.weekday()]}, {_now:%d/%m/%Y %H:%M:%S} (Asia/Ho_Chi_Minh)"

    ctx_with_dt = f"Current date/time: {_datetime_str}\n\n{conversation_context}" if conversation_context else f"Current date/time: {_datetime_str}"

    crew = Crew(agents=[agents["router"]], tasks=[route_task])
    log.info("[route] Kicking off router crew...")
    out = crew.kickoff(
        inputs={
            "user_query": user_query,
            "student_id": student_id,
            "from_date": from_date,
            "to_date": to_date,
            "conversation_context": ctx_with_dt,
        }
    )
    raw = out.raw if hasattr(out, "raw") else str(out)
    log.info("[route] Router raw output (%d chars): %s", len(raw), raw[:500])

    # Extract JSON from potentially messy output (thinking text, tags, etc.)
    from .llm_utils import extract_json
    cleaned = extract_json(raw)
    if not cleaned:
        cleaned = strip_think_tags(raw)

    try:
        parsed = json_repair.loads(cleaned)
        routes = parsed.get("routes", [])

        # Safety: if routes=[] but query is substantive, force web
        _trivial = {"hello","hi","thanks","thank","bye","quit","stats","facts","clear"}
        query_lower = user_query.strip().lower()
        if not routes and query_lower not in _trivial and len(query_lower) > 5:
            log.warning("[route] Router returned empty routes for substantive query — forcing [web]")
            parsed["routes"] = ["web"]

        log.info("[route] Parsed routes=%s policy_domain=%s", parsed.get("routes"), parsed.get("policy_domain"))
        return parsed
    except Exception as exc:
        log.warning("[route] Failed to parse JSON: %s — falling back to [web]", exc)
        return {
            "routes": ["web"],
            "student_id": student_id,
            "from_date": from_date,
            "to_date": to_date,
            "policy_domain": "other",
            "router_raw": raw,
        }


def _build_crew(
    cfg: AppConfig,
    routes: List[str],
    stream: bool = False,
) -> Crew:
    """Build a Crew with the right agents/tasks but do NOT kickoff yet."""
    log.info("[build_crew] routes=%s stream=%s", routes, stream)
    llm = _make_llm(cfg)
    agents = _make_agents(cfg, llm)

    # When streaming, disable verbose on ALL agents to prevent double output
    # (verbose prints to stdout AND streaming chunks go to our callback)
    if stream:
        for agent in agents.values():
            agent.verbose = False

    tasks: List[Task] = []
    if "web" in routes:
        tasks.append(_make_task(cfg, "web_task", agents))
    if "student" in routes:
        tasks.append(_make_task(cfg, "student_task", agents))
    if "policy_rag" in routes:
        tasks.append(_make_task(cfg, "policy_task", agents))

    # Always compose last
    tasks.append(_make_task(cfg, "compose_task", agents))
    log.info("[build_crew] %d tasks: %s", len(tasks), [t.description[:50] + "..." for t in tasks])

    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        stream=stream,
    )


def run_crew(
    cfg: AppConfig,
    routes: List[str],
    inputs: Dict[str, Any],
    stream: bool = False,
):
    """Synchronous kickoff. Returns CrewOutput (or streaming iterator if stream=True)."""
    log.info("[run_crew] Starting kickoff routes=%s stream=%s", routes, stream)
    crew = _build_crew(cfg, routes, stream=stream)
    log.info("[run_crew] Crew built, calling kickoff(inputs=%s)", list(inputs.keys()))
    return crew.kickoff(inputs=inputs)


async def run_crew_async(
    cfg: AppConfig,
    routes: List[str],
    inputs: Dict[str, Any],
    stream: bool = False,
):
    """Async kickoff using CrewAI's native akickoff().

    - stream=False → returns CrewOutput
    - stream=True  → returns async-iterable StreamingOutput
                      (async for chunk in output: ...)
                      then access output.result for final CrewOutput
    """
    crew = _build_crew(cfg, routes, stream=stream)
    return await crew.akickoff(inputs=inputs)


def bootstrap(cfg: AppConfig) -> None:
    """Make tool config available to tools at startup."""
    set_tool_config(cfg.tools)


# ── Multi-turn memory support ─────────────────────────────────────────

def run_crew_with_memory(
    cfg: AppConfig,
    routes: List[str],
    inputs: Dict[str, Any],
    memory: Any,  # ConversationMemory (import at runtime to avoid circular)
    stream_callback: Any = None,  # callable(chunk) — called for each streaming chunk
    status_callback: Any = None,  # callable(msg) — called for status updates (e.g. "Extracting facts...")
) -> str:
    """
    Run crew with multi-turn conversation context.

    Flow:
      1. memory.build_context() → compress if needed, get context string
      2. Inject context into inputs as {conversation_context}
      3. crew.kickoff() — streaming if stream_callback provided
      4. Save assistant turn to memory
      5. Return raw answer

    Args:
        stream_callback: If provided, enables CrewAI streaming. Called with each
            chunk object (has .content, .task_name, .agent_role, .chunk_type attrs).
        status_callback: If provided, called with status strings during post-processing.
    """
    def _status(msg):
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass

    query = inputs.get("user_query", "")

    # 1. Build context (triggers compression automatically)
    context = memory.build_context(current_query=query)
    log.info("[run_crew_with_memory] Context: %d chars for session %s",
             len(context), memory.session_id)

    # 2. Inject into inputs + current datetime (so agents don't need to call datetime tool)
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZI
    _now = _dt.now(_ZI("Asia/Ho_Chi_Minh"))
    _weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    _datetime_str = f"{_weekdays[_now.weekday()]}, {_now:%d/%m/%Y %H:%M:%S} (Asia/Ho_Chi_Minh)"

    enriched = {
        **inputs,
        "conversation_context": f"Current date/time: {_datetime_str}\n\n{context}" if context else f"Current date/time: {_datetime_str}",
        "current_datetime": _datetime_str,
    }

    # 3. Run crew (streaming or blocking)
    use_stream = stream_callback is not None
    crew = _build_crew(cfg, routes, stream=use_stream)

    if use_stream:
        streaming = crew.kickoff(inputs=enriched)
        composer_text = []  # ONLY composer output → becomes raw_answer
        all_text = []       # everything (debug fallback)
        _task_switches = 0
        _current_task = ""
        _is_last_task = False
        # Count tasks to know when we're on the last one (composer)
        _total_tasks = len([r for r in routes if r in ("web","student","policy_rag")]) + 1  # +1 for composer

        try:
            _first_chunk = True
            for chunk in streaming:
                # Debug: log first chunk's attributes to help diagnose filtering
                if _first_chunk:
                    _first_chunk = False
                    attrs = {k: getattr(chunk, k, '?') for k in
                             ['content','agent_role','agent','role','task_name','chunk_type','type']}
                    log.debug("[stream] First chunk attrs: %s", attrs)
                content = getattr(chunk, "content", None) or ""
                agent = (
                    getattr(chunk, "agent_role", None)
                    or getattr(chunk, "agent", None)
                    or getattr(chunk, "role", None)
                    or ""
                )
                task = getattr(chunk, "task_name", None) or ""

                # Track task switches
                task_key = f"{agent}|{task}"
                if task_key and task_key != _current_task and (agent or task):
                    _current_task = task_key
                    _task_switches += 1
                    if _task_switches >= _total_tasks:
                        _is_last_task = True

                if content:
                    all_text.append(content)
                    # Capture composer output: by name or by position (last task)
                    is_composer = (
                        ("composer" in str(agent).lower()) or
                        _is_last_task
                    )
                    if is_composer:
                        composer_text.append(content)

                # Forward to caller's callback
                try:
                    stream_callback(chunk)
                except Exception as cb_exc:
                    log.debug("[run_crew_with_memory] Callback error (non-fatal): %s", cb_exc)
        except Exception as exc:
            log.error("[run_crew_with_memory] Streaming generator error: %s", exc, exc_info=True)

        # Capture raw answer: prefer CrewAI result > composer chunks > all chunks
        raw_answer = ""
        try:
            if streaming.is_completed and streaming._result is not None:
                raw_answer = streaming.result.raw
                log.debug("[run_crew_with_memory] Got answer from streaming.result.raw (%d chars)", len(raw_answer))
        except Exception:
            pass
        if not raw_answer:
            try:
                raw_answer = streaming.get_full_text()
                log.debug("[run_crew_with_memory] Got answer from get_full_text() (%d chars)", len(raw_answer))
            except Exception:
                pass
        if not raw_answer and composer_text:
            raw_answer = "".join(composer_text)
            log.debug("[run_crew_with_memory] Got answer from composer_text buffer (%d chars)", len(raw_answer))
        if not raw_answer and all_text:
            raw_answer = "".join(all_text)
            log.warning("[run_crew_with_memory] Fell back to all_text buffer (%d chars) — may include intermediate agents", len(raw_answer))
        if not raw_answer:
            log.warning("[run_crew_with_memory] ⚠️ Empty answer after streaming!")

    else:
        result = crew.kickoff(inputs=enriched)
        raw_answer = result.raw if hasattr(result, "raw") else str(result)

    # Strip think tags ONLY for memory storage — display gets raw
    clean_answer = strip_think_tags(raw_answer)
    # Strip "Final Answer:" prefix (we tell LLM to include it for CrewAI, but memory doesn't need it)
    if clean_answer.lstrip().startswith("Final Answer:"):
        clean_answer = clean_answer.lstrip().removeprefix("Final Answer:").lstrip()

    log.info("[run_crew_with_memory] raw_answer=%d chars, clean_answer=%d chars",
             len(raw_answer), len(clean_answer))

    # 4. Save CLEAN answer to memory (user turn already added by caller)
    memory.add_assistant_turn(clean_answer, routes=routes)
    _status("💾 Saving to memory...")

    # 5. Extract facts from this turn pair (uses clean turns from memory)
    if memory.facts:
        _status("🧠 Extracting facts...")
        try:
            from openai import OpenAI
            oai = OpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key or "no-key")
            model = cfg.llm.model
            if model.startswith("openai/"):
                model = model[len("openai/"):]
            extra = get_llm_extra_body(cfg)
            fact_counts = memory.extract_facts(llm_client=oai, model=model, extra_body=extra,
                                                max_tokens=cfg.llm.structured_max_tokens)
            log.info("[run_crew_with_memory] Facts extracted: %s", fact_counts)
            _status(f"🧠 Facts: +{fact_counts.get('entities',0)}E +{fact_counts.get('relations',0)}R +{fact_counts.get('facts',0)}F")
        except Exception as exc:
            log.error("[run_crew_with_memory] ⚠️ Fact extraction FAILED: %s", exc, exc_info=True)
            _status("⚠️ Fact extraction failed")

    # 6. Persist
    memory.save()
    _status("✅ Done")

    # 6. Persist
    memory.save()

    log.info("[run_crew_with_memory] Done. Answer %d chars (clean %d). Stats: %s",
             len(raw_answer), len(clean_answer), memory.get_stats())
    return raw_answer  # caller gets raw for display; memory already has clean version


def make_openai_client(cfg: AppConfig):
    """Create an OpenAI-compatible client from LLM config (for compressor)."""
    from openai import OpenAI
    return OpenAI(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key or "no-key",
    )


def get_llm_extra_body(cfg: AppConfig) -> dict:
    """Extra params for direct OpenAI SDK calls (top_k, repetition_penalty).
    
    These are non-standard OpenAI params supported by LMStudio/vLLM/etc.
    Pass as extra_body= in .chat.completions.create().
    """
    return {
        "top_k": cfg.llm.top_k,
        "repetition_penalty": cfg.llm.repetition_penalty,
    }
