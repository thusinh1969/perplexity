from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import json_repair

from crewai import Agent, Crew, Task, LLM

from .config import AppConfig
from .tool_context import set_tool_config
from .llm_utils import strip_think_tags, extract_json
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

        # Safety: only force web if router says [] BUT query has CLEAR web signals
        # Trust the router for everything else — it has the full prompt + conversation context
        query_lower = user_query.strip().lower()

        if not routes and len(query_lower) > 5:
            _web_signals = (
                # Time-sensitive / live data
                "hôm nay", "hiện tại", "bây giờ", "mới nhất", "latest", "today",
                "this week", "tuần này", "tháng này", "this month", "right now",
                "just happened", "vừa xảy ra", "breaking", "tin mới",
                # Explicit search/verify intent
                "search", "tìm kiếm", "google", "verify", "kiểm chứng",
                "nguồn", "source", "link", "citation", "fact check",
                # Price/market/live data
                "giá", "price", "stock", "cổ phiếu", "tỷ giá", "exchange rate",
                "thời tiết", "weather", "score", "kết quả",
                # Current role holders
                "ai đang là", "who is the current", "hiện là",
            )
            has_web_signal = any(s in query_lower for s in _web_signals)
            if has_web_signal:
                log.warning("[route] Router returned [] but query has web signals — forcing [web]")
                parsed["routes"] = ["web"]
            else:
                log.info("[route] Router returned [] — trusting it (no web signals detected)")

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
    # ── Vision: direct VLM call with base64 → inject analysis as text ──
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

    # ── Clear vision context ──
    # (Vision analysis already injected as text before crew.kickoff — no cleanup needed)

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

    # 5. Extract facts from VERIFIED EVIDENCE (not from LLM output)
    #    Only when routes include evidence sources (web, rag, api)
    evidence_routes = [r for r in routes if r in ("web", "policy_rag", "student")]
    if memory.facts and evidence_routes:
        _status("🧠 Extracting facts from evidence...")
        try:
            # Get task outputs from CrewAI result
            crew_result = None
            if use_stream:
                try:
                    if streaming.is_completed and streaming._result is not None:
                        crew_result = streaming.result
                except Exception:
                    pass
            else:
                crew_result = result  # noqa: F821 — assigned in else branch above

            evidence_text, evidence_sources, evidence_source_type = _extract_evidence(crew_result, routes)

            if evidence_text:
                from openai import OpenAI
                oai = OpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key or "no-key")
                model = cfg.llm.model
                if model.startswith("openai/"):
                    model = model[len("openai/"):]
                extra = get_llm_extra_body(cfg)
                fact_counts = memory.extract_facts_from_evidence(
                    llm_client=oai, model=model,
                    evidence_text=evidence_text,
                    source_type=evidence_source_type,
                    sources=evidence_sources,
                    user_question=enriched.get("user_query", ""),
                    extra_body=extra,
                    max_tokens=cfg.llm.structured_max_tokens,
                )
                log.info("[run_crew_with_memory] Evidence facts: %s", fact_counts)
                _status(f"🧠 Facts: +{fact_counts.get('entities',0)}E +{fact_counts.get('relations',0)}R +{fact_counts.get('facts',0)}F")
            else:
                log.info("[run_crew_with_memory] No evidence text found in task outputs")
                _status("🧠 No evidence to extract facts from")
        except Exception as exc:
            log.error("[run_crew_with_memory] ⚠️ Evidence fact extraction FAILED: %s", exc, exc_info=True)
            _status("⚠️ Fact extraction failed")
    elif not evidence_routes:
        log.info("[run_crew_with_memory] routes=%s — no evidence sources, skipping fact extraction", routes)

    # 6. Generate smart follow-up suggestions
    try:
        suggestions = _generate_followups(cfg, clean_answer, enriched.get("user_query", ""), routes)
        if suggestions:
            clean_answer = clean_answer.rstrip() + "\n\n" + suggestions
            _status(suggestions)  # Print suggestions directly (streaming may skip return value)
    except Exception as exc:
        log.debug("[followup] Failed (non-critical): %s", exc)

    # 7. Persist
    memory.save()
    _status("✅ Done")

    log.info("[run_crew_with_memory] Done. Answer %d chars (clean %d). Stats: %s",
             len(raw_answer), len(clean_answer), memory.get_stats())
    return clean_answer


def _generate_followups(cfg: AppConfig, answer: str, user_query: str, routes: list[str]) -> str:
    """Generate 2-3 smart follow-up suggestions based on the answer.

    Returns formatted string like:
        💡 Bạn có muốn tôi:
        1. Tìm thêm về tác dụng phụ của Prospan?
        2. So sánh giá Prospan tại các nhà thuốc online?
        3. Phân tích thành phần với các siro ho thay thế?
    """
    from openai import OpenAI
    from .llm_utils import strip_think_tags, extract_json, NO_THINK_SYSTEM, no_think_extra_body

    client = OpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key or "no-key")
    model = cfg.llm.model
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    extra = get_llm_extra_body(cfg)

    prompt = f"""Based on this Q&A, suggest 2-3 natural follow-up questions the user might want to explore next.

USER QUESTION: {user_query[:500]}

ANSWER (summary): {answer[:1000]}

RULES:
- Write in the SAME language as the user's question (Vietnamese if they wrote Vietnamese).
- Each suggestion should explore a DIFFERENT angle: deeper analysis, comparison, practical action, related topic.
- Keep each suggestion short (under 15 words).
- Make them specific to the content, NOT generic.
- Return ONLY a JSON array of 2-3 strings. No explanation.
- Start with [ and end with ]

JSON array:"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": NO_THINK_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=cfg.llm.structured_max_tokens,
        temperature=0.5,
        extra_body=no_think_extra_body(extra),
    )

    raw = strip_think_tags(resp.choices[0].message.content or "").strip()
    cleaned = extract_json(raw)
    if not cleaned:
        return ""

    suggestions = json_repair.loads(cleaned)
    if not isinstance(suggestions, list) or not suggestions:
        return ""

    lines = ["💡 Bạn có muốn tôi:"]
    for i, s in enumerate(suggestions[:3], 1):
        lines.append(f"   {i}. {str(s).strip()}")
    return "\n".join(lines)


def _normalize_image_to_jpeg(b64_data: str, mime: str) -> tuple[str, str]:
    """Convert any image to JPEG base64. vLLM/some backends reject webp."""
    if mime in ("image/jpeg", "image/jpg", "image/png"):
        return b64_data, mime
    import base64, io
    from PIL import Image
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _analyze_images_direct(cfg: AppConfig, images: list[dict], user_query: str) -> str:
    """Call VLM directly with base64 images — bypasses CrewAI entirely.

    This is the ONLY reliable way to get vision working with CrewAI,
    because CrewAI constructs its own messages and doesn't support
    multimodal content natively.

    The VLM receives the raw base64 image(s) + user question,
    returns a detailed analysis that gets injected as text context
    into the crew pipeline.
    """
    from openai import OpenAI
    from .llm_utils import strip_think_tags

    client = OpenAI(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key or "no-key",
    )

    # Build multimodal message: text + images
    content: list[dict] = [
        {"type": "text", "text": (
            f"Analyze the image(s) below in detail, in the context of this question:\n"
            f"{user_query}\n\n"
            f"Provide:\n"
            f"1. What the image shows (chart type, data, labels, axes, legends)\n"
            f"2. Key observations (trends, patterns, anomalies, notable data points)\n"
            f"3. Numbers/dates/values visible in the image\n"
            f"Be specific and factual. Describe what you SEE, don't speculate."
        )},
    ]
    for img in images:
        # Normalize to JPEG — some backends (vLLM) don't support webp
        b64_data, mime = _normalize_image_to_jpeg(img["b64"], img["mime"])
        print(f"[DEBUG] Image: original={img['mime']}, sending={mime}, b64={len(b64_data)} chars", flush=True)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64_data}"},
        })

    model = cfg.llm.model
    if model.startswith("openai/"):
        model = model[len("openai/"):]

    extra = get_llm_extra_body(cfg)

    log.info("[vision] Direct VLM call: model=%s, %d image(s), %d chars query",
             model, len(images), len(user_query))

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg.llm.max_tokens,
        temperature=0.3,
        extra_body=extra,
    )

    raw = resp.choices[0].message.content or ""
    analysis = strip_think_tags(raw).strip()
    log.info("[vision] VLM analysis: %d chars", len(analysis))
    return analysis


def _extract_evidence(crew_result, routes: list[str]) -> tuple[str, list[dict], str]:
    """Extract raw evidence text + sources from CrewAI task outputs.

    Returns:
        (evidence_text, sources_list, source_type)
        evidence_text: Raw evidence from web/rag/api tasks (NOT composer)
        sources_list: List of {"id": "S1", "url": "...", "title": "..."} if available
        source_type: "web" | "rag" | "api"
    """
    if crew_result is None or not hasattr(crew_result, "tasks_output"):
        return "", [], "web"

    # Map agent names to source types
    agent_source_map = {
        "web researcher": "web",
        "web_researcher": "web",
        "student data fetcher": "api",
        "student_data": "api",
        "policy/rag specialist": "rag",
        "policy_rag": "rag",
    }

    evidence_parts = []
    all_sources = []
    source_type = "web"  # default

    for task_output in crew_result.tasks_output:
        agent_name = (task_output.agent or "").lower().strip()

        # Skip composer — that's LLM synthesis, not evidence
        if "composer" in agent_name or "answer" in agent_name:
            continue

        # Match to source type
        matched_type = None
        for key, stype in agent_source_map.items():
            if key in agent_name:
                matched_type = stype
                break

        if matched_type is None:
            # Also check by route: if only "web" in routes and not composer, it's likely web
            if "router" in agent_name:
                continue  # skip router output
            continue

        raw = task_output.raw or ""
        if not raw.strip():
            continue

        source_type = matched_type
        evidence_parts.append(raw)

        # Try to parse sources from JSON evidence (web_result format)
        try:
            # Strip "Final Answer:" prefix if present
            clean = raw.strip()
            if clean.startswith("Final Answer:"):
                clean = clean[len("Final Answer:"):].strip()

            parsed = json_repair.loads(extract_json(clean) or clean)
            if isinstance(parsed, dict) and "sources" in parsed:
                for s in parsed["sources"]:
                    if isinstance(s, dict):
                        all_sources.append({
                            "id": s.get("id", ""),
                            "url": s.get("url", ""),
                            "title": s.get("title", ""),
                        })
        except Exception:
            pass  # Not JSON — still use raw text as evidence

    evidence_text = "\n\n".join(evidence_parts)
    log.info("[_extract_evidence] Found %d evidence parts (%d chars), %d sources, type=%s",
             len(evidence_parts), len(evidence_text), len(all_sources), source_type)
    return evidence_text, all_sources, source_type


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
