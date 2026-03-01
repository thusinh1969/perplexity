from __future__ import annotations
import json
import logging
import time
from typing import Any

from tavily import TavilyClient
from crewai.tools import tool
from ..tool_context import get_tool_config

log = logging.getLogger("school_agents.tools.web")

# Max chars returned to LLM (rough: 1 token ~ 4 chars → 8000 chars ~ 2000 tokens)
MAX_TOOL_RESULT_CHARS = 8192


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate tool result to avoid blowing up LLM context."""
    if len(text) <= limit:
        return text
    log.warning(
        "[truncate] Result %d chars → truncated to %d chars (~%d tokens saved)",
        len(text), limit, (len(text) - limit) // 4,
    )
    return text[:limit] + f"\n... [TRUNCATED — {len(text)} total chars]"


def _get_client() -> TavilyClient:
    """Create a TavilyClient from the YAML tool config."""
    cfg = get_tool_config()["web"]["tavily"]
    log.debug("[tavily] Creating client (api_key=%s...)", cfg["api_key"][:12])
    return TavilyClient(api_key=cfg["api_key"])


# ── internal helpers (plain functions, NOT @tool) ──────────────────────

def _search_deep(
    query: str,
    max_results: int = 3,
    search_depth: str = "basic",
    include_raw_content: bool = False,
) -> dict:
    """Tavily SDK search -> returns dict."""
    # Sanitize include_raw_content — LLM may pass string "false"/"true"
    if isinstance(include_raw_content, str):
        include_raw_content = include_raw_content.lower() in ("true", "1", "yes")

    log.info(
        "[tavily:search] query=%r depth=%s max=%d raw_content=%s",
        query, search_depth, max_results, include_raw_content,
    )
    client = _get_client()
    t0 = time.perf_counter()
    try:
        result = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )
        elapsed = time.perf_counter() - t0
        result_json = json.dumps(result, ensure_ascii=False, default=str)
        n_results = len(result.get("results", []))
        log.info(
            "[tavily:search] OK in %.2fs | %d results | %d chars JSON",
            elapsed, n_results, len(result_json),
        )
        # Log rough token count (1 token ~ 4 chars)
        approx_tokens = len(result_json) // 4
        log.warning(
            "[tavily:search] ⚠️  Response ~%d tokens (JSON %d chars). "
            "If > model context window → LLM returns None!",
            approx_tokens, len(result_json),
        )
        for i, r in enumerate(result.get("results", [])[:3]):
            log.debug(
                "[tavily:search] result[%d] url=%s title=%s content_len=%d",
                i, r.get("url", "?"), r.get("title", "?")[:60],
                len(r.get("content", "")),
            )
        return result
    except Exception as exc:
        log.error("[tavily:search] FAILED after %.2fs: %s", time.perf_counter() - t0, exc)
        raise


def _crawl_url(
    url: str,
    instructions: str | None = None,
    max_depth: int = 1,
    limit: int = 5,
    allow_external: bool = False,
    select_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> dict:
    """Tavily SDK crawl -> returns dict."""
    log.info("[tavily:crawl] url=%s depth=%d limit=%d", url, max_depth, limit)
    client = _get_client()
    kwargs: dict[str, Any] = {
        "url": url,
        "max_depth": max_depth,
        "limit": limit,
        "allow_external": allow_external,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if select_paths:
        kwargs["select_paths"] = select_paths
    if exclude_paths:
        kwargs["exclude_paths"] = exclude_paths

    t0 = time.perf_counter()
    try:
        result = client.crawl(**kwargs)
        elapsed = time.perf_counter() - t0
        result_json = json.dumps(result, ensure_ascii=False, default=str)
        n_pages = len(result.get("results", []))
        log.info(
            "[tavily:crawl] OK in %.2fs | %d pages | %d chars JSON",
            elapsed, n_pages, len(result_json),
        )
        return result
    except Exception as exc:
        log.error("[tavily:crawl] FAILED after %.2fs: %s", time.perf_counter() - t0, exc)
        raise


# ── CrewAI tools (thin wrappers returning JSON str) ────────────────────

@tool("web_search_deep")
def web_search_deep(
    query: str,
    max_results: int = 3,
    search_depth: str = "basic",
) -> str:
    """Tavily Search via official SDK. Returns JSON string with web search results."""
    log.info("[tool:web_search_deep] CALLED query=%r", query)
    result = _search_deep(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_raw_content=False,
    )
    out = json.dumps(result, ensure_ascii=False, default=str)
    log.info("[tool:web_search_deep] RETURNING %d chars to LLM", len(out))
    return _truncate(out)


@tool("web_crawl_url")
def web_crawl_url(
    url: str,
    instructions: str = "",
    max_depth: int = 1,
    limit: int = 5,
    allow_external: bool = False,
) -> str:
    """Tavily Crawl via official SDK. Starts from one URL and collects multiple pages. Returns JSON string."""
    log.info("[tool:web_crawl_url] CALLED url=%s", url)
    result = _crawl_url(
        url=url,
        instructions=instructions or None,
        max_depth=max_depth,
        limit=limit,
        allow_external=allow_external,
    )
    out = json.dumps(result, ensure_ascii=False, default=str)
    log.info("[tool:web_crawl_url] RETURNING %d chars to LLM", len(out))
    return _truncate(out)


@tool("web_search_then_crawl")
def web_search_then_crawl(query: str, seed_top_k: int = 1) -> str:
    """Convenience: search -> pick top K urls -> crawl each -> return aggregated JSON."""
    log.info("[tool:web_search_then_crawl] CALLED query=%r top_k=%d", query, seed_top_k)
    search_data = _search_deep(query=query)
    seeds = [
        r["url"]
        for r in (search_data.get("results") or [])[:seed_top_k]
        if r.get("url")
    ]
    log.info("[tool:web_search_then_crawl] seeds=%s", seeds)
    crawls = []
    for u in seeds:
        crawls.append(
            _crawl_url(
                url=u,
                instructions=f"Find pages most relevant to: {query}",
                max_depth=1,
                limit=5,
                allow_external=False,
            )
        )
    out = json.dumps(
        {"query": query, "seeds": seeds, "search": search_data, "crawls": crawls},
        ensure_ascii=False,
        default=str,
    )
    log.info("[tool:web_search_then_crawl] RETURNING %d chars to LLM", len(out))
    return _truncate(out)


@tool("web_search_expanded")
def web_search_expanded(
    query: str,
    max_results_per_query: int = 3,
    search_depth: str = "basic",
) -> str:
    """Multi-query expanded search: generates 3 related queries from user question, searches all, merges results via rank fusion. Use for complex or vague questions. Returns JSON with merged results."""
    log.info("[tool:web_search_expanded] CALLED query=%r", query)

    from ..query_expander import QueryExpander
    from ..crew_runner import make_openai_client, get_llm_extra_body
    from ..config import load_config
    from pathlib import Path

    # Get LLM config for expander
    try:
        cfg_dir = Path(__file__).resolve().parent.parent / "config"
        cfg = load_config(cfg_dir)
        oai = make_openai_client(cfg)
        extra = get_llm_extra_body(cfg)
        model = cfg.llm.model
        if model.startswith("openai/"):
            model = model[len("openai/"):]
    except Exception as exc:
        log.warning("[tool:web_search_expanded] Config load failed: %s — doing single search", exc)
        return web_search_deep.run(query=query, max_results=max_results_per_query * 2, search_depth=search_depth)

    expander = QueryExpander(openai_client=oai, model=model, extra_body=extra)

    # 1. Expand query
    queries = expander.expand(query)
    log.info("[tool:web_search_expanded] Expanded to %d queries: %s", len(queries), queries)

    # 2. Search all + merge
    merged = expander.search_expanded(
        queries=queries,
        search_fn=_search_deep,
        max_results_per_query=max_results_per_query,
        search_depth=search_depth,
        include_raw_content=False,
    )

    out = json.dumps(merged, ensure_ascii=False, default=str)
    log.info("[tool:web_search_expanded] RETURNING %d merged results, %d chars",
             len(merged.get("results", [])), len(out))
    return _truncate(out)


# ── Standalone helpers (used by run_chat.py interactive mode) ──────────

def expand_and_search(
    user_query: str,
    openai_client: Any,
    model: str,
    max_results_per_query: int = 3,
    search_depth: str = "basic",
    selected_queries: list[str] | None = None,
    extra_body: dict | None = None,
) -> dict:
    """
    Expand query + search, for use outside CrewAI tool context.

    Args:
        user_query: Original question
        openai_client: OpenAI-compatible client
        model: Model name
        max_results_per_query: Results per query
        search_depth: "basic" or "advanced"
        selected_queries: If provided, skip expansion and use these queries directly
        extra_body: Extra params for LLM calls (top_k, repetition_penalty)

    Returns:
        Dict with queries, merged results, stats
    """
    from ..query_expander import QueryExpander

    expander = QueryExpander(openai_client=openai_client, model=model, extra_body=extra_body)

    if selected_queries:
        queries = selected_queries
    else:
        queries = expander.expand(user_query)

    merged = expander.search_expanded(
        queries=queries,
        search_fn=_search_deep,
        max_results_per_query=max_results_per_query,
        search_depth=search_depth,
        include_raw_content=False,
    )

    return merged


def expand_queries_only(
    user_query: str,
    openai_client: Any,
    model: str,
    extra_body: dict | None = None,
    progress_callback=None,
    max_tokens: int = 16384,
) -> list[str]:
    """Just expand, don't search. For interactive confirmation flow."""
    from ..query_expander import QueryExpander
    expander = QueryExpander(openai_client=openai_client, model=model,
                             extra_body=extra_body, max_tokens=max_tokens)
    return expander.expand(user_query, progress_callback=progress_callback)
