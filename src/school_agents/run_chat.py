"""
run_chat.py — Interactive multi-turn chat with conversation memory.

Usage:
    # New session with auto-generated ID
    python -m school_agents.run_chat --query "Python 3.13 có gì mới?"

    # Continue existing session
    python -m school_agents.run_chat --session 123456789 --query "So sánh với 3.12?"

    # Interactive REPL mode
    python -m school_agents.run_chat --session 123456789 --interactive

    # With debug logging
    python -m school_agents.run_chat --session 123456789 --interactive --debug

    # With LLMLingua compression (if installed)
    python -m school_agents.run_chat --session 123456789 --interactive --compressor llmlingua

    # Show session stats
    python -m school_agents.run_chat --session 123456789 --stats

    # List all sessions
    python -m school_agents.run_chat --list-sessions
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

from .config import load_config, AppConfig
from .crew_runner import bootstrap, route, run_crew_with_memory, make_openai_client, get_llm_extra_body
from .memory_bank import MemoryDB
from .context_compressor import ContextCompressor
from .conversation_memory import ConversationMemory


def _setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    fmt = (
        "%(asctime)s │ %(levelname)-5s │ %(name)-40s │ %(message)s"
        if debug else
        "%(asctime)s │ %(levelname)-5s │ %(message)s"
    )
    logging.basicConfig(
        level=level, format=fmt, datefmt="%H:%M:%S",
        stream=sys.stderr, force=True,
    )
    if not debug:
        for name in ("httpx", "httpcore", "openai", "crewai", "LiteLLM", "litellm"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _create_memory(
    session_id: str,
    cfg,
    overrides: dict | None = None,
) -> tuple["ConversationMemory", "MemoryDB"]:
    """
    Create MemoryBank + ConversationMemory from cfg.memory + CLI overrides.

    Override keys: backend, data_dir, compressor, max_recent_turns, max_context_tokens
    Returns: (memory, bank) tuple
    """
    mc = cfg.memory
    ov = overrides or {}

    # Resolve values: CLI override > YAML config > default
    backend = ov.get("backend", mc.backend)
    data_dir = ov.get("data_dir", mc.data_dir)
    compressor_strategy = ov.get("compressor", mc.compressor)
    max_recent_turns = ov.get("max_recent_turns", mc.max_recent_turns)
    max_context_tokens = ov.get("max_context_tokens", mc.max_context_tokens)

    # Create bank
    if backend == "redis":
        from .memory_bank import RedisMemoryBank
        bank = RedisMemoryBank(
            url=mc.redis_url,
            prefix=mc.redis_prefix,
            ttl=mc.redis_ttl,
        )
    else:
        bank = MemoryDB(data_dir=data_dir)

    # Create compressor
    oai_client = make_openai_client(cfg)
    model = cfg.llm.model
    if model.startswith("openai/"):
        model = model[len("openai/"):]

    compressor = ContextCompressor(
        strategy=compressor_strategy,
        openai_client=oai_client,
        model=model,
        lingua_target_ratio=mc.lingua_target_ratio,
        lingua_device=mc.lingua_device,
        lingua_model=mc.lingua_model,
        extra_body=get_llm_extra_body(cfg),
        max_tokens=cfg.llm.structured_max_tokens,
    )

    memory = ConversationMemory(
        session_id=session_id,
        bank=bank,
        compressor=compressor,
        max_recent_turns=max_recent_turns,
        max_context_tokens=max_context_tokens,
        enable_facts=mc.enable_facts,
    )
    return memory, bank


def _run_one_turn(
    cfg,
    memory: ConversationMemory,
    query: str,
    student_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    interactive: bool = False,
    stream_callback=None,
    status_callback=None,
) -> str:
    """Execute a single conversation turn with memory and optional query expansion."""
    log = logging.getLogger("school_agents.run_chat")

    # 1. Record user turn
    memory.add_user_turn(query)

    # 2. Build context for routing
    context = memory.build_context(current_query=query)

    # 3. Route
    t0 = time.perf_counter()
    routing = route(
        cfg, query, student_id, from_date, to_date,
        conversation_context=context,
    )
    routes = routing.get("routes")
    if routes is None:
        routes = ["web"]  # only fallback when key missing entirely
    policy_domain = routing.get("policy_domain", "other")
    log.info("Routed in %.1fs → %s", time.perf_counter() - t0, routes)

    # 4. Query expansion (only for web routes)
    expanded_context = ""
    mc = cfg.memory
    if "web" in routes and mc.expand_enabled:
        expanded_context = _handle_query_expansion(
            cfg, query, interactive=interactive,
        )

    # 5. Run crew with memory
    inputs = {
        "user_query": query,
        "student_id": student_id,
        "from_date": from_date,
        "to_date": to_date,
        "policy_domain": policy_domain,
    }

    # If we got pre-expanded results, inject them
    if expanded_context:
        inputs["user_query"] = f"{query}\n\n[Pre-searched results]\n{expanded_context}"

    t1 = time.perf_counter()
    answer = run_crew_with_memory(cfg, routes, inputs, memory,
                                  stream_callback=stream_callback,
                                  status_callback=status_callback)
    log.info("Crew done in %.1fs", time.perf_counter() - t1)

    return answer


def _handle_query_expansion(
    cfg,
    user_query: str,
    interactive: bool = False,
) -> str:
    """
    Handle query expansion: expand → optionally confirm → search → return merged results.

    Returns:
        Formatted search results string to inject, or "" if skipped.
    """
    log = logging.getLogger("school_agents.run_chat")
    mc = cfg.memory

    try:
        oai = make_openai_client(cfg)
        extra = get_llm_extra_body(cfg)
        model = cfg.llm.model
        if model.startswith("openai/"):
            model = model[len("openai/"):]

        from .tools.web_tools import expand_queries_only, expand_and_search

        # Step 1: Generate expanded queries (with progress dots during LLM thinking)
        if interactive:
            print("⏳ Generating search queries", end="", flush=True)
            def _progress(n):
                print(".", end="", flush=True)
            expanded = expand_queries_only(user_query, oai, model, extra_body=extra,
                                           progress_callback=_progress,
                                           max_tokens=cfg.llm.structured_max_tokens)
            print(" done!", flush=True)
        else:
            expanded = expand_queries_only(user_query, oai, model, extra_body=extra,
                                           max_tokens=cfg.llm.structured_max_tokens)

        if len(expanded) <= 1:
            log.warning("[expand] Expansion failed — got only original query back")
            if interactive:
                print("⚠️  Query expansion failed (LLM returned no sub-queries). Using original query.")
            return ""

        # Step 2: Confirm or auto
        if interactive and mc.expand_mode == "confirm":
            # Show queries and ask user
            print(f"\n🔍 Query expansion ({len(expanded)} queries):")
            for i, q in enumerate(expanded, 1):
                print(f"   {i}. {q}")

            try:
                choice = input("\n   Search all? [Y/n/edit] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ""

            if choice in ("n", "no", "skip"):
                log.info("[expand] User skipped expansion")
                return ""
            elif choice.startswith("e"):
                # Let user edit — remove queries by number
                try:
                    remove_input = input("   Remove which? (numbers, comma-separated, or Enter to keep all): ").strip()
                    if remove_input:
                        remove_indices = {int(x.strip()) - 1 for x in remove_input.split(",")}
                        expanded = [q for i, q in enumerate(expanded) if i not in remove_indices]
                        print(f"   → Searching {len(expanded)} queries: {expanded}")
                except (ValueError, EOFError):
                    pass  # keep all
            else:
                # Y or Enter — proceed with all
                pass

            if not expanded:
                return ""

            print(f"   Searching {len(expanded)} queries...")

        elif mc.expand_mode == "auto":
            log.info("[expand:auto] Searching %d expanded queries silently", len(expanded))
        else:
            # confirm mode but not interactive — skip expansion
            return ""

        # Step 3: Search all expanded queries
        merged = expand_and_search(
            user_query=user_query,
            openai_client=oai,
            model=model,
            max_results_per_query=mc.expand_max_results_per_query,
            selected_queries=expanded,
            extra_body=extra,
        )

        # Step 4: Format results for injection
        results = merged.get("results", [])
        if not results:
            return ""

        lines = [f"Found {len(results)} results from {len(expanded)} queries:"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "")[:300]
            score = r.get("_rrf_score", 0)
            lines.append(f"\n[{i}] {title} (relevance: {score:.4f})")
            if url:
                lines.append(f"    URL: {url}")
            if content:
                lines.append(f"    {content}")

        result_text = "\n".join(lines)

        if interactive and mc.expand_mode == "confirm":
            print(f"   ✅ {len(results)} results merged.\n")

        return result_text

    except Exception as exc:
        log.warning("[expand] Failed (non-fatal): %s", exc)
        return ""


def _make_stream_callback(num_tasks: int = 2):
    """
    Create a streaming callback that prints ONLY the final task (composer) output.

    CrewAI streams chunks from ALL agents/tasks. We only want the last task
    (composer) visible to the user. Detection strategy:
      1. Check agent_role/agent/role attributes for "composer"
      2. Track task switches — composer is always the LAST task
      3. Fallback: if no agent info after many chunks, stream everything

    Args:
        num_tasks: Total tasks in crew (intermediate + composer). Default 2.

    Returns:
        (callback_fn, state_dict)
    """
    state = {
        "streamed": False,
        "header_printed": False,
        "task_switches": 0,
        "current_task": "",
        "chunk_count": 0,
        "is_last_task": False,
    }

    def callback(chunk):
        state["chunk_count"] += 1

        # Try to detect agent/task from chunk attributes
        agent = (
            getattr(chunk, "agent_role", None)
            or getattr(chunk, "agent", None)
            or getattr(chunk, "role", None)
            or ""
        )
        task = getattr(chunk, "task_name", None) or ""

        # Detect task switch
        task_key = f"{agent}|{task}"
        if task_key and task_key != state["current_task"] and (agent or task):
            state["current_task"] = task_key
            state["task_switches"] += 1

            # Show status for intermediate tasks
            label = agent or task or f"Task {state['task_switches']}"
            if state["task_switches"] < num_tasks:
                print(f"\n🔄 {label} working...", flush=True)
            else:
                state["is_last_task"] = True

        # Determine if this is composer output
        is_composer = False
        if agent and "composer" in str(agent).lower():
            is_composer = True
            state["is_last_task"] = True
        elif state["is_last_task"]:
            is_composer = True
        elif state["task_switches"] >= num_tasks:
            is_composer = True
            state["is_last_task"] = True
        # Fallback: if no task info detected after 50+ chunks, just stream
        elif state["chunk_count"] > 50 and state["task_switches"] == 0:
            is_composer = True

        content = getattr(chunk, "content", None) or ""
        if not is_composer or not content:
            return

        if not state["header_printed"]:
            print("\n\nAssistant: ", end="", flush=True)
            state["header_printed"] = True
        print(content, end="", flush=True)
        state["streamed"] = True

    return callback, state


def main():
    ap = argparse.ArgumentParser(description="Multi-turn chat with conversation memory")
    ap.add_argument("--config_dir", default=str(Path(__file__).resolve().parent / "config"))
    ap.add_argument("--session", "-s", default=None, help="Session ID (auto-generated if not set)")
    ap.add_argument("--query", "-q", default=None, help="Single query (non-interactive)")
    ap.add_argument("--student_id", default=None)
    ap.add_argument("--from_date", default=None)
    ap.add_argument("--to_date", default=None)
    ap.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    ap.add_argument("--stream", action="store_true",
                    help="Stream final answer token-by-token (Perplexity-style)")
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")

    # ── CLI overrides for memory.yaml values ──
    ap.add_argument("--compressor", default=None,
                    choices=["llm_summary", "llmlingua", "hybrid"],
                    help="Override compression strategy from memory.yaml")
    ap.add_argument("--data-dir", default=None, help="Override memory data directory")
    ap.add_argument("--max-context-tokens", type=int, default=None,
                    help="Override max context tokens before compression triggers")
    ap.add_argument("--max-recent-turns", type=int, default=None,
                    help="Override number of recent turns kept verbatim")
    ap.add_argument("--backend", default=None, choices=["memorydb", "redis"],
                    help="Override storage backend")
    ap.add_argument("--expand", default=None, choices=["auto", "confirm", "off"],
                    help="Override query expansion mode (off=disabled)")

    # ── Utility commands ──
    ap.add_argument("--stats", action="store_true", help="Show session stats and exit")
    ap.add_argument("--list-sessions", action="store_true", help="List all sessions and exit")
    ap.add_argument("--clear", action="store_true", help="Clear session history and exit")
    args = ap.parse_args()

    _setup_logging(debug=args.debug)
    log = logging.getLogger("school_agents.run_chat")

    # Init config
    cfg = load_config(args.config_dir)
    bootstrap(cfg)

    # Build CLI overrides dict (only non-None values)
    overrides = {}
    if args.compressor is not None:
        overrides["compressor"] = args.compressor
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.max_context_tokens is not None:
        overrides["max_context_tokens"] = args.max_context_tokens
    if args.max_recent_turns is not None:
        overrides["max_recent_turns"] = args.max_recent_turns
    if args.backend is not None:
        overrides["backend"] = args.backend
    if args.expand is not None:
        overrides["expand"] = args.expand

    # Apply expand override to frozen MemoryConfig
    if "expand" in overrides:
        from dataclasses import replace as dc_replace
        expand_val = overrides["expand"]
        if expand_val == "off":
            cfg = AppConfig(
                llm=cfg.llm, tools=cfg.tools, agents=cfg.agents, tasks=cfg.tasks,
                memory=dc_replace(cfg.memory, expand_enabled=False),
            )
        else:
            cfg = AppConfig(
                llm=cfg.llm, tools=cfg.tools, agents=cfg.agents, tasks=cfg.tasks,
                memory=dc_replace(cfg.memory, expand_enabled=True, expand_mode=expand_val),
            )

    # ── List sessions (no session_id needed) ──

    if args.list_sessions:
        # Need a bank to list — create temp one from config
        from .memory_bank import MemoryDB as _MemDB
        data_dir = overrides.get("data_dir", cfg.memory.data_dir)
        bank = _MemDB(data_dir=data_dir)
        keys = bank.search("session:")
        if not keys:
            print("No sessions found.")
        else:
            print(f"Found {len(keys)} session(s):")
            for k in sorted(keys):
                data = bank.get(k)
                sid = k.replace("session:", "")
                n_turns = data.get("turn_count", 0) if data else 0
                updated = data.get("updated_at", 0) if data else 0
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(updated)) if updated else "?"
                print(f"  {sid}  │  {n_turns} turns  │  last: {ts}")
        bank.close()
        return

    # Session ID
    session_id = args.session or str(uuid.uuid4())[:8]
    memory, bank = _create_memory(session_id, cfg, overrides)

    if args.stats:
        stats = memory.get_stats()
        print(f"Session: {session_id}")
        print(f"Config:  compressor={cfg.memory.compressor} "
              f"max_context_tokens={cfg.memory.max_context_tokens} "
              f"max_recent_turns={cfg.memory.max_recent_turns}")
        if overrides:
            print(f"CLI overrides: {overrides}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        bank.close()
        return

    if args.clear:
        memory.clear()
        print(f"Session {session_id} cleared.")
        bank.close()
        return

    # ── Chat modes ──

    mcfg = cfg.memory
    eff_compressor = overrides.get("compressor", mcfg.compressor)
    eff_ctx = overrides.get("max_context_tokens", mcfg.max_context_tokens)
    eff_turns = overrides.get("max_recent_turns", mcfg.max_recent_turns)

    eff_expand = "off"
    if cfg.memory.expand_enabled:
        eff_expand = cfg.memory.expand_mode

    eff_stream = "on" if args.stream else "off"

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Session: {session_id:<39s}║")
    print(f"║  Turns so far: {memory.turn_count:<34d}║")
    print(f"║  Compressor: {eff_compressor:<36s}║")
    print(f"║  Max context tokens: {eff_ctx:<28d}║")
    print(f"║  Max recent turns: {eff_turns:<30d}║")
    print(f"║  Query expansion: {eff_expand:<31s}║")
    print(f"║  Streaming: {eff_stream:<37s}║")
    print(f"╚══════════════════════════════════════════════════╝")

    if args.interactive:
        # ── Interactive REPL ──
        print("\nInteractive mode. Type 'quit' to exit, 'stats' for memory stats, 'facts' for knowledge graph.\n")
        print("Commands: quit, stats, facts, clear, expand <query>\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break
            if query.lower() == "stats":
                for k, v in memory.get_stats().items():
                    print(f"  {k}: {v}")
                continue
            if query.lower() == "clear":
                memory.clear()
                print("History cleared.")
                continue
            if query.lower() == "facts":
                if memory.facts:
                    ctx = memory.facts.to_context_string()
                    if ctx:
                        print(ctx)
                    else:
                        print("No facts extracted yet.")
                    print(f"\n  Stats: {memory.facts.get_stats()}")
                else:
                    print("Fact extraction disabled.")
                continue
            if query.lower().startswith("expand "):
                # Manual expansion test — just show expanded queries, don't search
                test_q = query[7:].strip()
                if test_q:
                    try:
                        oai = make_openai_client(cfg)
                        model = cfg.llm.model
                        if model.startswith("openai/"):
                            model = model[len("openai/"):]
                        from .tools.web_tools import expand_queries_only
                        print("⏳ Expanding", end="", flush=True)
                        def _ep(n): print(".", end="", flush=True)
                        expanded = expand_queries_only(test_q, oai, model,
                                                       progress_callback=_ep)
                        print(" done!")
                        print(f"\nExpanded queries for: \"{test_q}\"")
                        for i, eq in enumerate(expanded, 1):
                            print(f"  {i}. {eq}")
                        print()
                    except Exception as exc:
                        print(f"  Error: {exc}")
                else:
                    print("Usage: expand <your question>")
                continue

            try:
                t0 = time.perf_counter()

                def _status(msg):
                    print(f"\n{msg}", flush=True)

                if args.stream:
                    cb, cb_state = _make_stream_callback()
                    answer = _run_one_turn(
                        cfg, memory, query,
                        args.student_id, args.from_date, args.to_date,
                        interactive=True,
                        stream_callback=cb,
                        status_callback=_status,
                    )
                    elapsed = time.perf_counter() - t0
                    if cb_state["streamed"]:
                        print(f"\n({elapsed:.1f}s)\n")
                    else:
                        print(f"\nAssistant ({elapsed:.1f}s):\n{answer}\n")
                else:
                    answer = _run_one_turn(
                        cfg, memory, query,
                        args.student_id, args.from_date, args.to_date,
                        interactive=True,
                        status_callback=_status,
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"\nAssistant ({elapsed:.1f}s):\n{answer}\n")

                bank.flush()  # persist to JSONL immediately
            except Exception as exc:
                log.error("Error: %s", exc, exc_info=args.debug)
                print(f"\n⚠️  Error: {exc}\n")

    elif args.query:
        # ── Single query ──
        try:
            if args.stream:
                cb, cb_state = _make_stream_callback()
                answer = _run_one_turn(
                    cfg, memory, args.query,
                    args.student_id, args.from_date, args.to_date,
                    interactive=False,
                    stream_callback=cb,
                )
                if cb_state["streamed"]:
                    print()  # final newline after streaming
                else:
                    print(answer)
            else:
                answer = _run_one_turn(
                    cfg, memory, args.query,
                    args.student_id, args.from_date, args.to_date,
                    interactive=False,
                )
                print(answer)
            bank.flush()
        except Exception as exc:
            log.error("Error: %s", exc, exc_info=True)
            sys.exit(1)

    else:
        print("Use --query 'your question' or --interactive for REPL mode.")
        ap.print_help()

    # Cleanup
    bank.close()


if __name__ == "__main__":
    main()
