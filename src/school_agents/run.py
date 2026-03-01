from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .config import load_config
from .crew_runner import bootstrap, route, run_crew


def _setup_logging(debug: bool = False):
    """Configure logging for the entire app chain."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = (
        "%(asctime)s │ %(levelname)-5s │ %(name)-40s │ %(message)s"
        if debug else
        "%(asctime)s │ %(levelname)-5s │ %(message)s"
    )
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    if debug:
        # ── LiteLLM: see every HTTP request/response to LLM ──
        try:
            import litellm
            litellm.set_verbose = True
            # Also capture litellm logs
            logging.getLogger("LiteLLM").setLevel(logging.DEBUG)
            logging.getLogger("litellm").setLevel(logging.DEBUG)
        except ImportError:
            pass

        # ── OpenAI SDK: see raw HTTP if litellm uses it ──
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)

        # ── CrewAI internals ──
        logging.getLogger("crewai").setLevel(logging.DEBUG)

        # ── Our app ──
        logging.getLogger("school_agents").setLevel(logging.DEBUG)
    else:
        # Even without --debug, show our INFO logs
        logging.getLogger("school_agents").setLevel(logging.INFO)
        # Silence noisy libs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def main():
    ap = argparse.ArgumentParser(description="Run multi-agent crew (sync)")
    ap.add_argument("--config_dir", default=str(Path(__file__).resolve().parent / "config"))
    ap.add_argument("--query", required=True)
    ap.add_argument("--student_id", default=None)
    ap.add_argument("--from_date", default=None)
    ap.add_argument("--to_date", default=None)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Enable full debug logging (LLM requests, tool calls, etc.)")
    args = ap.parse_args()

    _setup_logging(debug=args.debug)
    log = logging.getLogger("school_agents.run")

    log.info("=" * 70)
    log.info("STARTING school_agents  debug=%s", args.debug)
    log.info("query=%r  student_id=%s", args.query, args.student_id)
    log.info("=" * 70)

    cfg = load_config(args.config_dir)
    bootstrap(cfg)

    # ── Step 1: route ──
    t0 = time.perf_counter()
    log.info("── PHASE 1: ROUTING ──")
    routing = route(cfg, args.query, args.student_id, args.from_date, args.to_date)
    routes = routing.get("routes") or ["web"]
    policy_domain = routing.get("policy_domain", "other")
    log.info(
        "── ROUTING DONE in %.1fs → routes=%s policy=%s ──",
        time.perf_counter() - t0, routes, policy_domain,
    )

    inputs = {
        "user_query": args.query,
        "student_id": args.student_id,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "policy_domain": policy_domain,
        "conversation_context": "",  # no memory in single-shot mode
    }

    # ── Step 2: execute ──
    t1 = time.perf_counter()
    log.info("── PHASE 2: CREW EXECUTION (routes=%s) ──", routes)
    try:
        if args.stream:
            streaming = run_crew(cfg, routes, inputs, stream=True)
            for chunk in streaming:
                print(chunk.content, end="", flush=True)
            print("\n\n---\nFINAL:\n")
            print(streaming.result.raw)
        else:
            out = run_crew(cfg, routes, inputs, stream=False)
            log.info(
                "── CREW DONE in %.1fs ── output=%d chars",
                time.perf_counter() - t1, len(out.raw),
            )
            print(out.raw)
    except Exception as exc:
        log.error("── CREW FAILED after %.1fs ──", time.perf_counter() - t1, exc_info=True)
        raise

    log.info(
        "── TOTAL TIME: %.1fs ──",
        time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
