from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .config import load_config
from .crew_runner import bootstrap, route, run_crew_async


async def amain():
    ap = argparse.ArgumentParser(description="Run multi-agent crew (async)")
    ap.add_argument("--config_dir", default=str(Path(__file__).resolve().parent / "config"))
    ap.add_argument("--query", required=True)
    ap.add_argument("--student_id", default=None)
    ap.add_argument("--from_date", default=None)
    ap.add_argument("--to_date", default=None)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config_dir)
    bootstrap(cfg)

    # Step 1: route (sync is fine here, it's a short call)
    routing = route(cfg, args.query, args.student_id, args.from_date, args.to_date)
    routes = routing.get("routes") or ["web"]
    policy_domain = routing.get("policy_domain", "other")

    inputs = {
        "user_query": args.query,
        "student_id": args.student_id,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "policy_domain": policy_domain,
    }

    # Step 2: async execute via akickoff()
    if args.stream:
        streaming = await run_crew_async(cfg, routes, inputs, stream=True)
        async for chunk in streaming:
            print(chunk.content, end="", flush=True)
        print("\n\nFINAL:\n", streaming.result.raw)
    else:
        result = await run_crew_async(cfg, routes, inputs, stream=False)
        print(result.raw)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
