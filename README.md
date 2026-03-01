# Simple Multi-Agent (CrewAI) — YAML-driven, OpenAI-compatible

This is a **small**, **config-first** multi-agent template:
- One shared repo for multi-agent orchestration (CrewAI)
- Works with **any OpenAI-compatible endpoint** (local Kimi via llama.cpp/vLLM, OpenAI, etc.)
- Claude support via either CrewAI native Anthropic integration OR an OpenAI-compatible proxy (e.g., LiteLLM)
- Tools:
  - Deep web search + **multi-page crawl** (Tavily Search + Tavily Crawl)
  - 3 Student APIs (profile / grades / attendance)
  - Internal RAG API (hybrid dense+BM25 + rerank)
  - Optional STT/TTS "audio I/O" wrapper

## Why CrewAI?
CrewAI supports:
- YAML-based agent/task definitions (recommended in their docs)
- Connecting to OpenAI-compatible LLMs via `base_url` + `api_key`
- Async kickoff and streaming output modes

Docs refs:
- LLM connection / OpenAI-compatible base_url
- Streaming crew execution
- Async kickoff

## Quick start

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure
Copy and edit:
- `src/school_agents/config/llm.yaml`
- `src/school_agents/config/tools.yaml`

### 3) Run (sync)
```bash
python -m school_agents.run --query "Tra cứu quy chế môn Y khoa về thực tập lâm sàng" --stream
```

### 4) Run (async)
```bash
python -m school_agents.run_async --query "Tổng hợp điểm của học sinh 123 từ 2025-09-01 đến 2026-01-31" --student_id 123 --from_date 2025-09-01 --to_date 2026-01-31
```

### 5) Run as an API (optional)
```bash
uvicorn school_agents.server:app --host 0.0.0.0 --port 8080
```

Then:
- POST `/run` (wait for full answer)
- POST `/run/stream` (SSE stream chunks)

## LLM endpoint examples

### Local Kimi via llama.cpp `llama-server`
Point `base_url` to your OpenAI-compatible server (e.g. `http://localhost:8000/v1`).

### Claude
Two easy options:
1) Use CrewAI native Anthropic provider
2) Use an OpenAI-compatible proxy (e.g. LiteLLM) and keep this repo unchanged

## Notes on deep web crawl
The `web_crawl_url` tool uses Tavily Crawl:
- `max_depth` (1..5)
- `limit` (# pages)
- `select_paths` / `exclude_paths` regex filters
This lets you start from **one URL** and collect many relevant pages.

## Security
If you switch to MCP servers (Tavily MCP / Playwright MCP), put a policy gateway in front:
- allowlist hosts/tools
- validate URLs to avoid SSRF
- rate limit
- log/audit tool calls

See MCP security best practices.

---

This repo is a template. Replace the student endpoints + RAG endpoints with your real internal services.
