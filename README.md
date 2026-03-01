<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/framework-CrewAI-orange.svg" alt="CrewAI">
  <img src="https://img.shields.io/badge/search-Tavily-green.svg" alt="Tavily">
  <img src="https://img.shields.io/badge/API-OpenAI_Compatible-blueviolet.svg" alt="OpenAI Compatible">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License">
</p>

# 🔍 Perplexity-Style Multi-Agent Search Framework

**An open, extensible framework that turns any local LLM into a Perplexity-like AI assistant — with smart web search, multi-agent orchestration, conversation memory, and a plug-and-play tool system.**

Works with any OpenAI-compatible API: LMStudio, vLLM, Ollama, or cloud providers.

---

> **Tiếng Việt**: Xem phần [Hướng dẫn tiếng Việt](#-hướng-dẫn-tiếng-việt) bên dưới.

---

## Why This Exists

Cloud AI search tools (Perplexity, ChatGPT Search, Gemini) are powerful but closed, expensive, and can't be customized. This framework gives you the same experience — intelligent query expansion, web search with citations, streaming answers — running on **your own infrastructure** with **any LLM**.

**What makes it different from a basic RAG chatbot:**

- 🧠 **Multi-agent architecture** — Specialized agents (router, researcher, composer) collaborate instead of one monolithic prompt
- 🔍 **Smart query expansion** — One user question becomes 3 optimized search queries (English + native language) for maximum coverage
- 💬 **Conversation memory** — Fact extraction, context compression, multi-turn awareness across sessions
- 🔌 **Plug-and-play tools** — Add your own data sources (student APIs, internal RAG, databases) alongside web search
- 🌊 **Real-time streaming** — Token-by-token output with SSE support for web clients
- 📡 **OpenAI-compatible API** — Drop-in replacement for any app expecting `/v1/chat/completions`

## Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────────────┐
│  Router  │────▶│  Query Expander  │──── 3 optimized search queries
│  Agent   │     └──────────────────┘
└────┬─────┘              │
     │                    ▼
     │         ┌────────────────────┐
     ├────────▶│  Web Researcher    │──── Tavily deep search + crawl
     │         └────────────────────┘
     │         ┌────────────────────┐
     ├────────▶│  Student Data API  │──── Internal REST APIs
     │         └────────────────────┘
     │         ┌────────────────────┐
     ├────────▶│  Policy RAG        │──── Vector search + reranking
     │         └────────────────────┘
     │                    │
     │                    ▼
     │         ┌────────────────────┐
     └────────▶│  Answer Composer   │──── Synthesize + cite sources
               └────────────────────┘
                          │
                          ▼
              Streaming Vietnamese/English
              answer with numbered sources
```

**Agent Roles:**

| Agent | Purpose | Tools |
|-------|---------|-------|
| **Router** | Classify intent, pick which agents to invoke | `get_datetime` |
| **Web Researcher** | Search the web, evaluate freshness, compile findings | `web_search_deep`, `web_search_expanded`, `web_crawl_url` |
| **Student Data** | Fetch student profiles, grades, attendance from APIs | `student_get_profile`, `student_get_grades` |
| **Policy RAG** | Query internal policy documents with hybrid search | `rag_query_policy` |
| **Composer** | Synthesize all evidence into a cited, natural answer | None (synthesis only) |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/simple-multi-agent-crewai.git
cd simple-multi-agent-crewai
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
nano .env  # Add your API keys
```

```env
LLM_API_KEY=your-lm-studio-or-vllm-key
TAVILY_API_KEY=your-tavily-api-key
```

Point to your LLM server in `src/school_agents/config/llm.yaml`:

```yaml
llm:
  model: "qwen/qwen3-32b"              # any model your server hosts
  base_url: "http://localhost:1234/v1"  # LMStudio, vLLM, Ollama
  api_key: "${LLM_API_KEY}"            # resolved from .env
  max_tokens: 8192
  structured_max_tokens: 16384         # thinking models: 16384, non-thinking: 4096
```

### 3. Run

**Interactive CLI (recommended):**
```bash
python -m school_agents.run_chat --session my_session --interactive --stream
```

**Single query:**
```bash
python -m school_agents.run --query "Latest developments in AI" --stream
```

**FastAPI server:**
```bash
uvicorn school_agents.server:app --host 0.0.0.0 --port 8000
```

```bash
# Sync
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query": "What happened in tech today?"}'

# Streaming (SSE)
curl -X POST http://localhost:8000/run/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "US-Iran situation analysis"}'
```

## Features in Detail

### 🔍 Smart Query Expansion

One question becomes 3 targeted searches for maximum coverage:

```
User: "Tình hình chiến tranh Mỹ Iran thế nào?"

Expanded:
  1. "US Iran war latest military updates March 2026"     ← English (broad coverage)
  2. "tình hình chiến tranh Mỹ Iran mới nhất"             ← Vietnamese (local sources)
  3. "US Iran conflict expert analysis escalation forecast" ← English (analytical angle)
```

Date-aware: the expander knows today's date and includes temporal context automatically.

### 🧠 Conversation Memory

- **Turn history** — Maintains context across messages within a session
- **Fact extraction** — Automatically extracts entities, relations, and facts from every exchange
- **Context compression** — LLM-powered summarization of older turns to stay within token limits
- **Persistent storage** — Sessions saved to disk, survives restarts

### 🌊 Streaming

- Token-by-token streaming from the Composer agent only (intermediate agents are filtered)
- Progress indicators during query expansion
- Status callbacks during post-processing
- SSE endpoint for web clients via FastAPI

### ⏰ Date/Time Awareness

Every agent receives the current date/time injected into its system prompt:

```
[SYSTEM TIME] Hôm nay là Thứ Bảy, ngày 01/03/2026, giờ 11:30:45 (Asia/Ho_Chi_Minh).
English: Saturday, March 01, 2026 at 11:30:45 +07.
```

No tool calls needed — agents reason about "next 72 hours", "yesterday", "this week" immediately.

### 🔧 Thinking Model Support

Built for thinking models (Qwen3, Qwen3.5) that output `<think>...</think>` tags:

- `extract_json()` — Robust JSON extraction from outputs mixed with thinking text
- `strip_think_tags()` — Clean display vs storage separation
- `structured_max_tokens` — One YAML setting controls token budget for all structured LLM calls

## Adding Your Own Tools

The framework is designed to be extended. Add a new data source in 5 steps:

**1. Create the tool** (`tools/my_tools.py`):
```python
from crewai.tools import tool

@tool("my_database_search")
def my_database_search(query: str) -> str:
    """Search the internal database for relevant records."""
    results = my_db.search(query)
    return json.dumps(results)
```

**2. Add the agent** (`config/agents.yaml`):
```yaml
my_specialist:
  role: "Database Specialist"
  goal: "Search internal database and return structured results."
  backstory: >
    You search the internal database. Call my_database_search ONCE,
    then immediately provide your Final Answer with the results.
  allow_delegation: false
```

**3. Add the task** (`config/tasks.yaml`):
```yaml
my_task:
  description: |
    Search the database for: {user_query}
    Write your response starting with "Final Answer:" followed by JSON.
  expected_output: "Final Answer: JSON with results."
  agent: my_specialist
```

**4. Wire it up** — Add your agent to `_make_agents()` and route in `crew_runner.py`

**5. Update routing** — Add `"my_route"` to the router logic in `agents.yaml`

## Project Structure

```
simple-multi-agent-crewai/
├── .env.example                    # Template for API keys
├── pyproject.toml
├── README.md
└── src/school_agents/
    ├── config.py                   # Config loader with ${ENV_VAR} resolution
    ├── crew_runner.py              # Agent/task/crew orchestration
    ├── run_chat.py                 # Interactive CLI (memory + streaming)
    ├── run.py                      # Single-shot CLI
    ├── server.py                   # FastAPI + SSE streaming server
    ├── conversation_memory.py      # Turn management + fact integration
    ├── memory_bank.py              # Persistent session storage (JSONL)
    ├── fact_store.py               # Entity/relation/fact extraction
    ├── context_compressor.py       # LLM summary / LLMLingua compression
    ├── query_expander.py           # Multi-query expansion with date awareness
    ├── llm_utils.py                # Think tag handling, JSON extraction
    ├── config/
    │   ├── llm.yaml                # LLM connection + generation params
    │   ├── tools.yaml              # Tool API keys + endpoints
    │   ├── agents.yaml             # Agent roles, goals, backstories
    │   ├── tasks.yaml              # Task descriptions + expected outputs
    │   └── memory.yaml             # Memory, compression, expansion settings
    └── tools/
        ├── web_tools.py            # Tavily search, crawl, expand+search
        ├── student_tools.py        # Student API integration (example)
        ├── rag_tools.py            # Policy RAG with hybrid search (example)
        ├── datetime_tools.py       # Vietnamese-aware date/time
        └── speech_tools.py         # STT/TTS (optional)
```

## Configuration Reference

All config lives in `src/school_agents/config/`. Secrets use `${ENV_VAR}` syntax, auto-resolved from `.env`.

| File | Purpose | Key settings |
|------|---------|--------------|
| `llm.yaml` | LLM connection | `model`, `base_url`, `max_tokens`, `structured_max_tokens` |
| `tools.yaml` | External APIs | Tavily, student APIs, RAG, audio endpoints |
| `agents.yaml` | Agent prompts | Role, goal, backstory for each agent |
| `tasks.yaml` | Task prompts | Description, expected output, agent assignment |
| `memory.yaml` | Memory system | Turn limits, compression strategy, query expansion |

## Supported LLM Backends

| Backend | Status | Notes |
|---------|--------|-------|
| **LMStudio** | ✅ Tested | Recommended for local dev |
| **vLLM** | ✅ Tested | Best for production GPU serving |
| **Ollama** | ✅ Tested | Easy setup, use `/v1` endpoint |
| **OpenAI** | ✅ Tested | Cloud fallback |
| **Any OpenAI-compatible** | ✅ | Just set `base_url` and `api_key` |

## Tech Stack

- **[CrewAI](https://github.com/crewAIInc/crewAI)** — Multi-agent orchestration with ReAct loop
- **[Tavily](https://tavily.com)** — Web search API (deep search, crawl, extract)
- **[FastAPI](https://fastapi.tiangolo.com)** — Async API server with SSE streaming
- **[LiteLLM](https://github.com/BerriAI/litellm)** — Universal LLM gateway
- **[json-repair](https://github.com/mangiucugna/json_repair)** — Robust JSON parsing from LLM output

---

## 🇻🇳 Hướng dẫn tiếng Việt

### Giới thiệu

Framework mã nguồn mở giúp biến bất kỳ LLM local nào thành trợ lý AI thông minh kiểu Perplexity — tìm kiếm web thông minh, trả lời có nguồn dẫn, streaming realtime. Chạy trên hạ tầng của bạn, dùng model của bạn.

### Tính năng chính

| | Tính năng | Chi tiết |
|---|-----------|----------|
| 🔍 | **Tìm kiếm thông minh** | 1 câu hỏi → 3 queries tối ưu (Anh + Việt) → kết quả toàn diện |
| 🤖 | **Đa agent** | Router → Researcher → Composer, không dùng 1 prompt đơn |
| 💬 | **Nhớ hội thoại** | Trích xuất facts, nén context, nhớ xuyên phiên |
| 🔌 | **Mở rộng dễ** | Thêm API nội bộ, RAG, database bên cạnh web search |
| 🌊 | **Streaming** | Token-by-token, SSE cho web client |
| ⏰ | **Biết ngày giờ** | Tự inject thời gian, hỗ trợ "72h tới", "tuần này" |

### Cài đặt nhanh

```bash
git clone https://github.com/YOUR_USERNAME/simple-multi-agent-crewai.git
cd simple-multi-agent-crewai

pip install -e .

cp .env.example .env
nano .env  # Điền: LLM_API_KEY, TAVILY_API_KEY

nano src/school_agents/config/llm.yaml  # Chỉnh model + base_url

python -m school_agents.run_chat --session test --interactive --stream
```

### Ví dụ

```
You: Tình hình chiến tranh Mỹ Iran thế nào? Dự đoán 72h tới?

⏳ Generating search queries.......... done!
🔍 Query expansion (3 queries):
   1. US Iran war latest military updates March 2026
   2. tình hình chiến tranh Mỹ Iran mới nhất
   3. US Iran conflict expert analysis escalation forecast
   Search all? [Y/n/edit]
   Searching 3 queries...
   ✅ 5 results merged.
🔄 Web Researcher working...

A: Theo các nguồn tin mới nhất, Mỹ và Israel đã tiến hành không kích
vào Iran...

Nguồn:
[1] https://www.livenowfox.com/news/us-military-strikes-iran
[2] https://vnexpress.net/hanh-trinh-my-iran-...

💾 Saving to memory...
🧠 Facts: +5E +3R +4F
✅ Done
(89.2s)
```

### Mở rộng

Framework thiết kế để thêm tool dễ dàng. Ví dụ đã có sẵn:
- `student_tools.py` — Tra cứu thông tin học sinh qua REST API
- `rag_tools.py` — Tìm kiếm chính sách/quy chế qua vector search
- `speech_tools.py` — Chuyển giọng nói thành text và ngược lại

Xem hướng dẫn chi tiết tại phần [Adding Your Own Tools](#adding-your-own-tools).

---

## License

MIT

## Contributing

PRs welcome. Please open an issue first to discuss what you would like to change.
