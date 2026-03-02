<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/framework-CrewAI-orange.svg" alt="CrewAI">
  <img src="https://img.shields.io/badge/search-Tavily-green.svg" alt="Tavily">
  <img src="https://img.shields.io/badge/backend-vLLM_%7C_Ollama_%7C_LMStudio-blueviolet.svg" alt="vLLM | Ollama | LMStudio">
  <img src="https://img.shields.io/badge/🖼️_vision-Multimodal-ff6b6b.svg" alt="Multimodal Vision">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License">
</p>

<h1 align="center">🔍 Multi-Agent Search Framework</h1>

<p align="center">
  <strong>Perplexity-style answers · Native image understanding · Near-infinite memory · Streaming</strong>
</p>

<p align="center">
  An open framework that turns any local LLM into a Perplexity-like AI assistant —<br>
  with smart web search, multi-agent orchestration, multimodal vision, conversation memory,<br>
  and a plug-and-play tool system. Your infrastructure, your models, your data.
</p>

---

> **🇻🇳 Tiếng Việt** — Xem phần [Hướng dẫn tiếng Việt](#-hướng-dẫn-tiếng-việt) bên dưới.

---

## ✨ What's New — Multimodal Vision

Send images alongside your questions. Vision models see the raw pixels — no pre-description, no lossy OCR, no information loss.

```
You: img:images/prescription.jpg img:images/pill.png Thuốc gì đây? Liều dùng?

  📎 images/prescription.jpg
  📎 images/pill.png
⏳ Generating search queries.. done!
🔍 3 queries → 12 results merged

A: Đây là thuốc Amoxicillin 500mg, kháng sinh nhóm Penicillin.
   Liều dùng thông thường cho người lớn: 500mg mỗi 8 giờ [S1].
   Toa thuốc ghi liều 3 lần/ngày × 7 ngày, phù hợp với
   hướng dẫn điều trị nhiễm khuẩn đường hô hấp [S2][S3].

   Nguồn:
   [S1] MIMS Vietnam — https://mims.com/vietnam/...
   [S2] BYT Hướng dẫn sử dụng kháng sinh — https://...
   [S3] Drugs.com Amoxicillin — https://drugs.com/...

💾 Memory saved · 🧠 +5 entities +3 relations +4 facts
(24.7s)
```

The model **saw** both the prescription scan and the pill photo, cross-referenced with web search results, and composed a cited Vietnamese answer — all in one turn.

---

## Why This Exists

Cloud AI search tools (Perplexity, ChatGPT Search, Gemini) are powerful but closed, expensive, and can't be customized. This framework gives you the same experience running on **your own infrastructure** with **any LLM**.

| | This Framework | Basic RAG Chatbot |
|---|---|---|
| 🖼️ **Vision** | Native multimodal — VLM sees raw images via ChatML | ❌ Text only |
| 🤖 **Architecture** | 5 specialized agents collaborate | 1 monolithic prompt |
| 🔍 **Search** | 3 expanded queries × RRF merge = 10-15 diverse results | 1 query, top-3 |
| 💬 **Memory** | 3-tier: facts (permanent) + summary + recent turns | Last N messages |
| 🔌 **Tools** | Plug-and-play: web, APIs, RAG, databases | Web search only |
| 🌊 **Streaming** | Token-by-token via CLI + SSE API | Usually blocking |
| ⏰ **Time-aware** | Agents know today's date, reason about "72h tới" | Static |
| 🧠 **Thinking models** | Native `<think>` tag handling (Qwen3, QwQ) | ❌ |

---

## Architecture

```
                        User: text + images
                              │
                ┌─────────────┼─────────────┐
                │                           │
           ┌────▼─────┐              ┌──────▼──────┐
           │  Router   │              │ img:a.jpg   │
           │  Agent    │              │ img:b.png   │
           └────┬──────┘              │  ↓ base64   │
                │                     └──────┬──────┘
       ┌────────┼────────┐                   │
       ▼        ▼        ▼                   │
  ┌─────────┐ ┌──────┐ ┌─────┐              │
  │  Web    │ │Stud. │ │ RAG │              │
  │Research.│ │ API  │ │     │              │
  └────┬────┘ └──┬───┘ └──┬──┘              │
       └─────────┼────────┘                  │
                 ▼                           │
         ┌──────────────┐                    │
         │   Composer    │◄───────────────────┘
         │   Agent       │     images injected via
         └──────┬───────┘      litellm vision patch
                │
                ▼
    ┌───────────────────────┐
    │  vLLM / Ollama        │
    │  (Qwen2.5-VL, etc.)  │
    │  Sees text + images   │
    └───────────┬───────────┘
                │
                ▼
      Streaming answer with
      [S1][S2] citations
                │
                ▼
    ┌───────────────────────┐
    │  3-Tier Memory Store  │
    │  Facts · Summary ·    │
    │  Recent (persistent)  │
    └───────────────────────┘
```

**Agents at a glance:**

| Agent | Job | Tools | Sees images? |
|-------|-----|-------|:---:|
| **Router** | Classify intent, pick agents | `get_datetime` | ✅ |
| **Web Researcher** | Multi-source search + verify | `web_search_deep`, `web_search_expanded`, `web_crawl_url` | ✅ |
| **Student Data** | Internal API lookups | `student_get_profile`, `student_get_grades`, `student_get_attendance` | — |
| **Policy RAG** | Policy/regulation vector search | `rag_query_policy` | — |
| **Composer** | Synthesize evidence → cited answer | *(synthesis only)* | ✅ |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/simple-multi-agent-crewai.git
cd simple-multi-agent-crewai
git checkout feature/multimodal
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

```env
LLM_API_KEY=your-key-here
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
```

Edit `src/school_agents/config/llm.yaml`:

```yaml
llm:
  model: "qwen/qwen2.5-vl-32b"         # vision model for image support
  # model: "qwen/qwen3-32b"            # text-only (images will be ignored)
  base_url: "http://localhost:1234/v1"  # LMStudio / vLLM / Ollama
  api_key: "${LLM_API_KEY}"
  temperature: 0.4
  max_tokens: 8192
  structured_max_tokens: 16384          # thinking models need more headroom
```

### 3. Run

```bash
# Interactive REPL with streaming (recommended)
python -m school_agents.run_chat --session demo --interactive --stream

# Single query with images
python -m school_agents.run_chat \
    -q "Compare these two prescriptions" \
    --image images/rx1.jpg images/rx2.jpg \
    --stream

# FastAPI server
uvicorn school_agents.server:app --host 0.0.0.0 --port 8000
```

---

## Multimodal Vision — How It Works

Images are injected at the lowest possible layer: a one-time monkey-patch on `litellm.completion`. When images are present, the last user message transforms from a plain string into the standard OpenAI Vision / ChatML multimodal array:

```python
# Without images (business as usual)
{"role": "user", "content": "What's the weather?"}

# With images (automatic transformation)
{"role": "user", "content": [
    {"type": "text",      "text": "What medicine is this?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4A..."}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO..."}},
]}
```

### CLI — Two Ways to Send Images

```bash
# Flag syntax (single query mode)
python -m school_agents.run_chat \
    -q "What's in these photos?" \
    --image photo1.jpg photo2.png \
    --stream

# Inline syntax (REPL mode)
You: img:images/thuoc.jpg What is this medicine?
  📎 images/thuoc.jpg

You: img:a.jpg img:b.jpg img:c.png Compare all three
  📎 a.jpg
  📎 b.jpg
  📎 c.png

You: Thanks, summarize our conversation     ← text only, no images
```

### API — Multipart Upload or JSON

```bash
# Multipart (file upload — web forms, curl)
curl -X POST http://localhost:8000/chat \
     -F 'query=Identify this medicine' \
     -F 'session_id=pharma01' \
     -F 'images=@pill_front.jpg' \
     -F 'images=@pill_back.jpg'

# JSON with base64 (mobile apps, programmatic clients)
curl -X POST http://localhost:8000/chat/json \
     -H 'Content-Type: application/json' \
     -d '{
       "query": "What is this?",
       "session_id": "pharma01",
       "images": [
         {"b64": "/9j/4AAQ...", "mime": "image/jpeg"},
         {"b64": "iVBORw0K...", "mime": "image/png"}
       ]
     }'
```

### Supported Vision Models

| Model | vLLM | Ollama | LMStudio | Notes |
|-------|:---:|:---:|:---:|-------|
| **Qwen2.5-VL** (7B/32B/72B) | ✅ | ✅ | ✅ | Recommended — strong Vietnamese + English |
| **InternVL2.5** | ✅ | — | — | Strong multilingual vision |
| **LLaVA-NeXT** | ✅ | ✅ | ✅ | Good general-purpose |
| **Gemma 3** | ✅ | ✅ | ✅ | Google's latest multimodal |

> Text-only models (Qwen3, Llama) will ignore image inputs or error — use a VLM for vision features.

---

## Smart Query Expansion

One question becomes 3 targeted searches for maximum recall:

```
User: "Doanh số FPT Retail phụ thuộc Long Châu đúng không?"

Expanded:
  1. "FPT Retail revenue dependence Long Châu 2026"
  2. "doanh số FPT Retail phụ thuộc Long Châu"
  3. "FPT Retail financial analysis Long Châu revenue share"

→ Tavily search × 3 queries × 5 results each
→ Reciprocal Rank Fusion merge + dedup
→ 12 unique results injected as [Pre-searched results]
```

Date-aware: the expander knows today's date and adds temporal context. Three modes: `auto` (silent), `confirm` (ask before searching), or `off`.

---

## Conversation Memory — 3-Tier Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Tier 1: FACTS                    Never deleted, permanent │
│  Entities + relations + facts extracted every turn.        │
│  Up to 10 per category per turn. Accumulates forever.      │
│                                                             │
│  "FPT_Retail → revenue_68%_from → Long_Châu"              │
│  "Amoxicillin → treats → respiratory_infection"             │
├────────────────────────────────────────────────────────────┤
│  Tier 3: SUMMARY                  LLM-compressed old turns │
│  Scales with conversation length:                           │
│    2-4 turns → 3-5 sentences                                │
│    5-10 turns → 1-2 paragraphs                              │
│    10+ turns → 2-4 paragraphs organized by topic            │
├────────────────────────────────────────────────────────────┤
│  Tier 2: RECENT TURNS             Last 4, verbatim          │
│  Size-guarded: auto-truncated if exceeds token budget.      │
│  Full fidelity for immediate conversation context.          │
└────────────────────────────────────────────────────────────┘

Persistent on disk (sessions.jsonl). Survives restarts.
Sessions load instantly on reconnect. Nothing is ever deleted.
```

---

## FastAPI Server

Full-featured API server with session management and multimodal support.

```bash
uvicorn school_agents.server:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Multimodal chat (multipart: text + image files) |
| `POST` | `/chat/stream` | SSE streaming (multipart: text + image files) |
| `POST` | `/chat/json` | JSON-only chat (base64 images in body) |
| `GET` | `/sessions/{id}` | Session stats: turns, facts, summary |
| `DELETE` | `/sessions/{id}` | Clear session history |
| `GET` | `/health` | Health check + model info |

### Form Fields (`/chat` and `/chat/stream`)

| Field | Type | Required | Description |
|-------|------|:---:|-------------|
| `query` | string | ✅ | User question |
| `session_id` | string | — | Session ID (auto-generated if omitted) |
| `images` | file(s) | — | Image files — JPEG, PNG, WebP. Multiple OK. |
| `student_id` | string | — | For student API lookups |
| `from_date` | string | — | Date filter (YYYY-MM-DD) |
| `to_date` | string | — | Date filter (YYYY-MM-DD) |

### Response

```json
{
  "answer": "Đây là thuốc Amoxicillin 500mg...",
  "session_id": "pharma01",
  "routes": ["web"],
  "elapsed_seconds": 24.7,
  "turn_count": 3
}
```

### SSE Events (`/chat/stream`)

```
event: chunk    data: {"content": "Đây", "agent": "composer"}
event: chunk    data: {"content": " là", "agent": "composer"}
event: chunk    data: {"content": " thuốc", "agent": "composer"}
...
event: done     data: {"answer": "...", "session_id": "pharma01", ...}
```

### Examples

```bash
# Text-only query
curl -X POST http://localhost:8000/chat \
     -F 'query=Tin tức AI mới nhất' \
     -F 'session_id=news01'

# With images
curl -X POST http://localhost:8000/chat \
     -F 'query=So sánh 2 toa thuốc' \
     -F 'session_id=pharma01' \
     -F 'images=@images/rx1.jpg' \
     -F 'images=@images/rx2.jpg'

# JSON body (mobile/programmatic clients)
curl -X POST http://localhost:8000/chat/json \
     -H 'Content-Type: application/json' \
     -d '{"query":"Thời tiết Hà Nội","session_id":"weather01"}'

# SSE streaming
curl -N -X POST http://localhost:8000/chat/stream \
     -F 'query=US-Iran situation analysis' \
     -F 'session_id=geo01'

# Session management
curl http://localhost:8000/sessions/pharma01
curl -X DELETE http://localhost:8000/sessions/pharma01
```

---

## Streaming

Token-by-token streaming from the Composer agent only — intermediate agents (router, researcher) are filtered out. The `Final Answer:` prefix required by CrewAI is auto-stripped before display.

```
⏳ Generating search queries.. done!
🔍 Query expansion (3 queries):
   1. ...
   2. ...
   3. ...
   Searching 3 queries...
   ✅ 12 results merged.
🔄 Web Researcher working...

A: [streaming tokens appear here...]

💾 Memory saved
🧠 +5E +3R +4F
(24.7s)
```

---

## Adding Your Own Tools

The framework is designed to be extended. Add a new data source in 5 steps:

**1. Create the tool** in `tools/my_tools.py`:
```python
from crewai.tools import tool

@tool("my_database_search")
def my_database_search(query: str) -> str:
    """Search the internal database for relevant records."""
    results = my_db.search(query)
    return json.dumps(results)
```

**2. Add the agent** in `config/agents.yaml`:
```yaml
my_specialist:
  role: "Database Specialist"
  goal: "Search internal database and return structured results."
  backstory: >
    You search the internal database. Call my_database_search ONCE,
    then return JSON prefixed with "Final Answer:" (system requirement).
  allow_delegation: false
```

**3. Add the task** in `config/tasks.yaml`:
```yaml
my_task:
  description: |
    Search the database for: {user_query}
    You MUST prefix your JSON with "Final Answer:" — SYSTEM REQUIREMENT.
  expected_output: "Final Answer: JSON with results."
  agent: my_specialist
```

**4. Wire it up** — Add your agent to `_make_agents()` and route in `crew_runner.py`

**5. Update routing** — Add `"my_route"` to the router logic in `agents.yaml` and `tasks.yaml`

---

## Project Structure

```
simple-multi-agent-crewai/
├── .env.example                    # API key template
├── pyproject.toml
├── README.md                       # ← you are here
├── ARCHITECTURE.md                 # Detailed flow diagrams (491 lines)
│
└── src/school_agents/
    │
    ├── server.py                   # FastAPI: /chat, /chat/stream, /chat/json
    ├── run_chat.py                 # Interactive CLI: REPL + images + streaming
    ├── run.py                      # Legacy single-shot CLI (no memory)
    │
    ├── crew_runner.py              # CrewAI orchestration + vision litellm patch
    ├── image_context.py            # Thread-local image store (set/get/clear)
    ├── config.py                   # YAML loader with ${ENV_VAR} resolution
    ├── tool_context.py             # Thread-local tool config
    ├── llm_utils.py                # <think> tag handling, JSON extraction
    │
    ├── conversation_memory.py      # 3-tier memory: facts + summary + recent
    ├── memory_bank.py              # JSONL persistent session backend
    ├── fact_store.py               # Entity / relation / fact extraction
    ├── context_compressor.py       # LLM summary compression (scales by length)
    ├── query_expander.py           # 3-query expansion + reciprocal rank fusion
    │
    ├── config/
    │   ├── llm.yaml                # Model, base_url, temperature, max_tokens
    │   ├── agents.yaml             # 5 agents: Perplexity-style prompts
    │   ├── tasks.yaml              # 5 tasks: output formats + Final Answer rules
    │   ├── tools.yaml              # Tavily, student API, RAG endpoints
    │   └── memory.yaml             # Compression, expansion, facts settings
    │
    ├── tools/
    │   ├── web_tools.py            # Tavily search, crawl, expand+search
    │   ├── student_tools.py        # Student REST API (example)
    │   ├── rag_tools.py            # Policy RAG / vector search (example)
    │   ├── datetime_tools.py       # Vietnamese-aware date/time
    │   └── speech_tools.py         # STT/TTS skeleton (optional)
    │
    └── images/                     # Test images folder
```

---

## Configuration Reference

All config lives in `src/school_agents/config/`. Secrets use `${ENV_VAR}` syntax, auto-resolved from `.env`.

| File | Purpose | Key Settings |
|------|---------|--------------|
| **llm.yaml** | LLM connection | `model`, `base_url`, `max_tokens`, `structured_max_tokens` |
| **tools.yaml** | External APIs | Tavily (`search_depth: advanced`, `max_results: 5`), student API, RAG |
| **agents.yaml** | Agent prompts | Perplexity-style routing, multi-source verification, claim-level citations |
| **tasks.yaml** | Task prompts | Output formats, `Final Answer:` requirement for CrewAI parser |
| **memory.yaml** | Memory system | Turn limits, compressor strategy, expand mode (`auto`/`confirm`/`off`), fact extraction |

---

## Supported Backends

| Backend | Text | Vision | Best For |
|---------|:---:|:---:|----------|
| **vLLM** | ✅ | ✅ | Production — continuous batching, PagedAttention, H100/H200 optimized |
| **Ollama** | ✅ | ✅ | Quick start — `ollama pull qwen2.5-vl` and go |
| **LMStudio** | ✅ | ✅ | Local dev — GUI model management |
| **OpenAI** | ✅ | ✅ | Cloud fallback — GPT-4o for vision |
| **Any OpenAI-compatible** | ✅ | ⚠️ | Vision requires ChatML multimodal format support |

---

## Tech Stack

| Component | Role |
|-----------|------|
| [**CrewAI**](https://github.com/crewAIInc/crewAI) | Multi-agent orchestration, ReAct loop, sequential task pipeline |
| [**Tavily**](https://tavily.com) | Web search API — deep search, crawl, extract (handles PDFs natively) |
| [**FastAPI**](https://fastapi.tiangolo.com) | Async API server — SSE streaming, multipart upload, session management |
| [**LiteLLM**](https://github.com/BerriAI/litellm) | Universal LLM gateway — patched for multimodal vision injection |
| [**json-repair**](https://github.com/mangiucugna/json_repair) | Robust JSON parsing from LLM output with thinking tags |

---

## 🇻🇳 Hướng dẫn tiếng Việt

### Giới thiệu

Framework mã nguồn mở biến bất kỳ LLM local nào thành trợ lý AI kiểu Perplexity — tìm kiếm web thông minh, trả lời có nguồn dẫn, streaming realtime, **hiểu ảnh trực tiếp qua vision model**, và nhớ hội thoại gần như vô tận.

Chạy trên hạ tầng của bạn, dùng model của bạn.

### Tính năng chính

| | Tính năng | Chi tiết |
|---|-----------|----------|
| 🖼️ | **Hiểu ảnh** | Gửi ảnh kèm câu hỏi — VLM nhìn ảnh gốc, không qua mô tả trung gian |
| 🔍 | **Tìm kiếm thông minh** | 1 câu → 3 queries (Anh + Việt + phân tích) → RRF merge → 10-15 kết quả |
| 🤖 | **Đa agent** | Router → Researcher → Composer, phong cách Perplexity với nguồn dẫn [S1][S2] |
| 💬 | **Nhớ hội thoại** | 3 tầng: facts vĩnh viễn + summary nén + 4 turns gần nhất. Nhớ xuyên phiên. |
| 🔌 | **Mở rộng dễ** | Thêm API nội bộ, RAG, database bên cạnh web search |
| 🌊 | **Streaming** | Token-by-token qua CLI và SSE cho web/mobile client |
| ⏰ | **Biết ngày giờ** | Tự inject thời gian, tính "72h tới", "tuần này" chính xác |
| 🧠 | **Thinking model** | Hỗ trợ Qwen3, QwQ — xử lý `<think>` tags, JSON extraction robust |

### Cài đặt & Chạy

```bash
git clone https://github.com/YOUR_USERNAME/simple-multi-agent-crewai.git
cd simple-multi-agent-crewai && git checkout feature/multimodal
pip install -e .

cp .env.example .env && nano .env      # Điền: LLM_API_KEY, TAVILY_API_KEY
nano src/school_agents/config/llm.yaml  # Chỉnh model + base_url
```

```bash
# CLI tương tác
python -m school_agents.run_chat --session test --interactive --stream

# Với ảnh
python -m school_agents.run_chat -q "Thuốc gì?" --image images/thuoc.jpg --stream
```

```bash
# REPL — dùng img: prefix
You: img:images/thuoc.jpg Thuốc gì đây?
  📎 images/thuoc.jpg
You: img:images/a.jpg img:images/b.png So sánh 2 toa thuốc
  📎 images/a.jpg
  📎 images/b.png
You: Tóm tắt cuộc trò chuyện     ← chỉ text, không ảnh
```

```bash
# Server API
uvicorn school_agents.server:app --host 0.0.0.0 --port 8000

# Text
curl -X POST http://localhost:8000/chat \
     -F 'query=Tin tức hôm nay' -F 'session_id=test'

# Với ảnh
curl -X POST http://localhost:8000/chat \
     -F 'query=Thuốc gì đây?' -F 'session_id=test' \
     -F 'images=@images/thuoc.jpg'

# Session
curl http://localhost:8000/sessions/test          # Xem stats
curl -X DELETE http://localhost:8000/sessions/test # Xóa
```

### Ví dụ chạy thực

```
You: Doanh số FPT Retail phụ thuộc chủ yếu vào Long Châu đúng không?

⏳ Generating search queries.. done!
🔍 Query expansion (3 queries):
   1. FPT Retail revenue dependence on Long Châu 2026
   2. Doanh số và thu nhập của FPT Retail phụ thuộc vào Long Châu
   3. FPT Retail financial performance analysis Long Châu impact
   Search all? [Y/n/edit]
   Searching 3 queries...
   ✅ 12 results merged.
🔄 Web Researcher working...

A: Đúng, Long Châu đóng góp 68% doanh thu FPT Retail năm 2025 [S4].
   Kinh doanh dược phẩm chiếm 91% tổng doanh thu dự kiến [S1].
   FPT Retail dự kiến tăng trưởng 15% năm 2026 nhờ mở rộng Long Châu [S3].

   Nguồn:
   [S1] FPT Retail (FRT) BUY — vietcap.com.vn/...
   [S3] Vietnam Consumer 2026 — gtjai.com.vn/...
   [S4] FPT Retail nhóm tốt nhất châu Á — vnexpress.net/...

💾 Saving to memory...
🧠 Facts: +5E +5R +5F
(89.2s)
```

### Mở rộng

Framework thiết kế để thêm tool dễ dàng. Ví dụ có sẵn: `student_tools.py` (REST API), `rag_tools.py` (vector search), `speech_tools.py` (STT/TTS). Xem chi tiết tại [Adding Your Own Tools](#adding-your-own-tools) và kiến trúc hệ thống tại [ARCHITECTURE.md](ARCHITECTURE.md).

---

## License

MIT

## Contributing

PRs welcome. Please open an issue first to discuss what you would like to change.
