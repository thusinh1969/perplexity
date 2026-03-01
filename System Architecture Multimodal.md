# System Architecture — Full Flow

## 1. Entry Points

```
┌──────────────────────────────────────────────────────────┐
│                    2 ENTRY POINTS                         │
│                                                           │
│  Console (run_chat.py)           FastAPI (server.py)      │
│  ├─ Interactive REPL             ├─ POST /chat            │
│  ├─ Single query --query         ├─ POST /chat/stream     │
│  ├─ --image flag                 ├─ POST /chat/json       │
│  └─ img:path.jpg inline          └─ multipart upload      │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Console Flow (run_chat.py)

### 2.1 Startup

```bash
python -m school_agents.run_chat --session mytest --interactive --stream
```

```
main()
  │
  ├─ load_config("config/")
  │   ├─ llm.yaml      → model, base_url, temperature
  │   ├─ agents.yaml    → 5 agent definitions
  │   ├─ tasks.yaml     → 5 task definitions
  │   ├─ tools.yaml     → Tavily API keys, RAG config
  │   └─ memory.yaml    → compressor, expand, facts settings
  │
  ├─ bootstrap(cfg)     → set_tool_config() for @tool functions
  │
  ├─ _init_memory(cfg, session_id)
  │   ├─ MemoryDB("data/memory/sessions.jsonl")  ← load all sessions
  │   ├─ ContextCompressor(strategy="llm_summary")
  │   └─ ConversationMemory(session_id, bank, compressor)
  │       └─ FactStore.load() → entities, relations, facts from disk
  │
  └─ Print banner → enter REPL loop
```

### 2.2 User Input Parsing (REPL)

```
You: img:thuoc.jpg img:label.png Thuốc gì đây?
     ─────┬─────  ─────┬──────  ──────┬──────
          │            │              │
     _encode_image()   │         text query
     → {b64, mime}     │
                  _encode_image()
                  → {b64, mime}

Result:
  query = "Thuốc gì đây?"
  turn_images = [
      {"b64": "/9j/4AAQ...", "mime": "image/jpeg"},
      {"b64": "iVBORw0K...", "mime": "image/png"},
  ]
```

**CLI mode:**
```bash
python -m school_agents.run_chat \
    -q "So sánh 2 toa này" \
    --image images/toa1.jpg images/toa2.jpg \
    --stream
```

### 2.3 Turn Execution (_run_one_turn)

```
_run_one_turn(cfg, memory, query, images=[...])
  │
  │  ①  RECORD
  ├─── memory.add_user_turn("Thuốc gì đây? [📎 2 image(s) attached]")
  │    (base64 NOT stored in memory — only text note)
  │
  │  ②  BUILD CONTEXT
  ├─── memory.build_context(query)
  │    │
  │    ├─ _needs_compression()?
  │    │   unsummarized turns > 4 AND tokens > max_context_tokens?
  │    │   YES → _compress(): LLM summarizes old turns
  │    │
  │    └─ Return 3-tier context:
  │       ┌─────────────────────────────────────────────┐
  │       │ [Known facts]                    ← Tier 1   │
  │       │ - FPT_Retail (org): revenue 68% Long Châu   │
  │       │ - Long_Chau (org): pharmacy chain            │
  │       │                                              │
  │       │ [Conversation history summary]   ← Tier 3   │
  │       │ User hỏi về FPT Retail...                    │
  │       │                                              │
  │       │ [Recent conversation]            ← Tier 2   │
  │       │ User: ...                                    │
  │       │ Assistant: ...                               │
  │       │ User: ...                                    │
  │       │ Assistant: ...                               │
  │       └─────────────────────────────────────────────┘
  │
  │  ③  ROUTE
  ├─── route(cfg, query, conversation_context=context)
  │    │
  │    │  CrewAI runs router agent with get_datetime tool:
  │    │  ┌─────────────────────────────────────────────┐
  │    │  │ Router sees: "Thuốc gì đây? [📎 2 images]" │
  │    │  │                                              │
  │    │  │ STEP 1: Call get_datetime → "01/03/2026"     │
  │    │  │ STEP 2: Analyze query                        │
  │    │  │   - Has images → need VLM to see them        │
  │    │  │   - Could be a web query (identify medicine) │
  │    │  │   - NOT a conversation/meta query             │
  │    │  │ STEP 3: Output JSON                          │
  │    │  │   {"routes": ["web"], "student_id": null,    │
  │    │  │    "policy_domain": "other"}                  │
  │    │  └─────────────────────────────────────────────┘
  │    │
  │    └─ Python safety net:
  │       routes=["web"] + query not trivial + not meta → keep ["web"]
  │
  │  ④  QUERY EXPANSION (only if "web" in routes)
  ├─── _handle_query_expansion(cfg, query, interactive=True)
  │    │
  │    ├─ LLM generates 3 queries:
  │    │   1. "identify medicine tablet image Vietnamese"
  │    │   2. "nhận dạng thuốc qua hình ảnh"
  │    │   3. "medicine pill identification by shape color"
  │    │
  │    ├─ Interactive confirm: "Search all? [Y/n/edit]"
  │    │   (auto mode for API — no confirmation)
  │    │
  │    ├─ Tavily search × 3 queries × 5 results each
  │    │
  │    └─ RRF merge: 15 raw → dedup → ~10-15 merged results
  │       → formatted as "[Pre-searched results]" text
  │
  │  ⑤  RUN CREW
  └─── run_crew_with_memory(cfg, routes, inputs, memory, images=[...])
       │
       │  (see Section 4 below)
       │
       └─ Returns: clean answer string (no "Final Answer:" prefix)
```

---

## 3. FastAPI Flow (server.py)

### 3.1 Startup

```
uvicorn school_agents.server:app --host 0.0.0.0 --port 8000
  │
  ├─ load_config()
  ├─ bootstrap()
  ├─ MemoryDB()         ← shared across all requests
  ├─ ContextCompressor() ← shared, stateless
  └─ make_openai_client() ← shared LLM client
```

### 3.2 Endpoints

```
POST /chat              ← multipart: text + image files
POST /chat/stream       ← multipart: text + image files → SSE
POST /chat/json         ← JSON body: text + base64 images
GET  /sessions/{id}     ← session stats (turns, facts, summary)
DELETE /sessions/{id}   ← clear session
GET  /health            ← health check
```

### 3.3 Request Flow (/chat with images)

```
Client:
  curl -X POST http://localhost:8000/chat \
       -F 'query=Thuốc gì đây?' \
       -F 'session_id=abc123' \
       -F 'images=@thuoc.jpg' \
       -F 'images=@label.png'

Server:
  chat()
    │
    ├─ _encode_uploads([UploadFile, UploadFile])
    │   → [{"b64": "...", "mime": "image/jpeg"},
    │      {"b64": "...", "mime": "image/png"}]
    │
    ├─ loop.run_in_executor(None, _execute_turn)
    │   │                    ↑ runs in thread (non-blocking)
    │   │
    │   └─ _execute_turn(session_id, query, images, ...)
    │      │
    │      ├─ memory.add_user_turn(...)
    │      ├─ route(...)
    │      ├─ _expand_and_search(query)    ← auto mode, no confirm
    │      └─ run_crew_with_memory(...)    ← same as console
    │
    └─ Return JSON:
       {
         "answer": "Đây là thuốc Paracetamol 500mg...",
         "session_id": "abc123",
         "routes": ["web"],
         "elapsed_seconds": 12.3,
         "turn_count": 5
       }
```

### 3.4 SSE Streaming Flow (/chat/stream)

```
Client:
  curl -N -X POST http://localhost:8000/chat/stream \
       -F 'query=Tin tức Việt Nam' -F 'session_id=abc123'

Server:
  event_gen() (async generator)
    │
    ├─ asyncio.Queue connects thread → async
    │
    ├─ Thread: _execute_turn(stream_callback=_cb)
    │   │
    │   │  CrewAI streams chunks:
    │   │    chunk.content = "Theo" → queue.put("chunk", {...})
    │   │    chunk.content = " các" → queue.put("chunk", {...})
    │   │    chunk.content = " nguồn" → queue.put("chunk", {...})
    │   │    ...
    │   │    _execute_turn returns → queue.put("done", {answer, routes, ...})
    │   │    queue.put(None)  ← sentinel
    │   │
    │
    └─ Async loop: queue.get() → yield SSE events
       event: chunk    data: {"content": "Theo", "agent": "composer"}
       event: chunk    data: {"content": " các", "agent": "composer"}
       event: chunk    data: {"content": " nguồn", "agent": "composer"}
       ...
       event: done     data: {"answer": "...", "session_id": "abc123", ...}
```

---

## 4. CrewAI Execution (crew_runner.py)

### 4.1 run_crew_with_memory — the core

```
run_crew_with_memory(cfg, routes, inputs, memory, images=[...])
  │
  │  ① INJECT CONTEXT + DATETIME
  ├── enriched = {
  │     "user_query": "Thuốc gì đây?\n\n[Pre-searched results]\n...",
  │     "conversation_context": "Current date: 01/03/2026\n\n[Known facts]\n...",
  │     "current_datetime": "Saturday, 01/03/2026 14:30:00",
  │     "student_id": null,
  │     ...
  │   }
  │
  │  ② SET UP VISION (if images)
  ├── set_images([{b64, mime}, ...])        ← thread-local storage
  ├── _ensure_vision_patch()                ← one-time litellm monkey-patch
  │
  │  ③ BUILD CREW
  ├── _build_crew(cfg, routes=["web"], stream=True)
  │   │
  │   │  routes=["web"] → tasks = [web_task, compose_task]
  │   │  routes=[]      → tasks = [compose_task]  (conversation-only)
  │   │  routes=["web","student"] → tasks = [web_task, student_task, compose_task]
  │   │
  │   └─ Crew(agents=[...], tasks=[...], process=sequential)
  │
  │  ④ KICKOFF
  ├── crew.kickoff(inputs=enriched)
  │   │
  │   │  CrewAI loop per task:
  │   │
  │   │  ┌─ web_task (web_researcher agent) ──────────────────────┐
  │   │  │  1. Agent reads task description (with {conversation_   │
  │   │  │     context} and {user_query} substituted)              │
  │   │  │  2. Checks [Pre-searched results] — sufficient?         │
  │   │  │  3. If not: calls web_search_deep or web_search_expanded│
  │   │  │  4. Returns: Final Answer: {"findings":[...],           │
  │   │  │     "sources":[...], "as_of":"2026-03-01"}              │
  │   │  └────────────────────────────────────────────────────────┘
  │   │          ↓ output passed to next task
  │   │  ┌─ compose_task (composer agent) ────────────────────────┐
  │   │  │  1. Receives web_result as evidence                     │
  │   │  │  2. Reads conversation_context for continuity           │
  │   │  │  3. Writes Perplexity-style Vietnamese answer           │
  │   │  │  4. Returns: Final Answer: <prose with [S1] citations>  │
  │   │  └────────────────────────────────────────────────────────┘
  │   │
  │   │  MEANWHILE (vision):
  │   │  Every litellm.completion() call goes through our patch:
  │   │  ┌──────────────────────────────────────────────────────┐
  │   │  │  _vision_completion(*args, **kwargs)                  │
  │   │  │    images = get_images()                              │
  │   │  │    if images:                                         │
  │   │  │      find last user message in kwargs["messages"]     │
  │   │  │      transform: "text string"                         │
  │   │  │        → [                                            │
  │   │  │            {"type": "text", "text": "..."},           │
  │   │  │            {"type": "image_url", "image_url":         │
  │   │  │              {"url": "data:image/jpeg;base64,..."}},   │
  │   │  │            {"type": "image_url", "image_url":         │
  │   │  │              {"url": "data:image/png;base64,..."}},    │
  │   │  │          ]                                            │
  │   │  │    return original_litellm_completion(...)             │
  │   │  └──────────────────────────────────────────────────────┘
  │   │
  │   │  MEANWHILE (streaming):
  │   │  Each chunk → stream_callback(chunk) → user sees tokens
  │   │
  │
  │  ⑤ CLEAR VISION
  ├── clear_images()
  │
  │  ⑥ POST-PROCESS
  ├── strip_think_tags(raw_answer)     ← remove <think>...</think>
  ├── strip "Final Answer:" prefix     ← user doesn't see this
  │
  │  ⑦ SAVE TO MEMORY
  ├── memory.add_assistant_turn(clean_answer)
  ├── memory.extract_facts(llm_client)  ← LLM extracts entities/relations
  ├── memory.save()
  │
  └── return clean_answer
```

### 4.2 Image injection detail

```
WITHOUT images (text-only, current behavior):
  litellm.completion(messages=[
      {"role": "system", "content": "You are..."},
      {"role": "user",   "content": "Tin tức hôm nay?"},
  ])

WITH images (after vision patch):
  litellm.completion(messages=[
      {"role": "system", "content": "You are..."},
      {"role": "user",   "content": [
          {"type": "text",      "text": "Thuốc gì đây?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO..."}},
      ]},
  ])

This is standard OpenAI Vision / ChatML multimodal format.
vLLM and Ollama both support it for VLM models (Qwen2.5-VL, InternVL, etc).
```

---

## 5. Memory System

```
                          ┌─────────────────────────┐
                          │   sessions.jsonl (disk)  │
                          │   (ALL turns, forever)   │
                          └──────────┬──────────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │     MemoryDB (RAM)       │
                          │  session:abc123 → {      │
                          │    turns: [T0..T25],     │
                          │    summary: "...",       │
                          │    summary_covers: 19    │
                          │  }                       │
                          └──────────┬──────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     │               │               │
              ┌──────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐
              │  Tier 1     │  │  Tier 3     │  │  Tier 2  │
              │  FACTS      │  │  SUMMARY    │  │  RECENT  │
              │             │  │             │  │  TURNS   │
              │ 22 entities │  │ LLM-compressed│ │ Last 4  │
              │ 42 relations│  │ turns 0-19  │  │ verbatim │
              │ 39 facts    │  │ 374→~1500   │  │ (size-   │
              │             │  │ chars       │  │  guarded)│
              │ NEVER       │  │             │  │          │
              │ compressed  │  │ Re-compressed│ │ Truncated│
              │ NEVER       │  │ when new    │  │ if >     │
              │ deleted     │  │ turns added │  │ budget   │
              └─────────────┘  └─────────────┘  └──────────┘
                     │               │               │
                     └───────────────┼───────────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │    build_context()       │
                          │    → inject into CrewAI  │
                          │    task descriptions     │
                          └─────────────────────────┘
```

---

## 6. File Map

```
src/school_agents/
├── config/
│   ├── llm.yaml           # Model, base_url, temperature, max_tokens
│   ├── agents.yaml         # 5 agents: router, web, student, policy, composer
│   ├── tasks.yaml          # 5 tasks: route, web, student, policy, compose
│   ├── tools.yaml          # Tavily, RAG, student API configs
│   └── memory.yaml         # Compressor, expansion, facts settings
│
├── server.py               # FastAPI: /chat, /chat/stream, /chat/json ← NEW
├── run_chat.py             # Console: REPL + single query + img: support
├── run.py                  # Legacy single-shot (no memory)
│
├── crew_runner.py          # CrewAI orchestration + vision patch
├── image_context.py        # Thread-local image storage ← NEW
│
├── config.py               # YAML loader + AppConfig dataclass
├── tool_context.py         # Thread-local tool config
├── llm_utils.py            # strip_think_tags, no_think_extra_body
│
├── conversation_memory.py  # 3-tier memory: facts + summary + recent
├── context_compressor.py   # LLM summary / LLMLingua compression
├── fact_store.py           # Entity/relation/fact extraction
├── memory_bank.py          # JSONL persistence backend
├── query_expander.py       # Multi-query expansion + RRF merge
│
├── tools/
│   ├── web_tools.py        # Tavily search/crawl + expansion
│   ├── student_tools.py    # Student API tools
│   ├── rag_tools.py        # Policy RAG tools
│   └── datetime_tools.py   # get_datetime tool
│
├── data/
│   └── memory/
│       └── sessions.jsonl  # Persistent memory store
│
└── images/                 # Steve's image folder for testing
```

---

## 7. Quick Reference — How to Use

### Console

```bash
# Text only
python -m school_agents.run_chat -s mytest -i --stream

# With image (single query)
python -m school_agents.run_chat -q "Thuốc gì?" --image images/thuoc.jpg --stream

# REPL with images
You: img:images/thuoc.jpg Thuốc gì đây?
You: img:images/a.jpg img:images/b.png So sánh 2 ảnh này
You: Tóm tắt cuộc trò chuyện         ← no image, conversation-only
```

### FastAPI

```bash
# Start server
uvicorn school_agents.server:app --host 0.0.0.0 --port 8000

# Text query
curl -X POST http://localhost:8000/chat \
     -F 'query=Tin tức hôm nay' -F 'session_id=mytest'

# With images
curl -X POST http://localhost:8000/chat \
     -F 'query=Thuốc gì?' -F 'session_id=mytest' \
     -F 'images=@images/thuoc.jpg' -F 'images=@images/label.png'

# JSON body (base64 images — for mobile apps)
curl -X POST http://localhost:8000/chat/json \
     -H 'Content-Type: application/json' \
     -d '{"query":"Thuốc gì?","session_id":"mytest","images":[{"b64":"...","mime":"image/jpeg"}]}'

# SSE Streaming
curl -N -X POST http://localhost:8000/chat/stream \
     -F 'query=Thời tiết Hà Nội' -F 'session_id=mytest'

# Session management
curl http://localhost:8000/sessions/mytest
curl -X DELETE http://localhost:8000/sessions/mytest
```
