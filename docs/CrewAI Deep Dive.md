# CrewAI Deep Dive: Cách hoạt động qua bộ code school_agents

## 1. KHÁI NIỆM CỐT LÕI

### CrewAI là gì?

CrewAI là một **orchestration framework** — nó không phải AI, mà là "người quản lý" điều phối nhiều AI agents làm việc cùng nhau. Giống như một đạo diễn phim: đạo diễn không đóng vai, mà chỉ đạo từng diễn viên đóng đúng cảnh của mình.

### 4 khái niệm cốt lõi

```
┌──────────────────────────────────────────────────────────────────┐
│                        CREWAI CONCEPTS                           │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │  AGENT  │    │  TASK   │    │  TOOL   │    │  CREW   │      │
│  │         │    │         │    │         │    │         │      │
│  │ "Ai?"   │    │ "Làm    │    │ "Dùng   │    │ "Đội    │      │
│  │ Con     │    │  gì?"   │    │  gì để  │    │  hình   │      │
│  │ người   │    │ Nhiệm   │    │  làm?"  │    │  nào?"  │      │
│  │ nào?    │    │ vụ cụ   │    │ API,    │    │ Ai +    │      │
│  │ Role +  │    │ thể     │    │ Search, │    │ Task    │      │
│  │ Goal    │    │         │    │ DB...   │    │ nào?    │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│                                                                  │
│  agents.yaml    tasks.yaml     tools/*.py     crew_runner.py    │
└──────────────────────────────────────────────────────────────────┘
```

Tương tự thế giới thực:
- **Agent** = Nhân viên (có chức danh, mục tiêu, tính cách)
- **Task** = Phiếu giao việc (mô tả việc cần làm, kết quả mong đợi)
- **Tool** = Dụng cụ (máy tính, điện thoại, sách tra cứu...)
- **Crew** = Đội nhóm (gom nhân viên + phiếu việc lại, rồi chạy)

---

## 2. CẤU TRÚC THƯ MỤC — MỖI FILE LÀM GÌ?

```
src/school_agents/
│
├── config/                      ◄── "HỒ SƠ CÔNG TY" (YAML)
│   ├── llm.yaml                 Bộ não AI (model, server, token budgets)
│   ├── agents.yaml              Hồ sơ nhân viên (smart prompts chi tiết)
│   ├── tasks.yaml               Bảng mô tả công việc (Final Answer format)
│   ├── tools.yaml               API keys dùng ${ENV_VAR} (resolve từ .env)
│   └── memory.yaml              Cấu hình memory, compression, query expansion
│
├── tools/                       ◄── "DỤNG CỤ" (Python functions)
│   ├── web_tools.py             Tìm kiếm web (Tavily) + query expansion + crawl
│   ├── student_tools.py         Tra cứu dữ liệu học sinh (REST API)
│   ├── rag_tools.py             Tìm kiếm tài liệu nội bộ (vector + BM25)
│   ├── datetime_tools.py        Ngày giờ tiếng Việt (get_datetime, date_add_delta)
│   └── speech_tools.py          Chuyển giọng nói ↔ text (optional)
│
├── config.py                    ◄── Đọc YAML → Python objects + resolve ${ENV_VAR}
├── tool_context.py              ◄── Chia sẻ config cho tools
├── crew_runner.py               ◄── "ĐẠO DIỄN" — tạo agents, inject datetime, chạy
│
├── run_chat.py                  ◄── Interactive CLI (memory + streaming + expansion)
├── run.py                       ◄── Simple CLI (single-shot, no memory)
├── server.py                    ◄── FastAPI + SSE streaming server
│
├── conversation_memory.py       ◄── Quản lý turns + tích hợp facts
├── memory_bank.py               ◄── Lưu trữ session vĩnh viễn (JSONL)
├── fact_store.py                ◄── Trích xuất entity/relation/fact từ hội thoại
├── context_compressor.py        ◄── Nén history cũ (LLM summary / LLMLingua)
├── query_expander.py            ◄── Mở rộng 1 câu hỏi → 3 search queries
└── llm_utils.py                 ◄── Xử lý thinking models (<think> tags)
```

---

## 3. YAML FILES — "HỒ SƠ" VIẾT BẰNG TIẾNG NGƯỜI

### 3.1 llm.yaml — Bộ não AI

```yaml
llm:
  model: "qwen/qwen3-coder-next"           # Model AI nào?
  base_url: "http://10.211.55.2:1234/v1"   # Server nào chạy model?
  api_key: "${LLM_API_KEY}"                 # ← Resolve từ .env, KHÔNG hardcode!
  temperature: 1.0                          # 0=máy móc, 1=sáng tạo
  max_tokens: 8192                          # Giới hạn output cho CrewAI agents
  structured_max_tokens: 16384              # Cho fact extraction, query expansion
                                            # Think models: 16384, Non-think: 4096
```

File này trả lời: **"Bộ não AI ở đâu, cấu hình thế nào, token budget bao nhiêu?"**

**Quan trọng**: `structured_max_tokens` — một chỗ duy nhất chỉnh token limit cho TẤT CẢ structured output calls (fact extraction, query expansion, context compression). Think models cần 16K vì chúng "suy nghĩ" 3-5K tokens trước khi output JSON.

### 3.2 agents.yaml — Smart Prompts (Hồ sơ nhân viên chi tiết)

```yaml
web_researcher:
  role: "Web Researcher"
  goal: "Search the web for information, then summarize findings with citations."
  backstory: >
    You are an expert web researcher. Your job is to find the FRESHEST, most
    relevant information from the internet for any query.

    CRITICAL WORKFLOW:
    1) Read any [Pre-searched results] provided in the task.
    2) Evaluate: recent enough? sufficient depth? If NO → search again.
    3) Call web_search_deep or web_search_expanded ONCE.
    4) Combine ALL evidence → write Final Answer as JSON.

    IMPORTANT RULES:
    - ALWAYS prefer FRESH information over stale pre-searched results.
    - NEVER call the same tool twice with the same query.
    - After receiving tool results, IMMEDIATELY write your Final Answer.
  allow_delegation: false
```

**Khác biệt so với v1**: Backstory giờ rất CHI TIẾT — không chỉ nói "bạn là ai" mà hướng dẫn STEP-BY-STEP cụ thể. LLM hiểu rõ hơn, ít loop vô hạn.

Mỗi trường sẽ được CrewAI nhét vào **system prompt** gửi cho LLM:

```
         agents.yaml                        System Prompt gửi cho LLM
  ┌─────────────────────────┐      ┌──────────────────────────────────────┐
  │ [SYSTEM TIME] Hôm nay   │ ──►  │ [SYSTEM TIME] Hôm nay là Thứ Bảy,  │
  │   là ...  (injected!)   │      │ ngày 01/03/2026, giờ 11:30:45...    │
  │                         │      │                                      │
  │ role: "Web Researcher"  │ ──►  │ You are Web Researcher.              │
  │ goal: "Search..."       │ ──►  │ Your goal is: Search the web...      │
  │ backstory: "You are an  │ ──►  │ You are an expert web researcher.    │
  │   expert..."            │      │ CRITICAL WORKFLOW: 1) Read...        │
  └─────────────────────────┘      └──────────────────────────────────────┘
```

### 3.3 tasks.yaml — Phiếu giao việc (Final Answer format)

```yaml
web_task:
  description: |
    {conversation_context}

    === YOUR TASK ===
    Research this question using web search:
    {user_query}

    === WORKFLOW ===
    STEP 1: Check [Pre-searched results] if provided.
    STEP 2: Evaluate freshness and depth. Search again if needed.
    STEP 3: Call ONE search tool with well-crafted query.
    STEP 4: Combine ALL evidence → Final Answer.

    Final Answer: {"findings": [...], "sources": [{"url":"...", "title":"..."}]}

    You MUST prefix with "Final Answer:" or your response will be rejected.
  expected_output: "Final Answer: JSON with findings and sources."
  agent: web_researcher
```

**Bắt buộc**: `"Final Answer:"` prefix — CrewAI ReAct parser CHỈ accept response có prefix này. Không có → CrewAI reject → agent retry → double tokens.

`{user_query}`, `{conversation_context}` là template variables — khi chạy, CrewAI thay bằng giá trị thật từ `inputs={}`.

### 3.4 tools.yaml — Config cho dụng cụ (secrets qua .env)

```yaml
web:
  tavily:
    api_key: "${TAVILY_API_KEY}"     # ← Resolve từ .env, KHÔNG hardcode!
    base_url: "https://api.tavily.com"
    search_depth: "advanced"
    max_results: 5
```

### 3.5 .env — Secrets thật (KHÔNG BAO GIỜ commit lên Git!)

```env
LLM_API_KEY=sk-lm-ckOnzOQj:J4Sj1siIHyt4NrTIjiwr
TAVILY_API_KEY=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
STUDENT_API_KEY=your-key-here
RAG_API_KEY=your-key-here
```

`.gitignore` chặn `.env`. Chỉ có `.env.example` (template) được commit.

---

## 4. CONFIG.PY — TRUNG TÂM LOAD + RESOLVE ${ENV_VAR}

```python
# config.py làm 3 việc:

# 1. Đọc .env vào os.environ (KHÔNG cần python-dotenv)
_load_dotenv(project_root)

# 2. Load YAML files
llm_raw = load_yaml(root / "llm.yaml")["llm"]
tools = load_yaml(root / "tools.yaml")

# 3. Resolve ${VAR} patterns — đệ quy cho nested dicts/lists
llm_raw = _resolve_env(llm_raw)
# "${LLM_API_KEY}" → "sk-lm-ckOnzOQj:J4Sj1siIHyt4NrTIjiwr"
```

Flow resolve:

```
.env file                    YAML file                   Python object
┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────────────┐
│ LLM_API_KEY=     │   │ api_key: "${LLM_    │   │ LLMConfig(               │
│   sk-lm-xxx     │ + │   API_KEY}"          │ → │   api_key="sk-lm-xxx",   │
│ TAVILY_API_KEY=  │   │ max_tokens: 8192    │   │   max_tokens=8192,       │
│   tvly-xxx      │   │ structured_max_      │   │   structured_max_tokens= │
│                  │   │   tokens: 16384     │   │     16384                │
└──────────────────┘   └─────────────────────┘   └──────────────────────────┘
```

Nếu env var chưa set → warning log + empty string (không crash).

---

## 5. PYTHON CODE — AI "LÀM VIỆC" NHƯ THẾ NÀO?

### 5.1 Tools — Hàm Python mà AI có thể gọi

```python
# tools/web_tools.py

@tool("web_search_deep")                          # ← Decorator biến hàm thành "tool"
def web_search_deep(
    query: str,                                    # ← LLM sẽ điền giá trị này
    max_results: int = 3,
    search_depth: str = "basic",
    include_raw_content: bool = False,
) -> str:
    """Tavily Search via official SDK. Returns JSON string with web search results."""
    #  ↑ Docstring này = mô tả tool gửi cho LLM
    result = _search_deep(query=query, ...)
    return json.dumps(result)
```

CrewAI đọc `@tool` decorator và tự động tạo **OpenAI function calling schema**:

```
     Python @tool                           JSON Schema gửi cho LLM
┌──────────────────────┐              ┌─────────────────────────────────┐
│ def web_search_deep( │              │ {                               │
│   query: str,        │  ──────────► │   "name": "web_search_deep",   │
│   max_results:       │              │   "description": "Tavily...",   │
│     int = 3          │              │   "parameters": {              │
│ ) -> str:            │              │     "query": {"type":"string"},│
│   """Tavily."""      │              │     "max_results": {"type":    │
└──────────────────────┘              │       "integer","default":3}   │
                                      │   }                            │
                                      │ }                               │
                                      └─────────────────────────────────┘
```

**Quan trọng**: LLM thấy default values từ Python signature (`= 3`), KHÔNG phải từ tools.yaml.

### 5.2 crew_runner.py — "Đạo diễn" với 3 tầng Datetime Injection

```python
def _make_agents(cfg, llm):
    """Tạo nhân viên — MỖI NGƯỜI đều nhận datetime trong backstory."""

    # ── Tầng 1: Inject datetime vào TẤT CẢ agent backstories ──
    _now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    _weekdays_vi = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]
    _datetime_line = (
        f"[SYSTEM TIME] Hôm nay là {_weekdays_vi[_now.weekday()]}, "
        f"ngày {_now:%d/%m/%Y}, giờ {_now:%H:%M:%S} (Asia/Ho_Chi_Minh). "
        f"English: {_now:%A, %B %d, %Y at %H:%M:%S %Z}."
    )

    def _inject_datetime(agent_cfg):
        cfg_copy = dict(agent_cfg)
        cfg_copy["backstory"] = f"{_datetime_line}\n\n{cfg_copy['backstory']}"
        return cfg_copy

    # Router: datetime tools + verbose=False (output không hiện)
    router = Agent(llm=llm, verbose=False,
                   tools=[get_datetime, date_add_delta_days],
                   **_inject_datetime(a["router"]))

    # Web researcher: 4 web tools, max_iter=3
    web = Agent(llm=llm, verbose=True,
                tools=[web_search_deep, web_search_expanded,
                       web_crawl_url, web_search_then_crawl],
                max_iter=3,
                **_inject_datetime(a["web_researcher"]))

    # Composer: KHÔNG có tools, max_iter=1 (one-shot, tránh double output)
    composer = Agent(llm=llm, verbose=True, max_iter=1,
                     **_inject_datetime(a["composer"]))

    return {"router": router, "web_researcher": web, ..., "composer": composer}
```

**3 tầng datetime**:
1. **Agent backstory** — `_inject_datetime()` prepend vào mọi agent (code trên)
2. **Query expansion** — `query_expander.py` inject `"Today is March 1, 2026"` vào expand prompt
3. **Task context** — `run_crew_with_memory()` thêm `Current date/time: ...` vào `{conversation_context}`

### 5.3 Run Crew — Routing → Build → Kickoff

```python
def _build_crew(cfg, routes, stream=False):
    """Ghép đội theo routes. Composer LUÔN ở cuối."""
    agents = _make_agents(cfg, llm)

    # Khi streaming: tắt verbose TẤT CẢ agents (tránh double output)
    if stream:
        for agent in agents.values():
            agent.verbose = False

    tasks = []
    if "web" in routes:       tasks.append(_make_task("web_task"))
    if "student" in routes:   tasks.append(_make_task("student_task"))
    if "policy_rag" in routes: tasks.append(_make_task("policy_task"))
    tasks.append(_make_task("compose_task"))   # LUÔN cuối cùng

    return Crew(agents=list(agents.values()), tasks=tasks, stream=stream)
```

---

## 6. RUNTIME — CHUYỆN GÌ XẢY RA TỪ A TỚI Z

### 6.1 Full Flow (Interactive CLI với Memory + Streaming)

```
 Terminal: python -m school_agents.run_chat --session 123 --interactive --stream
    │
    ▼
 run_chat.py
    │  1. load_config("config/") → đọc 5 YAML + resolve ${ENV_VAR} từ .env
    │  2. bootstrap(cfg) → chia sẻ tools.yaml cho tools
    │  3. Khởi tạo ConversationMemory (load history + facts từ disk)
    │
    │  ════════════════ USER INPUT ════════════════
    │
    │  You: "Tình hình chiến tranh Mỹ Iran? Dự đoán 72h tới?"
    │
    │  ════════════════ PHASE 0: QUERY EXPANSION ════════════════
    │
    │  4. query_expander.py → gửi user query + "Today is March 1, 2026" cho LLM
    │     ⏳ Generating search queries.......... done!
    │     LLM trả JSON array:
    │     [
    │       "US Iran war latest military updates March 2026",      ← English
    │       "tình hình chiến tranh Mỹ Iran mới nhất",              ← Vietnamese
    │       "US Iran conflict expert analysis escalation forecast"  ← Analytical
    │     ]
    │
    │  5. 🔍 Query expansion (3 queries): [hiện ra terminal]
    │     Search all? [Y/n/edit]
    │     Tavily search × 3 → merge + deduplicate → 5 kết quả
    │     ✅ 5 results merged.
    │
    │  ════════════════ PHASE 1: ROUTING ════════════════
    │
    │  6. route(cfg, query) →
    │     Router agent nhận: [SYSTEM TIME] datetime + context + user query
    │     Router trả: {"routes": ["web"], "policy_domain": "other"}
    │
    │     (Safety: nếu routes=[] cho query dài → force ["web"])
    │
    │  ════════════════ PHASE 2: CREW EXECUTION ════════════════
    │
    │  7. run_crew_with_memory()
    │     ├── memory.build_context() → nén nếu cần, trả context string
    │     ├── Inject context + datetime + pre-searched results vào inputs
    │     ├── _build_crew(routes=["web"]) → [web_task, compose_task]
    │     └── crew.kickoff(inputs={...})
    │
    │     ┌──── TASK 1: web_task (Web Researcher) ──────────────────────┐
    │     │                                                              │
    │     │  Nhận: conversation context + user query                    │
    │     │        + [Pre-searched results] (5 kết quả Tavily)          │
    │     │                                                              │
    │     │  HTTP #1: prompt + tools → LLM (Qwen3)                     │
    │     │  LLM <think>: "results đã có sẵn, đánh giá freshness..."   │
    │     │  LLM trả: tool_call web_search_deep("US Iran attack...")    │
    │     │           (hoặc không gọi tool nếu pre-searched đủ)        │
    │     │                                                              │
    │     │  CrewAI chạy Python: web_search_deep() → Tavily API        │
    │     │  → truncate kết quả nếu > 8000 chars                       │
    │     │                                                              │
    │     │  HTTP #2: tool result → LLM                                 │
    │     │  LLM trả: "Final Answer: {findings: [...], sources: [...]}" │
    │     │                                                              │
    │     │  → output = JSON string                                     │
    │     └───────────────────────┬──────────────────────────────────────┘
    │                             │
    │       output task 1 tự động truyền vào context task 2
    │                             │
    │     ┌──── TASK 2: compose_task (Answer Composer) ─────────────────┐
    │     │                                                              │
    │     │  Nhận: context + query + ALL evidence from task 1           │
    │     │  + datetime ("Hôm nay 01/03/2026, 72h = 01-04/03/2026")   │
    │     │                                                              │
    │     │  🌊 STREAMING: CHỈ output composer hiện ra terminal!        │
    │     │     (web_researcher output = filtered, user không thấy)     │
    │     │                                                              │
    │     │  HTTP #3: task desc + evidence → LLM                        │
    │     │  LLM trả: "Final Answer: Câu trả lời Việt + Nguồn..."     │
    │     │                                                              │
    │     └───────────────────────┬──────────────────────────────────────┘
    │                             │
    │  ════════════════ PHASE 3: POST-PROCESSING ════════════════
    │
    │  8. Strip "Final Answer:" prefix (chỉ cho CrewAI parser, user không cần thấy)
    │  9. Strip <think>...</think> tags (cho memory, display giữ nguyên)
    │ 10. 💾 Saving to memory... → add user turn + assistant turn
    │ 11. 🧠 Extracting facts... → LLM trích xuất entities/relations/facts
    │     → 🧠 Facts: +5E +3R +4F
    │ 12. memory.save() → persist to disk (JSONL)
    │     → ✅ Done (89.2s)
    │
    ▼
 "Theo các nguồn tin mới nhất, Mỹ và Israel đã tiến hành không kích..."
```

### 6.2 Chi tiết: CrewAI ghép prompt như thế nào?

Cho **web_researcher** làm **web_task**, CrewAI tạo ra prompt thế này:

```
┌─────────────────── SYSTEM MESSAGE ───────────────────────────────┐
│                                                                   │
│  [SYSTEM TIME] Hôm nay là Thứ Bảy,       ◄── crew_runner.py     │
│  ngày 01/03/2026, giờ 11:30:45               _inject_datetime() │
│  English: Saturday, March 01, 2026...                             │
│                                                                   │
│  You are Web Researcher.                  ◄── agents.yaml: role  │
│  You are an expert web researcher.        ◄── agents.yaml:       │
│  Your job is to find the FRESHEST...          backstory (smart)  │
│                                                                   │
│  CRITICAL WORKFLOW:                                               │
│  1) Read [Pre-searched results]...                                │
│  2) Evaluate freshness + depth...                                 │
│  3) Call search tool ONCE if needed...                            │
│  4) Combine ALL evidence...                                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────── USER MESSAGE ─────────────────────────────────┐
│                                                                   │
│  Current date/time: Saturday, 01/03/2026   ◄── run_crew_with_    │
│  [Previous conversation context]               memory() inject   │
│                                                                   │
│  === YOUR TASK ===                         ◄── tasks.yaml:       │
│  Research this question:                       description        │
│  Tình hình chiến tranh Mỹ Iran?                                  │
│                                                                   │
│  [Pre-searched results]                    ◄── query expansion   │
│  1. Live updates: US strikes Iran...           results injected  │
│  2. Iran tuyên bố 200 quân nhân Mỹ...                            │
│  3. ...                                                           │
│                                                                   │
│  You MUST prefix with "Final Answer:"      ◄── CrewAI format     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────── TOOLS (gửi kèm) ─────────────────────────────┐
│                                                                   │
│  tools: [                                  ◄── Từ @tool Python   │
│    {"name": "web_search_deep", ...},                              │
│    {"name": "web_search_expanded", ...},                          │
│    {"name": "web_crawl_url", ...},                                │
│    {"name": "web_search_then_crawl", ...}                         │
│  ]                                                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

Toàn bộ gửi qua HTTP POST tới LMStudio/vLLM.

### 6.3 Vòng lặp ReAct — Tool Calling (TRÁI TIM của CrewAI)

```
         CrewAI                      LLM (Qwen3)                   Tavily API
           │                             │                              │
           │  ── prompt + tools ──────►  │                              │
           │                             │  <think>                     │
           │                             │  Pre-searched đã có 5 kết   │
           │                             │  quả nhưng cần thêm chi     │
           │                             │  tiết về dự đoán...          │
           │                             │  </think>                    │
           │                             │                              │
           │  ◄── tool_call: ──────────  │                              │
           │      web_search_deep(       │                              │
           │        query="US Iran       │                              │
           │        conflict analysis    │                              │
           │        March 2026 forecast")│                              │
           │                             │                              │
           │  CrewAI thấy tool_call      │                              │
           │  → chạy hàm Python thật     │                              │
           │                             │                              │
           │  ── HTTP request ──────────────────────────────────────►   │
           │  ◄── JSON results (5 kết quả web) ────────────────────    │
           │                             │                              │
           │  ── gửi tool result ──────► │                              │
           │                             │  <think>                     │
           │                             │  đọc kết quả, tóm tắt...    │
           │                             │  </think>                    │
           │                             │                              │
           │  ◄── "Final Answer: ─────── │                              │
           │       {findings: [...],     │                              │
           │        sources: [...]}"     │                              │
           │                             │                              │
           │  ✅ Thấy "Final Answer"     │                              │
           │  → DỪNG vòng lặp           │                              │
           │  → chuyển sang task tiếp    │                              │
```

**CrewAI Decision Logic mỗi iteration:**

```
LLM output chứa "Final Answer:" → ✅ ACCEPT, dừng vòng lặp
LLM output là tool_call format   → ✅ EXECUTE tool, tiếp tục vòng lặp
LLM output là thứ khác bất kỳ    → ❌ REJECT, retry (LÃNG PHÍ tokens!)
Quá max_iter (3 cho researcher)  → ⛔ BUỘC DỪNG
```

### 6.4 Chuyền bóng giữa Tasks

CrewAI chạy **sequential** — output task trước = input context task sau:

```
 TASK 1 output:                         TASK 2 nhận được:
 ┌────────────────────────┐             ┌────────────────────────────┐
 │ Final Answer:          │             │ [system] [SYSTEM TIME]...  │
 │ {                      │   ──────►   │ You are Composer...        │
 │   "findings": [        │             │                            │
 │     "Mỹ không kích...",│             │ [user]                     │
 │     "Iran tuyên bố..." │             │ Current Task: Compose...   │
 │   ],                   │             │                            │
 │   "sources": [...]     │             │ CONTEXT từ task trước:     │
 │ }                      │             │ {"findings": [...],        │
 └────────────────────────┘             │  "sources": [...]}         │
                                        └────────────────────────────┘
```

Composer KHÔNG cần tools — chỉ đọc evidence và viết câu trả lời cho người dùng.

---

## 7. ROUTING — TẦNG QUYẾT ĐỊNH TRƯỚC KHI CHẠY CREW

```
                    User Query
                         │
                         ▼
              ┌─── route() ───────────────────────────────┐
              │                                            │
              │  Router agent nhận:                        │
              │  - [SYSTEM TIME] datetime (in backstory)  │
              │  - conversation_context (from memory)      │
              │  - user_query                              │
              │                                            │
              │  Router BIAS TOWARD WEB:                   │
              │  backstory says: "When in doubt, include   │
              │  web. Safe, never harmful."                │
              │                                            │
              │  Trả JSON:                                 │
              │  {"routes": ["web"],                       │
              │   "student_id": null,                      │
              │   "policy_domain": "other"}                │
              │                                            │
              │  Safety fallback:                          │
              │  - routes=[] + query > 5 chars → force web │
              │  - JSON parse fail → default to web        │
              │                                            │
              └────────────────┬───────────────────────────┘
                               │
                    Python code đọc routes
                               │
               ┌───────────────┼──────────────────┐
               │               │                  │
        "web" in routes? "student" in routes? "policy_rag"?
               │               │                  │
           Thêm             Thêm              Thêm
          web_task         student_task      policy_task
               │               │                  │
               └───────────────┼──────────────────┘
                               │
                        LUÔN thêm
                       compose_task (cuối cùng)
                               │
                               ▼
                     Crew([tasks...]).kickoff()
```

Routing cho phép hệ thống **chỉ chạy agent cần thiết** thay vì chạy hết.

---

## 8. QUERY EXPANSION — "1 CÂU HỎI → 3 SEARCH QUERIES"

Trước khi CrewAI chạy, `run_chat.py` gọi `query_expander.py` mở rộng câu hỏi:

```
User: "Tình hình chiến tranh Mỹ Iran thế nào?"
                    │
                    ▼
           query_expander.py
           (biết: "Today is Saturday, March 01, 2026")
                    │
     ┌──────────────┼──────────────┐
     │              │              │
  Query 1       Query 2        Query 3
  (English)     (Vietnamese)   (English analytical)
  "US Iran      "tình hình     "US Iran conflict
   war latest    chiến tranh    expert analysis
   updates       Mỹ Iran       escalation
   March 2026"   mới nhất"     forecast"
     │              │              │
     └──────────────┼──────────────┘
                    │
              Tavily search × 3
              Merge + deduplicate
                    │
              5 best results
                    │
          Inject vào web_task description
          dưới label [Pre-searched results]
```

**Tại sao cần expand?** Tavily chỉ trả 5 results/query. 3 queries × 5 results = 15 candidates → merge thành 5 best → coverage rộng hơn nhiều so với 1 query.

**Date-aware**: Expander tự thêm "March 2026", "latest" vào queries cho current events.

---

## 9. MEMORY SYSTEM — BỘ NHỚ XÂY THÊM (CrewAI KHÔNG CÓ)

```
┌────────────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                                  │
│                                                                     │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ conversation │  │  fact_store.py   │  │ context_compressor   │  │
│  │ _memory.py   │  │                 │  │ .py                  │  │
│  │              │  │  Trích xuất:    │  │                      │  │
│  │  Lưu turns:  │  │  - Entities    │  │  Nén history cũ:     │  │
│  │  user +      │  │  - Relations   │  │  - LLM summary       │  │
│  │  assistant   │  │  - Facts       │  │  - LLMLingua         │  │
│  │  (list)      │  │  từ mỗi lượt   │  │  - Hybrid            │  │
│  └──────┬───────┘  └───────┬─────────┘  └──────────┬───────────┘  │
│         │                  │                        │              │
│         └──────────────────┼────────────────────────┘              │
│                            │                                        │
│                     memory_bank.py                                  │
│                     (JSONL on disk — survives restarts)             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Flow mỗi lượt hội thoại:

```
User hỏi → add_user_turn() → build_context()
                                    │
                        ┌───────────┼───────────────┐
                        │           │               │
                    Summary     Recent turns     Facts
                    (nén từ     (max 4 turns     (entities +
                     turns cũ)   gần nhất)        relations)
                        │           │               │
                        └───────────┼───────────────┘
                                    │
                        conversation_context string
                                    │
                           Inject vào {conversation_context}
                           trong task descriptions
                                    │
                           CrewAI chạy → trả answer
                                    │
                        add_assistant_turn(answer)
                                    │
                        extract_facts() → LLM trích xuất
                           → "🧠 Facts: +5E +3R +4F"
                                    │
                        memory.save() → JSONL on disk
```

### Fact Extraction — Xử lý Thinking Models

Vấn đề: Qwen3/3.5 output `<think>reasoning here...</think>` TRƯỚC JSON → `json.loads()` fail.

```python
# llm_utils.py — 3 tầng bảo vệ

# Tầng 1: System prompt yêu cầu KHÔNG suy nghĩ
NO_THINK_SYSTEM = (
    "You are a precise JSON extractor. "
    "Output ONLY valid JSON. No thinking, no reasoning, no markdown."
)

# Tầng 2: Disable thinking qua model API (Qwen3 hỗ trợ)
def no_think_extra_body(extra_body=None):
    merged = dict(extra_body or {})
    merged["chat_template_kwargs"] = {"enable_thinking": False}
    return merged

# Tầng 3: Robust parser tìm JSON trong mọi output format
def extract_json(text):
    """Handles: <think>...</think>, Markdown fences, mixed text + JSON"""
    # 1. Strip <think> tags
    # 2. Find last JSON object {...}
    # 3. Find last JSON array [...]
    # 4. Return empty string if nothing found
```

Khi extract facts:
```python
resp = llm.chat.completions.create(
    messages=[
        {"role": "system", "content": NO_THINK_SYSTEM},     # Tầng 1
        {"role": "user", "content": fact_extraction_prompt},
    ],
    max_tokens=cfg.llm.structured_max_tokens,  # 16384 — từ llm.yaml
    extra_body=no_think_extra_body(),           # Tầng 2
)
raw = resp.choices[0].message.content
cleaned = extract_json(raw)   # Tầng 3 — luôn tìm được JSON
```

---

## 10. STREAMING — CHỈ HIỆN COMPOSER OUTPUT

Vấn đề: CrewAI stream TẤT CẢ agents. User chỉ muốn thấy câu trả lời cuối (composer).

```
                  CrewAI streaming chunks
                          │
            ┌─────────────┼─────────────┐
            │             │             │
       web_researcher  student_data  composer
       JSON findings   JSON data     Vietnamese answer
            │             │             │
            ▼             ▼             ▼
        FILTERED      FILTERED     ✅ SHOWN
        (user không   (user không  (stream ra terminal)
         thấy)        thấy)
```

Cách filter trong `crew_runner.py`:

```python
# run_crew_with_memory() — streaming logic

composer_text = []  # CHỈ capture output từ composer
all_text = []       # Tất cả (debug fallback)
_task_switches = 0
_total_tasks = len(routes_tasks) + 1  # +1 cho composer

for chunk in streaming:
    agent = getattr(chunk, "agent_role", None) or ""
    content = getattr(chunk, "content", None) or ""
    task = getattr(chunk, "task_name", None) or ""

    # Track task switches — composer LUÔN là task CUỐI CÙNG
    task_key = f"{agent}|{task}"
    if task_key != _current_task and (agent or task):
        _task_switches += 1
        if _task_switches >= _total_tasks:
            _is_last_task = True

    all_text.append(content)

    # Detect composer bằng TÊN hoặc VỊ TRÍ (vì agent_role thường = empty!)
    is_composer = ("composer" in str(agent).lower()) or _is_last_task
    if is_composer:
        composer_text.append(content)
```

**Tại sao task counting?** CrewAI chunk objects có `agent_role` attribute nhưng thường = `None` hoặc `""`. Chỉ dựa vào tên sẽ bỏ sót. Dựa vào vị trí (last task) luôn đúng.

**Capture answer — 4 tầng fallback:**
```python
raw_answer = ""
# Tầng 1: CrewAI's official result
if streaming.is_completed: raw_answer = streaming.result.raw
# Tầng 2: CrewAI's full text
if not raw_answer: raw_answer = streaming.get_full_text()
# Tầng 3: Composer chunks only
if not raw_answer: raw_answer = "".join(composer_text)
# Tầng 4: ALL chunks (includes intermediate agents — last resort)
if not raw_answer: raw_answer = "".join(all_text)
```

---

## 11. HTTP COMMUNICATION — AI GỌI LLM NHƯ THẾ NÀO?

CrewAI dùng OpenAI-compatible API. Mọi giao tiếp là HTTP:

```
  crew_runner.py                    LMStudio / vLLM
  (Parallels VM)                    port 1234
       │                                │
       │  POST /v1/chat/completions     │
       │  {                             │
       │    "model": "qwen/qwen3-...", │
       │    "messages": [               │
       │      {"role": "system",        │
       │       "content":               │
       │        "[SYSTEM TIME]...       │  ◄── datetime injected
       │         You are Web...         │  ◄── backstory from yaml
       │        "},                     │
       │      {"role": "user",          │
       │       "content":               │
       │        "Current date/time...   │  ◄── memory context
       │         Task: Research...      │  ◄── task from yaml
       │         [Pre-searched]...      │  ◄── query expansion results
       │        "}                      │
       │    ],                          │
       │    "tools": [                  │  ◄── tool schemas from @tool
       │      {"name":"web_search_...", │
       │       "parameters":{...}}     │
       │    ],                          │
       │    "max_tokens": 8192,         │  ◄── từ llm.yaml
       │    "extra_body": {             │
       │      "top_k": 40,             │  ◄── từ llm.yaml
       │      "repetition_penalty":1.1 │
       │    }                           │
       │  }                             │
       │ ──────────────────────────►    │
       │                                │  Qwen3 <think>...</think>
       │                                │  generate response
       │  ◄──────────────────────────   │
       │  {                             │
       │    "choices": [{               │
       │      "message": {              │
       │        "content": "Final       │
       │          Answer: {...}"        │
       │        // HOẶC                 │
       │        "tool_calls": [{...}]   │
       │      }                         │
       │    }]                          │
       │  }                             │
```

Mỗi "lượt suy nghĩ" của agent = 1 HTTP request/response.
Web researcher thường gọi LLM 2 lần (tool call + final answer).
Composer gọi 1 lần (max_iter=1, no tools).

---

## 12. SERVER.PY — FASTAPI + SSE CHO WEB CLIENTS

```python
# server.py — 2 endpoints

@app.post("/run")           # Sync: trả 1 JSON response hoàn chỉnh
@app.post("/run/stream")    # SSE: stream từng token qua EventSource
```

SSE streaming flow:

```
Web Client                      FastAPI                     CrewAI
  │                                │                           │
  │  POST /run/stream             │                           │
  │  {"query": "..."}            │                           │
  │ ──────────────────────────►   │                           │
  │                                │  crew.kickoff(stream=T)  │
  │                                │ ─────────────────────────►│
  │                                │                           │
  │  event: chunk                 │  ◄── chunk.content        │
  │  data: {"content": "Theo"}   │                           │
  │ ◄─────────────────────────    │                           │
  │                                │                           │
  │  event: chunk                 │  ◄── chunk.content        │
  │  data: {"content": " các"}   │                           │
  │ ◄─────────────────────────    │                           │
  │  ...                          │                           │
  │                                │                           │
  │  event: final                 │  ◄── streaming done       │
  │  data: {"result": "Full..."}  │                           │
  │ ◄─────────────────────────    │                           │
```

---

## 13. DÒNG CHẢY DỮ LIỆU TOÀN BỘ — SƠ ĐỒ

```
 .env (secrets) ──► config.py ──► resolve ${VAR} trong YAML
                        │
                   AppConfig object
                        │
        ┌───────────────┼──────────────────────┐
        │               │                      │
   run_chat.py      server.py              run.py
   (interactive)    (FastAPI+SSE)          (single-shot)
        │               │                      │
        └───────────────┼──────────────────────┘
                        │
                 crew_runner.py ◄─── "Đạo diễn"
                        │
        ┌───────────────┼──────────────────────┐
        │               │                      │
  _inject_datetime() route()            _build_crew()
  (prepend vào        (Router agent     (ghép tasks
   ALL backstories)    quyết định)       theo routes)
        │               │                      │
        └───────────────┼──────────────────────┘
                        │
                   Crew.kickoff()
                        │
        ┌───────────────┼──────────────────────┐
        │               │                      │
     web_task      student_task         compose_task
     (Researcher)  (Student API)        (Composer)
        │               │                      │
     Tavily search   REST API calls     Tổng hợp evidence
     + tool_call     + tool_call        → Vietnamese answer
     → JSON          → JSON              + sources
        │               │                      │
        └───────────────┼──────────────────────┘
                        │
              conversation_memory.py
                ┌───────┼────────┐
                │       │        │
           save turns  extract   compress
           to bank     facts     old turns
                │       │        │
           memory_bank  fact_    context_
           .py (JSONL)  store   compressor
```

---

## 14. CÁC BÀI HỌC TỪ DEBUGGING

| # | Vấn đề | Nguyên nhân | Giải pháp |
|---|--------|-------------|-----------|
| 1 | Agent loop vô hạn | LLM không output "Final Answer:" | Thêm prefix requirement vào MỌI task description |
| 2 | Composer chạy 2 lần | `max_iter=2` + no Final Answer → retry | `max_iter=1` (one-shot, no retry) |
| 3 | Think model garbage JSON | `<think>3000 tokens</think>` trước JSON | 3 tầng: NO_THINK_SYSTEM + disable thinking + extract_json() |
| 4 | Streaming hiện TẤT CẢ agents | `chunk.agent_role` = empty | Task counting: composer = last task |
| 5 | Secrets bị push Git | Hardcode keys trong YAML | `${ENV_VAR}` syntax + `.env` + `.gitignore` |
| 6 | Response quá lớn → LLM chết | `include_raw_content=true` + 8 results → 100K chars | Giảm defaults + truncate output 8000 chars |
| 7 | Fact extraction = 0 results | `max_tokens=2048` hết cho thinking tokens | `structured_max_tokens=16384` (1 chỗ trong llm.yaml) |
| 8 | Composer hallucinate | Model training data cũ contradicts web sources | Rule 8: "TRUST web sources, do NOT fact-check with outdated knowledge" |
| 9 | Query expansion UI đứng | OpenAI SDK call blocking, không feedback | Streaming + progress dots (`⏳ .......... done!`) |
| 10 | Verbose + streaming = double output | `verbose=True` prints to stdout AND chunks stream | Tắt verbose khi streaming: `agent.verbose = False` |

---

## 15. TÓM TẮT BẰNG 1 CÂU

**Hệ thống đọc YAML + .env → inject datetime vào mọi agent → mở rộng 1 câu hỏi thành 3 search queries → Tavily search → route tới đúng agents → CrewAI ReAct loop (prompt → tool_call → execute → Final Answer) → stream chỉ composer output → trích xuất facts từ hội thoại → lưu memory → trả kết quả tiếng Việt có nguồn dẫn.**

Bản chất nó vẫn là **vòng lặp gửi/nhận HTTP** nhưng thông minh hơn: biết ngày giờ, nhớ context, mở rộng query, filter streaming, xử lý thinking models, và bảo mật secrets.
