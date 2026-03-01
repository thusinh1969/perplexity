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
│   ├── llm.yaml                 Cấu hình bộ não AI (model nào, server nào)
│   ├── agents.yaml              Hồ sơ nhân viên (role, goal, backstory)
│   ├── tasks.yaml               Bảng mô tả công việc cho từng nhân viên
│   └── tools.yaml               API keys, endpoints cho tools
│
├── tools/                       ◄── "DỤNG CỤ" (Python functions)
│   ├── web_tools.py             Tìm kiếm web (Tavily API)
│   ├── student_tools.py         Tra cứu dữ liệu học sinh
│   ├── rag_tools.py             Tìm kiếm tài liệu nội bộ
│   └── speech_tools.py          Chuyển giọng nói ↔ text
│
├── config.py                    ◄── Đọc YAML → Python objects
├── tool_context.py              ◄── Chia sẻ config cho tools
├── crew_runner.py               ◄── "ĐẠO DIỄN" — tạo agents, tasks, crew, chạy
├── run.py                       ◄── CLI entry point (chạy từ terminal)
├── run_async.py                 ◄── Async version
└── server.py                    ◄── HTTP API (FastAPI)
```

---

## 3. YAML FILES — "HỒ SƠ" VIẾT BẰNG TIẾNG NGƯỜI

### 3.1 llm.yaml — Bộ não AI

```yaml
llm:
  model: "qwen/qwen3-coder-next"           # Model AI nào?
  base_url: "http://10.211.55.2:1234/v1"   # Server nào chạy model?
  api_key: "sk-..."                         # Mật khẩu truy cập
  temperature: 0.2                          # 0=máy móc, 1=sáng tạo
  max_tokens: 4096                          # Giới hạn độ dài trả lời
```

File này trả lời: **"Bộ não AI ở đâu và cấu hình thế nào?"**

CrewAI không có AI riêng. Nó gọi tới một LLM server bên ngoài (LMStudio, OpenAI, vLLM...) qua HTTP API chuẩn OpenAI.

### 3.2 agents.yaml — Hồ sơ nhân viên

```yaml
web_researcher:
  role: "Web Researcher"                     # Chức danh
  goal: "Search the web, summarize..."       # Mục tiêu
  backstory: >                               # Tính cách + quy tắc
    You use web search tools to find information.
    CRITICAL WORKFLOW: 1) Call a search tool ONCE.
    2) Read the results. 3) Immediately write your Final Answer.
    NEVER call the same tool twice.
  allow_delegation: false                    # Không được ủy quyền cho agent khác
```

File này trả lời: **"Nhân viên nào, tính cách ra sao, giới hạn gì?"**

Mỗi trường sẽ được CrewAI nhét vào **system prompt** gửi cho LLM:

```
                agents.yaml                        System Prompt gửi cho LLM
         ┌─────────────────────┐          ┌──────────────────────────────────┐
         │ role: "Web          │  ──────► │ You are Web Researcher.          │
         │       Researcher"   │          │ Your personal goal is: Search    │
         │ goal: "Search..."   │          │ the web...                       │
         │ backstory: "You     │  ──────► │ You use web search tools to find │
         │   use web search    │          │ information. CRITICAL WORKFLOW:  │
         │   tools..."         │          │ 1) Call a search tool ONCE...    │
         └─────────────────────┘          └──────────────────────────────────┘
```

### 3.3 tasks.yaml — Phiếu giao việc

```yaml
web_task:
  description: |                     # Mô tả chi tiết việc cần làm
    Search the web for:
    {user_query}                     # ← Biến, sẽ được thay bằng câu hỏi thật

    Steps:
    1) Call web_search_deep with your query.
    2) Read the search results carefully.
    3) Provide your Final Answer as JSON.

    IMPORTANT: Do NOT call tools more than once.
  expected_output: "JSON with findings and sources."   # Kết quả mong đợi
  agent: web_researcher              # Giao cho ai? → map tới agents.yaml
```

File này trả lời: **"Việc gì, giao cho ai, kết quả mong đợi?"**

`{user_query}` là template variable — khi chạy, CrewAI thay bằng giá trị thật từ `inputs={}`.

### 3.4 tools.yaml — Config cho dụng cụ

```yaml
web:
  tavily:
    api_key: "tvly-xxx"        # API key để gọi Tavily
    search_depth: "basic"
    max_results: 3
```

File này trả lời: **"Dụng cụ nào cần mật khẩu gì, endpoint ở đâu?"**

Khác với 3 file trên (CrewAI đọc), file này **code Python đọc** qua `tool_context.py`.

---

## 4. PYTHON CODE — AI "LÀM VIỆC" NHƯ THẾ NÀO?

### 4.1 Tools — Hàm Python mà AI có thể gọi

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
         Python @tool                         JSON Schema gửi cho LLM
    ┌──────────────────┐              ┌─────────────────────────────────┐
    │ def web_search(  │              │ {                               │
    │   query: str,    │  ──────────► │   "name": "web_search_deep",   │
    │   max_results:   │              │   "description": "Tavily...",   │
    │     int = 3      │              │   "parameters": {              │
    │ ) -> str:        │              │     "query": {"type":"string"},│
    │   """Tavily."""  │              │     "max_results": {"type":    │
    └──────────────────┘              │       "integer","default":3}   │
                                      │   }                            │
                                      │ }                               │
                                      └─────────────────────────────────┘
```

**Quan trọng**: LLM thấy default values từ Python signature (`= 3`, `= "basic"`, `= False`), KHÔNG phải từ tools.yaml. Đó là lý do khi code ghi `max_results=8` mà tools.yaml ghi `max_results=3`, LLM vẫn gọi với 8.

### 4.2 crew_runner.py — "Đạo diễn" ghép mọi thứ

```python
def _make_llm(cfg):
    """Tạo connection tới LLM server"""
    return LLM(
        model=cfg.llm.model,          # từ llm.yaml
        base_url=cfg.llm.base_url,    # từ llm.yaml
        ...
    )

def _make_agents(cfg, llm):
    """Tạo nhân viên, mỗi người có role + tools riêng"""
    web = Agent(
        llm=llm,                                              # Bộ não chung
        tools=[web_search_deep, web_crawl_url, ...],          # Dụng cụ riêng
        max_iter=3,                                           # Max 3 lần gọi tool
        **agents_yaml["web_researcher"],                      # role, goal, backstory
    )
    # ... tương tự cho student, rag, composer ...

def _make_task(cfg, name, agents):
    """Tạo phiếu việc, gắn với agent"""
    return Task(
        description=tasks_yaml[name]["description"],          # từ tasks.yaml
        expected_output=tasks_yaml[name]["expected_output"],
        agent=agents[task["agent"]],                          # giao cho ai
    )

def run_crew(cfg, routes, inputs):
    """Ghép đội + chạy"""
    crew = Crew(
        agents=[...],          # Danh sách nhân viên
        tasks=[...],           # Danh sách phiếu việc (THỨ TỰ QUAN TRỌNG!)
    )
    return crew.kickoff(inputs=inputs)    # BẮT ĐẦU CHẠY!
```

---

## 5. RUNTIME — CHUYỆN GÌ XẢY RA KHI `crew.kickoff()` ?

Đây là phần quan trọng nhất. Khi gọi `crew.kickoff()`, CrewAI thực hiện vòng lặp **ReAct** (Reasoning + Acting) cho từng task:

### 5.1 Tổng quan flow

```
 kickoff(inputs={"user_query": "Python 3.13?"})
    │
    ▼
 ┌─ TASK 1: web_task (agent: web_researcher) ──────────────────────┐
 │                                                                  │
 │  CrewAI ghép prompt → gửi LLM → nhận response                  │
 │  → có tool_call? → chạy tool → gửi kết quả lại LLM            │
 │  → lặp cho tới khi LLM trả "Final Answer"                      │
 │                                                                  │
 │  Output task 1 = JSON findings + sources                        │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                    output task 1 tự động truyền vào context task 2
                                    │
                                    ▼
 ┌─ TASK 2: compose_task (agent: composer) ─────────────────────────┐
 │                                                                   │
 │  CrewAI ghép: task description + output từ task 1               │
 │  → gửi LLM → nhận Final Answer                                  │
 │                                                                   │
 │  Output = Câu trả lời tiếng Việt hoàn chỉnh                    │
 └──────────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
                            crew.kickoff() trả về
                              CrewOutput.raw
```

### 5.2 Chi tiết: CrewAI ghép prompt như thế nào?

Cho **web_researcher** làm **web_task**, CrewAI tạo ra prompt thế này:

```
┌─────────────────── SYSTEM MESSAGE ───────────────────────────────┐
│                                                                   │
│  You are Web Researcher.              ◄── agents.yaml: role      │
│  You use web search tools to find     ◄── agents.yaml: backstory │
│  information. CRITICAL WORKFLOW:                                  │
│  1) Call a search tool ONCE...                                    │
│  Your personal goal is: Search the    ◄── agents.yaml: goal      │
│  web for information, then summarize                              │
│  findings with citations.                                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────── USER MESSAGE ─────────────────────────────────┐
│                                                                   │
│  Current Task: Search the web for:    ◄── tasks.yaml: description│
│  Python 3.13 có gì mới?              ◄── {user_query} đã thay   │
│                                                                   │
│  Steps:                                                           │
│  1) Call web_search_deep with your query.                        │
│  2) Read the search results carefully.                           │
│  3) Provide your Final Answer as JSON.                           │
│                                                                   │
│  IMPORTANT: Do NOT call tools more than once.                    │
│                                                                   │
│  This is the expected criteria for    ◄── tasks.yaml:            │
│  your final answer:                       expected_output        │
│  JSON with findings and sources.                                 │
│                                                                   │
│  you MUST return the actual complete                              │
│  content as the final answer,         ◄── CrewAI tự thêm        │
│  not a summary.                                                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

┌─────────────────── TOOLS (gửi kèm) ─────────────────────────────┐
│                                                                   │
│  tools: [                             ◄── Từ @tool decorators    │
│    {                                                              │
│      "name": "web_search_deep",                                  │
│      "description": "Tavily Search...",                          │
│      "parameters": {                                              │
│        "query": {"type": "string"},                              │
│        "max_results": {"type": "integer", "default": 3},        │
│        "search_depth": {"type": "string", "default": "basic"},  │
│        "include_raw_content": {"type": "boolean", "default": false}│
│      }                                                            │
│    },                                                             │
│    { "name": "web_crawl_url", ... },                             │
│    { "name": "web_search_then_crawl", ... }                      │
│  ]                                                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

Toàn bộ cục này gửi qua HTTP POST tới LMStudio.

### 5.3 Vòng lặp ReAct — Tool Calling

```
         CrewAI                      LLM (Qwen3)                   Tavily API
           │                             │                              │
           │  ── prompt + tools ──────►  │                              │
           │                             │                              │
           │                             │  (suy nghĩ...)               │
           │                             │  "Tôi cần search web"        │
           │                             │                              │
           │  ◄── tool_call: ──────────  │                              │
           │      web_search_deep(       │                              │
           │        query="Python 3.13   │                              │
           │          new features",     │                              │
           │        max_results=3        │                              │
           │      )                      │                              │
           │                             │                              │
           │  CrewAI thấy tool_call      │                              │
           │  → chạy hàm Python thật     │                              │
           │                             │                              │
           │  ── HTTP request ──────────────────────────────────────►   │
           │                             │                              │
           │  ◄── JSON results (3 kết quả web) ────────────────────    │
           │                             │                              │
           │  ── gửi tool result ──────► │                              │
           │     cho LLM đọc             │                              │
           │                             │                              │
           │                             │  (đọc kết quả, tóm tắt)     │
           │                             │                              │
           │  ◄── "Final Answer: ─────── │                              │
           │       {findings: [...],     │                              │
           │        sources: [...]}"     │                              │
           │                             │                              │
           │  CrewAI thấy "Final Answer" │                              │
           │  → DỪNG vòng lặp           │                              │
           │  → chuyển sang task tiếp    │                              │
```

**Vòng lặp này là TRÁI TIM của CrewAI:**
1. Gửi prompt + tools cho LLM
2. LLM trả lời: tool_call HOẶC Final Answer
3. Nếu tool_call → chạy tool thật → gửi kết quả lại (quay bước 1)
4. Nếu Final Answer → DỪNG → chuyển task tiếp
5. Nếu quá `max_iter` lần → buộc dừng

### 5.4 Chuyền bóng giữa các Tasks

CrewAI mặc định chạy **sequential** (tuần tự). Output task trước = input context task sau:

```
 TASK 1 output:                         TASK 2 nhận được:
 ┌────────────────────────┐             ┌────────────────────────────┐
 │ {                      │             │ [system] You are Composer  │
 │   "findings": [        │   ──────►   │ [user]                     │
 │     "Python 3.13 có    │             │   Current Task: Compose... │
 │      JIT compiler...", │             │                            │
 │     "Free-threaded..." │             │   CONTEXT từ task trước:   │
 │   ],                   │             │   {"findings": [...],      │
 │   "sources": [...]     │             │    "sources": [...]}       │
 │ }                      │             │                            │
 └────────────────────────┘             └────────────────────────────┘
```

Composer không cần tools — nó chỉ đọc kết quả và viết lại cho đẹp.

---

## 6. ROUTING — TẦNG QUYẾT ĐỊNH TRƯỚC KHI CHẠY CREW

Code của chúng ta thêm một tầng **routing** TRƯỚC khi tạo Crew:

```
                    User Query
                         │
                         ▼
              ┌─── route() ───────────────────────────┐
              │                                        │
              │  Tạo mini-crew chỉ có Router agent    │
              │  Router nhận câu hỏi → trả JSON:      │
              │                                        │
              │  {"routes": ["web"],                   │
              │   "student_id": null,                  │
              │   "policy_domain": "other"}            │
              │                                        │
              └────────────────┬───────────────────────┘
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
                        Luôn thêm
                       compose_task
                               │
                               ▼
                     Crew([tasks...]).kickoff()
```

Routing cho phép hệ thống **chỉ chạy agent cần thiết** thay vì chạy hết.

---

## 7. HTTP COMMUNICATION — AI GỌI LLM NHƯ THẾ NÀO?

CrewAI dùng OpenAI-compatible API. Mọi giao tiếp là HTTP:

```
  crew_runner.py                    LMStudio (Mac)
  (Parallels VM)                    port 1234
       │                                │
       │  POST /v1/chat/completions     │
       │  {                             │
       │    "model": "qwen/qwen3-...", │
       │    "messages": [               │
       │      {"role": "system",        │
       │       "content": "You are..."} │
       │      {"role": "user",          │
       │       "content": "Task..."}    │
       │    ],                          │
       │    "tools": [...],             │
       │    "max_tokens": 4096          │
       │  }                             │
       │ ──────────────────────────►    │
       │                                │  Qwen3 suy nghĩ...
       │                                │  generate response
       │  ◄──────────────────────────   │
       │  {                             │
       │    "choices": [{               │
       │      "message": {              │
       │        "tool_calls": [{        │
       │          "function": {         │
       │            "name": "web_...",  │
       │            "arguments": "..."  │
       │          }                     │
       │        }]                      │
       │      }                         │
       │    }]                          │
       │  }                             │
```

Mỗi "lượt suy nghĩ" của agent = 1 HTTP request/response.
Một agent có thể gọi LLM 2-3 lần (tool call + final answer).

---

## 8. DÒNG CHẢY DỮ LIỆU TOÀN BỘ — TỪ A TỚI Z

```
 Terminal: python -m school_agents.run --query "Python 3.13 có gì mới?"
    │
    ▼
 run.py
    │  1. load_config("config/") → đọc 4 file YAML
    │  2. bootstrap(cfg) → chia sẻ tools.yaml cho tools
    │
    │  ══════════════ PHASE 1: ROUTING ══════════════
    │
    │  3. route(cfg, query, ...) →
    │     ├── _make_llm()        → LLM(model="qwen/...", base_url="http://...")
    │     ├── _make_agents()     → tạo 5 Agent objects (từ agents.yaml)
    │     ├── _make_task("route_task")  → tạo Task (từ tasks.yaml)
    │     ├── Crew([router], [route_task])
    │     └── crew.kickoff(inputs={user_query: "Python 3.13?"})
    │           │
    │           │  ┌──── HTTP ────┐
    │           │  │ POST /v1/... │ → LMStudio → Qwen3
    │           │  │ ◄── JSON ──  │ ← {"routes":["web"]}
    │           │  └──────────────┘
    │           │
    │     return {"routes": ["web"], "policy_domain": "other"}
    │
    │  ══════════════ PHASE 2: EXECUTION ══════════════
    │
    │  4. routes = ["web"]
    │  5. run_crew(cfg, routes, inputs) →
    │     ├── _build_crew(routes=["web"])
    │     │    ├── tasks = [web_task, compose_task]    (2 tasks vì route=web)
    │     │    └── Crew(agents=[all 5], tasks=[2])
    │     │
    │     └── crew.kickoff(inputs={user_query, student_id, ...})
    │
    │           ┌──── TASK 1: web_task ─────────────────────────┐
    │           │                                                │
    │           │  HTTP #1: prompt + tools → LLM                │
    │           │  LLM trả: tool_call web_search_deep(...)      │
    │           │                                                │
    │           │  CrewAI chạy Python: web_search_deep()         │
    │           │    → TavilyClient.search()                    │
    │           │    → HTTP tới api.tavily.com                  │
    │           │    → nhận 3 kết quả                           │
    │           │    → truncate nếu > 8000 chars                │
    │           │    → trả JSON string cho CrewAI               │
    │           │                                                │
    │           │  HTTP #2: tool result → LLM                   │
    │           │  LLM trả: "Final Answer: {findings, sources}" │
    │           │                                                │
    │           │  → output = JSON string                       │
    │           └───────────────────────┬────────────────────────┘
    │                                   │
    │           ┌──── TASK 2: compose_task ─────────────────────┐
    │           │                                                │
    │           │  HTTP #3: task desc + TASK 1 output → LLM     │
    │           │  LLM trả: câu trả lời tiếng Việt + nguồn     │
    │           │                                                │
    │           │  → output = final answer string               │
    │           └───────────────────────┬────────────────────────┘
    │                                   │
    │  6. print(out.raw)  → hiện kết quả
    ▼
 "Python 3.13 chính thức phát hành ngày 7/10/2024..."
```

---

## 9. CÁC BÀI HỌC QUAN TRỌNG TỪ DEBUGGING

### 9.1 LLM thấy code Python, KHÔNG thấy YAML

```
 tools.yaml: max_results: 3        ← Code Python đọc (cho app logic)
 @tool signature: max_results=8     ← LLM đọc (qua JSON schema)

 LLM gọi tool với max_results=8, KHÔNG PHẢI 3!
```

**Bài học**: Default trong `@tool` function signature = default mà LLM sẽ dùng.

### 9.2 Streaming + Tool Calling = Không tương thích (LMStudio)

```
 stream=True + tools → LMStudio trả tool_call qua SSE chunks
                     → CrewAI không parse được
                     → trả None
                     → crash

 stream=False + tools → LMStudio trả 1 JSON response hoàn chỉnh
                      → CrewAI parse OK ✅
```

### 9.3 Response quá lớn → LLM chết

```
 include_raw_content=true + max_results=8
   → Tavily trả ~50,000-100,000 chars (12K-25K tokens)
   → vượt context window model local
   → LLM trả None

 Giải pháp: giảm defaults + truncate output ở 8000 chars
```

### 9.4 Agent loop vô hạn nếu LLM không biết format

```
 CrewAI mong đợi: "Final Answer: ..."
 Qwen3 trả: gọi tool lại → gọi tool lại → ...

 Giải pháp 3 tầng:
   1. backstory nói rõ "call ONCE → Final Answer"
   2. task description nói "Do NOT call tools more than once"
   3. max_iter=3 (safety net)
```

---

## 10. TÓM TẮT BẰNG 1 CÂU

**CrewAI đọc YAML configs → tạo prompt cho từng agent → gửi prompt + tool schemas tới LLM qua HTTP → nhận tool_calls hoặc Final Answer → nếu tool_call thì chạy Python function rồi gửi kết quả lại → lặp tới khi xong → chuyền output cho task tiếp theo → cuối cùng trả kết quả.**

Bản chất nó chỉ là một **vòng lặp gửi/nhận HTTP thông minh** biết cách ghép prompt từ YAML, quản lý tool execution, và chuyền dữ liệu giữa các tasks.
