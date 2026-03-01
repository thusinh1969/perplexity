# School Agents — Full Technical Architecture Report

## System Overview

Multi-agent conversational AI system built on CrewAI, with 3-tier memory,
domain-agnostic knowledge extraction, multi-query search expansion, and
JSONL-persistent session management. All configuration driven by YAML with
CLI override support.

```
┌──────────────────────────── SYSTEM ARCHITECTURE ────────────────────────────┐
│                                                                             │
│   User Terminal                                                             │
│       │                                                                     │
│       ▼                                                                     │
│   run_chat.py ─── CLI entry point                                          │
│       │                                                                     │
│       ├── config.py ──────── loads 5 YAML files into frozen dataclasses    │
│       │   ├── llm.yaml          LLM endpoint + model config               │
│       │   ├── agents.yaml       Agent roles, goals, backstories           │
│       │   ├── tasks.yaml        Task descriptions + expected outputs      │
│       │   ├── tools.yaml        API keys, endpoints for tools             │
│       │   └── memory.yaml       Memory, compression, expansion config     │
│       │                                                                     │
│       ├── memory_bank.py ──── Persistence layer                            │
│       │   ├── MemoryDB          In-memory + JSONL write-ahead log         │
│       │   └── RedisMemoryBank   Redis-backed (production)                 │
│       │                                                                     │
│       ├── conversation_memory.py ── 3-tier context manager                 │
│       │   ├── Tier 1: FactStore      entities/relations/facts (NEVER cut) │
│       │   ├── Tier 2: Recent turns   sliding window (verbatim)            │
│       │   └── Tier 3: Summary        compressed older turns               │
│       │                                                                     │
│       ├── fact_store.py ──── Domain-agnostic knowledge graph               │
│       ├── context_compressor.py ── LLM summary / LLMLingua / hybrid       │
│       ├── query_expander.py ── Multi-query + reciprocal rank fusion        │
│       │                                                                     │
│       └── crew_runner.py ──── CrewAI orchestration                         │
│           ├── route()                 Router agent → JSON routing          │
│           ├── run_crew_with_memory()  Full pipeline with memory            │
│           └── _build_crew()           Agent + Task assembly                │
│               │                                                             │
│               └── tools/                                                    │
│                   ├── web_tools.py       Tavily search/crawl/expanded      │
│                   ├── student_tools.py   Internal student API              │
│                   └── rag_tools.py       Policy document retrieval         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Map

```
src/school_agents/
├── config/
│   ├── llm.yaml              LLM connection (model, base_url, temperature)
│   ├── agents.yaml           5 agents: router, web, student, rag, composer
│   ├── tasks.yaml            6 tasks: route, web, student, policy, compose
│   ├── tools.yaml            Tavily key, student API, RAG endpoint, audio
│   └── memory.yaml           Memory bank, compression, expansion, facts
│
├── config.py                 LLMConfig + MemoryConfig + AppConfig dataclasses
├── tool_context.py           Global tool config sharing
│
├── memory_bank.py            MemoryBankBase → MemoryDB (JSONL) + RedisMemoryBank
├── conversation_memory.py    ConversationMemory: 3-tier context manager
├── fact_store.py             FactStore: entity/relation/fact extraction + merge
├── context_compressor.py     LLM summary, LLMLingua, hybrid compression
├── query_expander.py         QueryExpander: multi-query + RRF fusion
│
├── crew_runner.py            CrewAI agent/task/crew assembly + execution
├── run_chat.py               Interactive multi-turn CLI (REPL + single-shot)
├── run.py                    Original single-shot CLI (backward compatible)
├── run_async.py              Async variant
├── server.py                 FastAPI HTTP interface
│
└── tools/
    ├── web_tools.py          Tavily: search, crawl, expanded, helpers
    ├── student_tools.py      Student API: profile, grades, attendance
    ├── rag_tools.py          RAG: policy document retrieval
    └── speech_tools.py       STT/TTS (disabled)
```

---

## End-to-End Flow — Single Turn

```
 python -m school_agents.run_chat -s 123456789 -i
 You: Điểm của Nguyễn Văn A mã SV 20210345?
  │
  │  ╔══════════════════ PHASE 0: STARTUP ══════════════════╗
  │  ║                                                       ║
  │  ║  load_config() → reads 5 YAML files                  ║
  │  ║  bootstrap()   → shares tools.yaml with tool_context  ║
  │  ║  MemoryDB()    → replays sessions.jsonl into RAM      ║
  │  ║  FactStore()   → loads facts:123456789 from bank      ║
  │  ║                                                       ║
  │  ╚═══════════════════════════════════════════════════════╝
  │
  ▼
 ┌─── _run_one_turn() ────────────────────────────────────────────────┐
 │                                                                     │
 │  STEP 1: memory.add_user_turn(query)                               │
 │          → appends Turn(role="user", content=query)                │
 │                                                                     │
 │  STEP 2: memory.build_context(current_query=query)                 │
 │          → checks _needs_compression()                             │
 │          → if over budget: _compress() older turns → update summary│
 │          → assembles 3-tier context string (see below)             │
 │                                                                     │
 │  STEP 3: route(cfg, query, context)                                │
 │          → mini Crew with router agent only                        │
 │          → 1 LLM call → JSON: {"routes":["student"]}              │
 │                                                                     │
 │  STEP 4: query expansion (only if "web" in routes)                 │
 │          → SKIPPED here (route = student, not web)                 │
 │                                                                     │
 │  STEP 5: run_crew_with_memory(cfg, routes, inputs, memory)         │
 │          → _build_crew() creates student_task + compose_task       │
 │          → crew.kickoff(inputs={user_query, conversation_context}) │
 │          │                                                          │
 │          │  ┌── TASK 1: student_task (agent: student_data) ───┐    │
 │          │  │ LLM call #1: prompt + tools → tool_call:        │    │
 │          │  │   student_get_profile("20210345")                │    │
 │          │  │ → Python executes → API response                 │    │
 │          │  │ LLM call #2: tool result → tool_call:            │    │
 │          │  │   student_get_grades("20210345", ...)            │    │
 │          │  │ → Python executes → API response                 │    │
 │          │  │ LLM call #3: tool result → "Final Answer: {...}" │    │
 │          │  └─────────────────────┬───────────────────────────┘    │
 │          │                        │ output auto-passed             │
 │          │  ┌── TASK 2: compose_task (agent: composer) ───────┐    │
 │          │  │ LLM call #4: context + student data → final     │    │
 │          │  │   Vietnamese answer with grades                  │    │
 │          │  └─────────────────────┬───────────────────────────┘    │
 │          │                        │                                 │
 │          │  memory.add_assistant_turn(answer)                      │
 │          │                                                          │
 │          │  STEP 6: Fact extraction                                │
 │          │  → LLM call #5: extract entities/relations/facts        │
 │          │    from last user+assistant turn pair                    │
 │          │  → merge into FactStore (dedup by id/key)               │
 │          │                                                          │
 │          │  memory.save() → bank.put() → in-memory dict            │
 │          │  bank.flush() → append JSONL to disk                    │
 │                                                                     │
 │  RETURN answer                                                      │
 └─────────────────────────────────────────────────────────────────────┘
  │
  ▼
 Assistant (18.3s):
 Nguyễn Văn A (mã SV 20210345) có điểm: Toán 8.5, Lý 7.0, Hóa 9.0...
```

**LLM calls per turn (worst case): 5-6**

```
 #1  Router            → route decision JSON
 #2  Query expansion   → 3 search queries (only web routes)
 #3  Tool agent call 1 → tool_call
 #4  Tool agent call 2 → Final Answer
 #5  Composer          → final Vietnamese answer
 #6  Fact extraction   → entity/relation/fact JSON
```

---

## 3-Tier Memory Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                     CONTEXT WINDOW LAYOUT                           ║
║                                                                     ║
║  ┌─────────────────────────────────────────────────────────┐        ║
║  │  SYSTEM PROMPT (CrewAI auto-generated)                  │        ║
║  │  Agent role + goal + backstory + tool schemas           │ FIXED  ║
║  │  ~500 tokens                                            │        ║
║  ├─────────────────────────────────────────────────────────┤        ║
║  │  TIER 1: KNOWN FACTS (FactStore)                        │        ║
║  │                                                         │        ║
║  │  [Known entities]                                       │ NEVER  ║
║  │  - Nguyễn Văn A [person] (student_id=20210345)          │ CUT    ║
║  │  - Python 3.13 [software] (release=2024-10-07)          │ NEVER  ║
║  │  [Known relations]                                      │ TRUNC  ║
║  │  - nguyen_van_a → scored → toan_8.5                     │        ║
║  │  [Known facts]                                          │ GROWS  ║
║  │  - retake_policy: Sinh viên dưới 4.0 được thi lại      │ ONLY   ║
║  │  ~200-800 tokens (grows with conversation)              │        ║
║  ├─────────────────────────────────────────────────────────┤        ║
║  │  TIER 3: NARRATIVE SUMMARY (compressed)                 │        ║
║  │                                                         │ COMP-  ║
║  │  "User hỏi điểm SV 20210345, sau đó hỏi quy chế       │ RESSED ║
║  │   thi lại. Tiếp theo hỏi về Python 3.13."              │        ║
║  │  ~100-200 tokens (stable, replaces older turns)         │        ║
║  ├─────────────────────────────────────────────────────────┤        ║
║  │  TIER 2: RECENT TURNS (verbatim, sliding window)        │        ║
║  │                                                         │ RAW    ║
║  │  User: Cái JIT compiler hoạt động thế nào?              │        ║
║  │  Assistant: JIT trong Python 3.13 là copy-and-patch...  │ 2-3    ║
║  │  User: Có benchmark so sánh tốc độ không?               │ PAIRS  ║
║  │  Assistant: Theo PyCon benchmarks, JIT cải thiện...     │        ║
║  │  ~500-1500 tokens                                       │        ║
║  ├─────────────────────────────────────────────────────────┤        ║
║  │  CURRENT TASK + QUERY                                   │ FROM   ║
║  │  + tool schemas                                         │ YAML   ║
║  │  ~200-500 tokens                                        │        ║
║  ├─────────────────────────────────────────────────────────┤        ║
║  │  ░░░░░░ HEADROOM FOR LLM OUTPUT ░░░░░░░░░░░░░░░░░░░    │        ║
║  │  ~2000-4000 tokens                                      │        ║
║  └─────────────────────────────────────────────────────────┘        ║
║                                                                     ║
║  Total: fits 8K-32K context window                                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Why 3 tiers instead of just summary?**

```
 Pure summary:
   "User asked about a student's grades" → LOST: which student, what grades

 3-tier:
   Tier 1 (facts):  nguyen_van_a, student_id=20210345, Toán=8.5 → PRESERVED
   Tier 3 (summary): "User asked about student grades"           → flow only
   Tier 2 (recent):  exact last 2-3 exchanges                    → detail
```

---

## Compression Flow

```
 Turn 1:  User asks X          ─┐
 Turn 2:  AI answers X'         │
 Turn 3:  User asks Y           │  All verbatim (under budget)
 Turn 4:  AI answers Y'         │
 Turn 5:  User asks Z          ─┘
 Turn 6:  AI answers Z'
 Turn 7:  User asks W           ← triggers _needs_compression()
                                    unsummarized tokens > max_context_tokens

 ┌─────────────────────────────────────────────────────────┐
 │ _compress() runs:                                       │
 │                                                         │
 │ Turns 1-4 → ContextCompressor.compress()                │
 │           → LLM summary call (temp=0.1, max_tokens=400) │
 │           → "User asked X (answered X'), then Y (Y')"   │
 │                                                         │
 │ summary_covers_up_to = 4                                │
 │                                                         │
 │ Result:                                                 │
 │   summary = "User asked X, then Y"  (~50 tokens)       │
 │   recent = [Turn 5, 6, 7]           (~750 tokens)       │
 │   facts = {entities, relations}      (~300 tokens)       │
 │   ────────────────────────                              │
 │   Total history: ~1100 tokens ✅ under budget           │
 └─────────────────────────────────────────────────────────┘
```

**Compression strategy options (memory.yaml):**

```
 llm_summary  → 1 LLM call, semantic, ~1-3s, best quality
 llmlingua    → BERT token filter, 0.1s, no LLM call, 2-3x compression
 hybrid       → LLM summary first, then LLMLingua if still too long
```

---

## Fact Extraction Pipeline

```
 After each turn:

 ┌─────────────────────────────────────┐
 │ User: Điểm Nguyễn Văn A mã 20210345│
 │ AI: Toán 8.5, Lý 7.0, Hóa 9.0     │
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌──────────────────────────────────────────────┐
 │ LLM call (temp=0.0):                         │
 │                                               │
 │ "Analyze conversation, extract JSON:          │
 │  entities, relations, facts.                  │
 │  Merge with existing knowledge."              │
 │                                               │
 │ Existing: [Known entities] ...                │
 └──────────────┬───────────────────────────────┘
                │
                ▼
 ┌──────────────────────────────────────────────┐
 │ LLM returns:                                  │
 │ {                                             │
 │   "entities": [                               │
 │     {"id":"nguyen_van_a", "type":"person",    │
 │      "name":"Nguyễn Văn A",                   │
 │      "attributes":{"student_id":"20210345"}}, │
 │     {"id":"toan", "type":"course",            │
 │      "name":"Toán"}                           │
 │   ],                                          │
 │   "relations": [                              │
 │     {"subject":"nguyen_van_a",                │
 │      "predicate":"scored",                    │
 │      "object":"8.5",                          │
 │      "detail":"môn Toán"}                     │
 │   ],                                          │
 │   "facts": [                                  │
 │     {"key":"grading_scale",                   │
 │      "value":"Thang điểm 10",                 │
 │      "confidence":"inferred"}                 │
 │   ]                                           │
 │ }                                             │
 └──────────────┬───────────────────────────────┘
                │
                ▼
 ┌──────────────────────────────────────────────┐
 │ FactStore._merge():                           │
 │                                               │
 │ - Existing entity? → merge attributes         │
 │ - New entity? → add                           │
 │ - Same relation key? → overwrite (newer wins) │
 │ - Same fact key? → overwrite                  │
 │                                               │
 │ Dedup by: entity.id, relation.key, fact.key   │
 └──────────────────────────────────────────────┘
```

---

## Query Expansion Flow

```
 User: "python mới có gì hay"
  │
  │ memory.yaml: query_expand.enabled=true, mode="confirm"
  │
  ▼
 ┌─── _handle_query_expansion() ───────────────────────────────────┐
 │                                                                  │
 │ STEP 1: QueryExpander.expand(query)                             │
 │         → LLM call (temp=0.3): "Generate 3 search queries"      │
 │         → ["Python 3.13 new features official",                 │
 │            "Python latest release improvements 2024",            │
 │            "Python 3.13 vs 3.12 comparison changes"]             │
 │                                                                  │
 │ STEP 2: Interactive confirmation (mode=confirm)                  │
 │         ┌────────────────────────────────────────────────┐       │
 │         │ 🔍 Query expansion (3 queries):                │       │
 │         │    1. Python 3.13 new features official        │       │
 │         │    2. Python latest release improvements 2024  │       │
 │         │    3. Python 3.13 vs 3.12 comparison changes   │       │
 │         │                                                │       │
 │         │    Search all? [Y/n/edit]                      │       │
 │         └────────────────────────────────────────────────┘       │
 │         User: Y                                                  │
 │                                                                  │
 │ STEP 3: Search all queries via Tavily                            │
 │         Q1 → 3 results                                          │
 │         Q2 → 3 results                                          │
 │         Q3 → 3 results                                          │
 │         Total: 9 raw results                                     │
 │                                                                  │
 │ STEP 4: Reciprocal Rank Fusion (RRF)                            │
 │                                                                  │
 │   For each URL across all queries:                              │
 │     score = Σ 1/(60 + rank_in_query)                            │
 │                                                                  │
 │   URL_A appeared: Q1 rank 1, Q2 rank 2                         │
 │     score = 1/61 + 1/62 = 0.0326                               │
 │   URL_B appeared: Q1 rank 3 only                                │
 │     score = 1/63 = 0.0159                                       │
 │                                                                  │
 │   Sort by score → dedup by URL → top 5                         │
 │                                                                  │
 │   Result: 5 merged, ranked results                              │
 │                                                                  │
 │ STEP 5: Inject into inputs["user_query"]                        │
 │   "python mới có gì hay\n\n[Pre-searched results]\n..."         │
 │                                                                  │
 │   Composer sees both the question AND pre-fetched results       │
 └──────────────────────────────────────────────────────────────────┘
```

**Mode comparison:**

```
 mode: "auto"      → expand + search silently, no user prompt
 mode: "confirm"   → show queries, ask [Y/n/edit], user can remove queries
 CLI: --expand off  → disable expansion entirely
```

---

## Persistence & Session Management

```
 ┌─────── MemoryDB Lifecycle ──────────────────────────────────────┐
 │                                                                  │
 │ STARTUP:                                                        │
 │   MemoryDB("./data/memory")                                    │
 │     → mkdir -p ./data/memory/                                   │
 │     → replay sessions.jsonl line by line                        │
 │     → rebuild in-memory dict                                    │
 │                                                                  │
 │ PER TURN:                                                       │
 │   bank.put("session:123", {turns, summary, ...})                │
 │     → writes to in-memory dict                                  │
 │     → marks key as dirty                                        │
 │   bank.put("facts:123", {entities, relations, facts})           │
 │     → writes to in-memory dict                                  │
 │     → marks key as dirty                                        │
 │   bank.flush()                                                  │
 │     → appends dirty keys to JSONL                               │
 │     → clears dirty set                                          │
 │                                                                  │
 │ SHUTDOWN:                                                       │
 │   bank.close()                                                  │
 │     → final flush()                                             │
 │                                                                  │
 │ JSONL FORMAT (append-only write-ahead log):                     │
 │   {"op":"put","key":"session:123","value":{...},"ts":1709...}   │
 │   {"op":"put","key":"facts:123","value":{...},"ts":1709...}     │
 │   {"op":"delete","key":"session:old","ts":1709...}              │
 │                                                                  │
 │ COMPACT (periodic maintenance):                                 │
 │   bank.compact()                                                │
 │     → rewrites JSONL with only current live state               │
 │     → removes old put/delete ops for same keys                  │
 │                                                                  │
 └──────────────────────────────────────────────────────────────────┘
```

**Two keys per session:**

```
 session:123456789  → {turns, summary, summary_covers_up_to, metadata, ...}
 facts:123456789    → {entities, relations, facts, extraction_count, ...}
```

---

## Configuration Priority Chain

```
 memory.yaml (source of truth)
      │
      ▼
 MemoryConfig dataclass (frozen, with defaults)
      │
      ▼
 CLI --flags (override at runtime)

 Example:
   memory.yaml:  max_context_tokens: 2000
   CLI:          --max-context-tokens 800
   Effective:    800

 Available CLI overrides:
   --compressor         llm_summary | llmlingua | hybrid
   --max-context-tokens int
   --max-recent-turns   int
   --backend            memorydb | redis
   --data-dir           path
   --expand             auto | confirm | off
```

---

## CrewAI Prompt Assembly

For every task, CrewAI assembles this prompt and sends it to the LLM:

```
 ┌───────── WHAT CREWAI SENDS TO LLM ──────────────────────────────┐
 │                                                                   │
 │ ┌─ messages[0]: system ─────────────────────────────────────────┐ │
 │ │ You are Web Researcher.                    ← agents.yaml role │ │
 │ │ You use web search tools...                ← agents.yaml      │ │
 │ │ CRITICAL WORKFLOW: 1) Call ONCE...            backstory       │ │
 │ │ Your personal goal is: Search the web...   ← agents.yaml goal │ │
 │ └───────────────────────────────────────────────────────────────┘ │
 │                                                                   │
 │ ┌─ messages[1]: user ───────────────────────────────────────────┐ │
 │ │ [Known entities]                           ← Tier 1 facts     │ │
 │ │ - Nguyễn Văn A [person] (student_id=...)                     │ │
 │ │ [Known relations]                                             │ │
 │ │ - nguyen_van_a → scored → 8.5                                │ │
 │ │ [Known facts]                                                 │ │
 │ │ - retake_policy: Dưới 4.0 được thi lại                       │ │
 │ │                                                               │ │
 │ │ [Conversation history summary]             ← Tier 3 summary   │ │
 │ │ User hỏi điểm SV, sau đó hỏi quy chế...                     │ │
 │ │                                                               │ │
 │ │ [Recent conversation]                      ← Tier 2 recent    │ │
 │ │ User: Cái JIT hoạt động thế nào?                             │ │
 │ │ Assistant: JIT trong Python 3.13...                           │ │
 │ │                                                               │ │
 │ │ Current question: Có benchmark không?      ← tasks.yaml       │ │
 │ │                                                               │ │
 │ │ Steps: 1) Call web_search_deep...          ← tasks.yaml       │ │
 │ │ IMPORTANT: Do NOT call tools more than once                   │ │
 │ │                                                               │ │
 │ │ Expected: JSON with findings and sources   ← tasks.yaml       │ │
 │ └───────────────────────────────────────────────────────────────┘ │
 │                                                                   │
 │ ┌─ tools[] ─────────────────────────────────────────────────────┐ │
 │ │ [{"name":"web_search_deep",                ← @tool decorator  │ │
 │ │   "description":"Tavily Search...",        ← docstring        │ │
 │ │   "parameters":{                           ← function sig     │ │
 │ │     "query":{"type":"string"},                                │ │
 │ │     "max_results":{"type":"int","default":3}                  │ │
 │ │   }},                                                         │ │
 │ │  {"name":"web_crawl_url",...},                                │ │
 │ │  {"name":"web_search_then_crawl",...}]                        │ │
 │ └───────────────────────────────────────────────────────────────┘ │
 │                                                                   │
 │ POST /v1/chat/completions → LMStudio/vLLM                       │
 └───────────────────────────────────────────────────────────────────┘
```

**Key insight: defaults in Python @tool signature = defaults LLM uses. tools.yaml is NOT read by LLM.**

---

## ReAct Loop Inside CrewAI

```
 crew.kickoff()
 │
 │ For each Task in sequential order:
 │
 ├── TASK: web_task (agent: web_researcher, max_iter=3)
 │   │
 │   │  iteration = 0
 │   │
 │   │  ┌── LLM call ─────────────────────────────────┐
 │   │  │ Send: system + user + tools                  │
 │   │  │ Recv: tool_call web_search_deep(query=...)   │
 │   │  └──────────────────────────────────────────────┘
 │   │       │
 │   │       ▼ CrewAI intercepts tool_call
 │   │       ▼ Executes Python function web_search_deep()
 │   │       ▼ Tavily HTTP → 3 results → JSON → _truncate(8000 chars)
 │   │
 │   │  iteration = 1
 │   │
 │   │  ┌── LLM call ─────────────────────────────────┐
 │   │  │ Send: previous messages + tool result        │
 │   │  │ Recv: "Final Answer: {findings, sources}"    │
 │   │  └──────────────────────────────────────────────┘
 │   │       │
 │   │       ▼ CrewAI sees "Final Answer" → STOP loop
 │   │       ▼ Store output as task result
 │   │
 │   │  (if LLM calls tool again instead of Final Answer:
 │   │   iteration = 2, repeat. If iteration >= max_iter=3: force stop)
 │   │
 │   output_task_1 = "{findings: [...], sources: [...]}"
 │   │
 │   │  auto-injected into next task's context
 │   │
 ├── TASK: compose_task (agent: composer, max_iter=2)
 │   │
 │   │  ┌── LLM call ─────────────────────────────────┐
 │   │  │ Send: task desc + output_task_1 as context   │
 │   │  │ Recv: "Final Answer: Vietnamese text..."     │
 │   │  └──────────────────────────────────────────────┘
 │   │
 │   output_final = "Python 3.13 chính thức phát hành..."
 │
 └── crew.kickoff() returns CrewOutput(raw=output_final)
```

---

## CLI Usage Reference

```bash
# ── Interactive chat ──
python -m school_agents.run_chat -s 123456789 -i
python -m school_agents.run_chat -s 123456789 -i --debug 2>debug.log

# ── Single query ──
python -m school_agents.run_chat -s 123456789 -q "Python 3.13?"

# ── Override config at runtime ──
python -m school_agents.run_chat -s 123456789 -i --max-context-tokens 800
python -m school_agents.run_chat -s 123456789 -i --compressor llmlingua
python -m school_agents.run_chat -s 123456789 -i --expand off
python -m school_agents.run_chat -s 123456789 -i --expand auto

# ── Session management ──
python -m school_agents.run_chat --list-sessions
python -m school_agents.run_chat -s 123456789 --stats
python -m school_agents.run_chat -s 123456789 --clear

# ── REPL commands (inside interactive mode) ──
# You: stats           show memory statistics
# You: facts           show extracted knowledge graph
# You: clear           reset session
# You: expand <query>  test query expansion without searching
# You: quit            exit
```

---

## Key Design Decisions

**1. Facts (Tier 1) are NEVER compressed or truncated.**
Structured data (entity IDs, scores, names) cannot survive summarization.
Summary says "user asked about grades" but facts keep "Toán=8.5".

**2. Compression only triggers when needed.**
`_needs_compression()` checks if unsummarized turns exceed `max_context_tokens`.
No wasted LLM calls on short conversations.

**3. Flush after every turn, not just on close.**
Other terminals/processes see current state immediately via JSONL.

**4. Tool defaults live in Python, not YAML.**
LLM reads `@tool` function signatures. `tools.yaml` is for API keys only.
`max_results=3`, `include_raw_content=False` in code = what LLM uses.

**5. Agent backstory is the primary loop-prevention mechanism.**
"Call tool ONCE → Final Answer" in backstory + `max_iter=3` safety net.
Local models (Qwen3) don't follow ReAct format without explicit instructions.

**6. Query expansion is opt-in interactive by default.**
`mode: "confirm"` lets user see and edit expanded queries before spending
Tavily API calls. `mode: "auto"` for API/non-interactive use.

**7. Reciprocal Rank Fusion over simple dedup.**
URLs appearing in multiple query results get higher scores.
k=60 constant ensures no single high rank dominates.
