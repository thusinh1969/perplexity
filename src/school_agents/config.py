from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_tokens: int = 4096
    structured_max_tokens: int = 16384  # for fact extraction, query expansion, compression

@dataclass(frozen=True)
class MemoryConfig:
    backend: str = "memorydb"              # "memorydb" or "redis"
    data_dir: str = "./data/memory"
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "conv:"
    redis_ttl: int = 7200
    # Conversation
    max_recent_turns: int = 4
    max_context_tokens: int = 2000
    enable_facts: bool = True
    # Compression
    compressor: str = "llm_summary"        # "llm_summary" | "llmlingua" | "hybrid"
    # Query expansion
    expand_enabled: bool = True
    expand_mode: str = "confirm"           # "auto" | "confirm"
    expand_max_queries: int = 3
    expand_max_results_per_query: int = 3
    # LLMLingua
    lingua_target_ratio: float = 0.4
    lingua_device: str = "cpu"
    lingua_model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    tools: Dict[str, Any]
    agents: Dict[str, Any]
    tasks: Dict[str, Any]
    memory: MemoryConfig = field(default_factory=MemoryConfig)

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(root: str | Path) -> AppConfig:
    root = Path(root)
    llm_raw = load_yaml(root / "llm.yaml")["llm"]
    tools = load_yaml(root / "tools.yaml")
    agents = load_yaml(root / "agents.yaml")
    tasks = load_yaml(root / "tasks.yaml")

    llm = LLMConfig(
        model=llm_raw["model"],
        base_url=llm_raw["base_url"],
        api_key=str(llm_raw.get("api_key", "")),
        temperature=float(llm_raw.get("temperature", 1.0)),
        top_p=float(llm_raw.get("top_p", 0.95)),
        top_k=int(llm_raw.get("top_k", 40)),
        repetition_penalty=float(llm_raw.get("repetition_penalty", 1.1)),
        max_tokens=int(llm_raw.get("max_tokens", 4096)),
        structured_max_tokens=int(llm_raw.get("structured_max_tokens", 16384)),
    )

    # Memory config (optional file — uses defaults if missing)
    memory_path = root / "memory.yaml"
    if memory_path.exists():
        m = load_yaml(memory_path).get("memory", {})
        redis_cfg = m.get("redis", {})
        lingua_cfg = m.get("llmlingua", {})
        expand_cfg = m.get("query_expand", {})
        memory = MemoryConfig(
            backend=m.get("backend", "memorydb"),
            data_dir=m.get("data_dir", "./data/memory"),
            redis_url=redis_cfg.get("url", "redis://localhost:6379/0"),
            redis_prefix=redis_cfg.get("prefix", "conv:"),
            redis_ttl=int(redis_cfg.get("ttl", 7200)),
            max_recent_turns=int(m.get("max_recent_turns", 4)),
            max_context_tokens=int(m.get("max_context_tokens", 2000)),
            enable_facts=bool(m.get("enable_facts", True)),
            compressor=m.get("compressor", "llm_summary"),
            expand_enabled=bool(expand_cfg.get("enabled", True)),
            expand_mode=expand_cfg.get("mode", "confirm"),
            expand_max_queries=int(expand_cfg.get("max_queries", 3)),
            expand_max_results_per_query=int(expand_cfg.get("max_results_per_query", 3)),
            lingua_target_ratio=float(lingua_cfg.get("target_ratio", 0.4)),
            lingua_device=lingua_cfg.get("device", "cpu"),
            lingua_model=lingua_cfg.get("model", MemoryConfig.lingua_model),
        )
    else:
        memory = MemoryConfig()

    return AppConfig(llm=llm, tools=tools, agents=agents, tasks=tasks, memory=memory)
