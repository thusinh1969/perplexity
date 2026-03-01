from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import os
import re
import yaml


# ── Env var resolution ──────────────────────────────────────────────

def _load_dotenv(project_root: Path | None = None) -> None:
    """Load .env file into os.environ (no dependency on python-dotenv)."""
    # Search order: project_root/.env → cwd/.env
    candidates = []
    if project_root:
        candidates.append(project_root / ".env")
    candidates.append(Path.cwd() / ".env")

    for env_path in candidates:
        if env_path.is_file():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:  # don't override existing env
                        os.environ[key] = value
            return  # loaded first found


_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")

def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${VAR} patterns in strings/dicts/lists."""
    if isinstance(value, str):
        def _replacer(m):
            var_name = m.group(1)
            env_val = os.environ.get(var_name, "")
            if not env_val:
                import logging
                logging.getLogger("school_agents.config").warning(
                    "⚠️  Env var %s not set (referenced in config). Using empty string.", var_name
                )
            return env_val
        return _ENV_PATTERN.sub(_replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value

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

    # Load .env from project root (2 levels up from config dir)
    project_root = root.parent.parent if root.name == "config" else root.parent
    _load_dotenv(project_root)

    # Load and resolve env vars in all yaml files
    llm_raw = _resolve_env(load_yaml(root / "llm.yaml")["llm"])
    tools = _resolve_env(load_yaml(root / "tools.yaml"))
    agents = load_yaml(root / "agents.yaml")   # no secrets in agents
    tasks = load_yaml(root / "tasks.yaml")     # no secrets in tasks

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
