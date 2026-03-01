"""
memory_bank.py — Persistent storage for conversation sessions.

Two backends:
  - MemoryDB:        In-memory dict + JSONL flush to disk (dev/test, also usable in small prod)
  - RedisMemoryBank: Redis-backed (production at scale)

Both implement the same MemoryBankBase interface:
  get(key)           → dict | None
  put(key, value)    → None
  delete(key)        → bool
  search(prefix)     → list[str]       # key search by prefix
  list_keys()        → list[str]
  close()            → None
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger("school_agents.memory_bank")


# ╭──────────────────────────────────────────────────────────────────╮
# │                      ABSTRACT BASE                               │
# ╰──────────────────────────────────────────────────────────────────╯

class MemoryBankBase(ABC):
    """Interface for conversation memory storage."""

    @abstractmethod
    def get(self, key: str) -> dict | None:
        """Retrieve a value by key. Returns None if not found."""

    @abstractmethod
    def put(self, key: str, value: dict) -> None:
        """Store a value. Overwrites if key exists."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""

    @abstractmethod
    def search(self, prefix: str) -> list[str]:
        """Return all keys matching a prefix."""

    @abstractmethod
    def list_keys(self) -> list[str]:
        """Return all keys."""

    @abstractmethod
    def close(self) -> None:
        """Flush/cleanup. Called on shutdown."""

    def flush(self) -> int:
        """Flush pending writes to storage. Returns ops written. Default: noop."""
        return 0


# ╭──────────────────────────────────────────────────────────────────╮
# │                   MemoryDB (In-Memory + JSONL)                   │
# ╰──────────────────────────────────────────────────────────────────╯

class MemoryDB(MemoryBankBase):
    """
    In-memory key-value store with JSONL persistence.

    Data lives in a dict for fast access. On close() or flush(),
    each session is appended to a JSONL file so state survives restarts.
    On init, existing JSONL is replayed to restore state.

    File format (one JSON object per line):
        {"op":"put","key":"session:123","value":{...},"ts":1234567890.0}
        {"op":"delete","key":"session:123","ts":1234567890.1}

    Usage:
        db = MemoryDB(data_dir="./data/memory")
        db.put("session:123", {"turns": [...], "summary": "..."})
        db.get("session:123")  # → dict
        db.close()             # flushes dirty keys to JSONL
    """

    def __init__(
        self,
        data_dir: str | Path = "./data/memory",
        filename: str = "sessions.jsonl",
        auto_flush_interval: float = 0,  # 0 = no auto-flush, >0 = seconds
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._data_dir / filename
        self._store: dict[str, dict] = {}
        self._dirty: set[str] = set()  # keys modified since last flush
        self._lock = threading.Lock()

        # Replay existing JSONL to restore state
        self._replay()

        # Optional auto-flush background thread
        self._auto_flush_thread = None
        if auto_flush_interval > 0:
            self._auto_flush_stop = threading.Event()
            self._auto_flush_thread = threading.Thread(
                target=self._auto_flush_loop,
                args=(auto_flush_interval,),
                daemon=True,
            )
            self._auto_flush_thread.start()

        log.info(
            "[MemoryDB] Initialized: %d sessions loaded from %s",
            len(self._store), self._filepath,
        )

    # ── Public API ──

    def get(self, key: str) -> dict | None:
        with self._lock:
            val = self._store.get(key)
            # Return a copy to prevent external mutation
            return json.loads(json.dumps(val)) if val is not None else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._store[key] = json.loads(json.dumps(value))  # deep copy
            self._dirty.add(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            if existed:
                self._dirty.add(key)  # mark for flush (will write delete op)
            return existed

    def search(self, prefix: str) -> list[str]:
        with self._lock:
            return [k for k in self._store if k.startswith(prefix)]

    def list_keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def flush(self) -> int:
        """Write dirty keys to JSONL. Returns number of ops written."""
        with self._lock:
            if not self._dirty:
                return 0
            ops_written = 0
            with open(self._filepath, "a", encoding="utf-8") as f:
                for key in self._dirty:
                    if key in self._store:
                        op = {
                            "op": "put",
                            "key": key,
                            "value": self._store[key],
                            "ts": time.time(),
                        }
                    else:
                        op = {"op": "delete", "key": key, "ts": time.time()}
                    f.write(json.dumps(op, ensure_ascii=False) + "\n")
                    ops_written += 1
            self._dirty.clear()
            log.debug("[MemoryDB] Flushed %d ops to %s", ops_written, self._filepath)
            return ops_written

    def close(self) -> None:
        """Flush all dirty data and stop background threads."""
        if self._auto_flush_thread is not None:
            self._auto_flush_stop.set()
            self._auto_flush_thread.join(timeout=5)
        n = self.flush()
        log.info("[MemoryDB] Closed. Flushed %d final ops.", n)

    def compact(self) -> int:
        """
        Rewrite JSONL with only current state (remove old ops).
        Call periodically to keep file small.
        Returns number of live keys written.
        """
        with self._lock:
            self._dirty.clear()  # compaction covers everything
            tmp = self._filepath.with_suffix(".jsonl.tmp")
            count = 0
            with open(tmp, "w", encoding="utf-8") as f:
                for key, value in self._store.items():
                    op = {"op": "put", "key": key, "value": value, "ts": time.time()}
                    f.write(json.dumps(op, ensure_ascii=False) + "\n")
                    count += 1
            tmp.replace(self._filepath)
            log.info("[MemoryDB] Compacted: %d live keys written.", count)
            return count

    # ── Internal ──

    def _replay(self) -> None:
        """Replay JSONL to rebuild in-memory state."""
        if not self._filepath.exists():
            return
        count = 0
        with open(self._filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    op = json.loads(line)
                    if op["op"] == "put":
                        self._store[op["key"]] = op["value"]
                    elif op["op"] == "delete":
                        self._store.pop(op["key"], None)
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    log.warning("[MemoryDB] Skipping corrupt line %d: %s", line_num, e)
        log.debug("[MemoryDB] Replayed %d ops from %s", count, self._filepath)

    def _auto_flush_loop(self, interval: float) -> None:
        while not self._auto_flush_stop.wait(timeout=interval):
            self.flush()


# ╭──────────────────────────────────────────────────────────────────╮
# │                  RedisMemoryBank (Production)                    │
# ╰──────────────────────────────────────────────────────────────────╯

class RedisMemoryBank(MemoryBankBase):
    """
    Redis-backed memory bank for production.

    Each key is stored as a Redis hash. Sessions have a configurable TTL.
    Requires: pip install redis

    Usage:
        db = RedisMemoryBank(url="redis://localhost:6379", prefix="conv:", ttl=3600)
        db.put("session:123", {"turns": [...]})
        db.get("session:123")
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "conv:",
        ttl: int = 7200,  # 2 hours default
    ):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "RedisMemoryBank requires 'redis' package. "
                "Install with: pip install redis"
            )
        self._prefix = prefix
        self._ttl = ttl
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        # Verify connection
        self._redis.ping()
        log.info("[RedisMemoryBank] Connected to %s (prefix=%s, ttl=%ds)", url, prefix, ttl)

    def _fk(self, key: str) -> str:
        """Full key with prefix."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> dict | None:
        raw = self._redis.get(self._fk(key))
        if raw is None:
            return None
        return json.loads(raw)

    def put(self, key: str, value: dict) -> None:
        self._redis.setex(
            self._fk(key),
            self._ttl,
            json.dumps(value, ensure_ascii=False),
        )

    def delete(self, key: str) -> bool:
        return self._redis.delete(self._fk(key)) > 0

    def search(self, prefix: str) -> list[str]:
        pattern = f"{self._prefix}{prefix}*"
        keys = self._redis.keys(pattern)
        plen = len(self._prefix)
        return [k[plen:] for k in keys]

    def list_keys(self) -> list[str]:
        return self.search("")

    def close(self) -> None:
        self._redis.close()
        log.info("[RedisMemoryBank] Connection closed.")
