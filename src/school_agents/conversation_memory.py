"""
conversation_memory.py — Multi-turn conversation state manager.

Manages:
  - Turn history (user/assistant pairs)
  - Automatic compression when context grows too large
  - Context building for CrewAI task injection
  - Persistence via MemoryBankBase (MemoryDB or Redis)

Lifecycle:
  1. load(session_id)          → restore from bank
  2. add_user_turn(text)       → append user message
  3. build_context(query)      → create context string for CrewAI (compress if needed)
  4. add_assistant_turn(text)  → append assistant response
  5. save()                    → persist to bank
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .memory_bank import MemoryBankBase
from .context_compressor import ContextCompressor
from .fact_store import FactStore

log = logging.getLogger("school_agents.conversation_memory")


@dataclass
class Turn:
    """Single conversation turn."""
    role: str               # "user" or "assistant"
    content: str
    timestamp: float = 0.0
    routes: list[str] | None = None  # what routes were used (assistant turns only)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def token_estimate(self) -> int:
        """Rough token count (1 token ≈ 4 chars)."""
        return len(self.content) // 4

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.routes:
            d["routes"] = self.routes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", 0.0),
            routes=d.get("routes"),
        )


class ConversationMemory:
    """
    Multi-turn conversation memory with automatic compression.

    Args:
        session_id: Unique session identifier
        bank: Storage backend (MemoryDB or RedisMemoryBank)
        compressor: Context compression strategy
        max_recent_turns: Number of recent turns to keep verbatim
        max_context_tokens: Max tokens for history portion of context
    """

    def __init__(
        self,
        session_id: str,
        bank: MemoryBankBase,
        compressor: ContextCompressor,
        max_recent_turns: int = 4,      # 2 user/assistant pairs
        max_context_tokens: int = 2000,  # budget for history in prompt
        enable_facts: bool = True,       # Tier-1 fact extraction
    ):
        self.session_id = session_id
        self._bank = bank
        self._compressor = compressor
        self.max_recent_turns = max_recent_turns
        self.max_context_tokens = max_context_tokens
        self.enable_facts = enable_facts

        # State
        self.turns: list[Turn] = []
        self.summary: str = ""
        self.summary_covers_up_to: int = 0  # index in self.turns
        self.metadata: dict[str, Any] = {}  # arbitrary session metadata
        self._created_at: float = time.time()
        self._turn_count: int = 0  # total turns ever (including compressed)

        # Tier-1: Knowledge graph
        self.facts = FactStore(bank, session_id) if enable_facts else None

        # Load existing state
        self._load()

    # ╭────────────────── Public API ──────────────────╮

    def add_user_turn(self, content: str) -> None:
        """Record a user message."""
        self.turns.append(Turn(role="user", content=content))
        self._turn_count += 1
        log.debug("[memory:%s] +user turn (%d chars)", self.session_id, len(content))

    def add_assistant_turn(self, content: str, routes: list[str] | None = None) -> None:
        """Record an assistant response."""
        self.turns.append(Turn(role="assistant", content=content, routes=routes))
        self._turn_count += 1
        log.debug("[memory:%s] +assistant turn (%d chars)", self.session_id, len(content))

    def build_context(self, current_query: str = "") -> str:
        """
        Build 3-tier context string to inject into CrewAI task descriptions.

        Tier 1: Known facts (structured, NEVER compressed)
        Tier 2: Recent turns (verbatim, sliding window)
        Tier 3: Narrative summary (compressed older turns)

        Automatically compresses older turns if budget exceeded.
        """
        # Step 1: Compress if needed
        if self._needs_compression():
            self._compress(current_query)

        # Step 2: Build 3-tier context
        parts = []

        # Tier 1: Known facts (entities, relations, facts)
        if self.facts:
            facts_ctx = self.facts.to_context_string()
            if facts_ctx:
                parts.append(facts_ctx)

        # Tier 3: Summary of older turns
        if self.summary:
            parts.append(f"[Conversation history summary]\n{self.summary}")

        # Tier 2: Recent turns (verbatim, with size guard)
        recent = self.turns[self.summary_covers_up_to:]
        recent = recent[-self.max_recent_turns:]
        if recent:
            recent_text = []
            # Budget for recent turns: max_context_tokens minus summary/facts overhead
            recent_budget = self.max_context_tokens
            if self.summary:
                recent_budget -= len(self.summary) // 4
            recent_budget = max(recent_budget, 1024)  # minimum 1024 tokens

            total_recent_tokens = sum(t.token_estimate for t in recent)

            for t in recent:
                label = "User" if t.role == "user" else "Assistant"
                content = t.content
                # If recent turns collectively exceed budget, truncate each proportionally
                if total_recent_tokens > recent_budget and t.token_estimate > 0:
                    max_chars = int(len(content) * (recent_budget / total_recent_tokens))
                    max_chars = max(max_chars, 500)  # at least 500 chars per turn
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n... [truncated from {len(t.content)} chars]"
                recent_text.append(f"{label}: {content}")
            parts.append("[Recent conversation]\n" + "\n".join(recent_text))

        context = "\n\n".join(parts)

        # Log stats
        approx_tokens = len(context) // 4
        facts_info = f", facts={len(self.facts.entities)}E/{len(self.facts.relations)}R/{len(self.facts.facts)}F" if self.facts else ""
        log.info(
            "[memory:%s] Context built: %d chars (~%d tokens), "
            "%d total turns, %d summarized, %d recent%s",
            self.session_id, len(context), approx_tokens,
            len(self.turns), self.summary_covers_up_to,
            len(recent), facts_info,
        )
        return context

    def extract_facts(self, llm_client: Any, model: str, max_chars_per_turn: int = 3000,
                       extra_body: dict | None = None, max_tokens: int = 16384) -> dict:
        """
        Extract entities/relations/facts from the latest turn pair.

        Call this AFTER adding both user + assistant turns.
        Uses the last 2 turns (user question + assistant answer).
        Truncates very long turns to avoid exceeding extraction LLM context.
        """
        if not self.facts or len(self.turns) < 2:
            return {"entities": 0, "relations": 0, "facts": 0}

        # Get last user/assistant pair, truncate if needed
        recent = self.turns[-2:]
        lines = []
        for t in recent:
            label = "User" if t.role == "user" else "Assistant"
            content = t.content
            if len(content) > max_chars_per_turn:
                content = content[:max_chars_per_turn] + f"\n... [truncated, {len(t.content)} total chars]"
            lines.append(f"{label}: {content}")
        conversation_text = "\n".join(lines)

        counts = self.facts.extract(
            llm_client=llm_client,
            model=model,
            conversation_text=conversation_text,
            extra_body=extra_body,
            max_tokens=max_tokens,
        )
        return counts

    def save(self) -> None:
        """Persist current state to memory bank."""
        data = {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary,
            "summary_covers_up_to": self.summary_covers_up_to,
            "metadata": self.metadata,
            "created_at": self._created_at,
            "turn_count": self._turn_count,
            "updated_at": time.time(),
        }
        self._bank.put(f"session:{self.session_id}", data)
        if self.facts:
            self.facts.save()
        log.debug("[memory:%s] Saved (%d turns, summary=%d chars)",
                  self.session_id, len(self.turns), len(self.summary))

    def clear(self) -> None:
        """Reset conversation (keep session_id)."""
        self.turns.clear()
        self.summary = ""
        self.summary_covers_up_to = 0
        self._turn_count = 0
        if self.facts:
            self.facts.clear()
        self.save()
        log.info("[memory:%s] Cleared.", self.session_id)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0 and not self.summary

    def get_stats(self) -> dict:
        """Return memory stats for debugging."""
        stats = {
            "session_id": self.session_id,
            "total_turns": self._turn_count,
            "live_turns": len(self.turns),
            "summarized_up_to": self.summary_covers_up_to,
            "summary_chars": len(self.summary),
            "summary_tokens_est": len(self.summary) // 4,
            "recent_turns_tokens_est": sum(
                t.token_estimate for t in self.turns[self.summary_covers_up_to:]
            ),
        }
        if self.facts:
            stats["facts"] = self.facts.get_stats()
        return stats

    # ╭────────────────── Internal ──────────────────╮

    def _load(self) -> None:
        """Load state from memory bank."""
        data = self._bank.get(f"session:{self.session_id}")
        if data is None:
            log.info("[memory:%s] New session (no prior state).", self.session_id)
            if self.facts:
                self.facts.load()  # will find nothing, but consistent
            return

        self.turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        self.summary = data.get("summary", "")
        self.summary_covers_up_to = data.get("summary_covers_up_to", 0)
        self.metadata = data.get("metadata", {})
        self._created_at = data.get("created_at", time.time())
        self._turn_count = data.get("turn_count", len(self.turns))

        if self.facts:
            self.facts.load()

        facts_info = ""
        if self.facts:
            fs = self.facts.get_stats()
            facts_info = f", facts={fs['entities']}E/{fs['relations']}R/{fs['facts']}F"

        log.info(
            "[memory:%s] Loaded: %d turns, summary=%d chars, total_ever=%d%s",
            self.session_id, len(self.turns), len(self.summary),
            self._turn_count, facts_info,
        )

    def _needs_compression(self) -> bool:
        """Check if unsummarized turns exceed token budget."""
        unsummarized = self.turns[self.summary_covers_up_to:]
        if len(unsummarized) <= self.max_recent_turns:
            # Even with few turns, check if they're individually huge
            # (e.g., 1 assistant turn with 30K chars of web search results)
            # If so, we can't compress (nothing older to compress), but
            # build_context will truncate them. So return False.
            return False
        total_tokens = sum(t.token_estimate for t in unsummarized)
        return total_tokens > self.max_context_tokens

    def _compress(self, current_query: str) -> None:
        """Compress older turns into summary."""
        unsummarized = self.turns[self.summary_covers_up_to:]
        # Keep max_recent_turns at the end, compress the rest
        to_compress = unsummarized[:-self.max_recent_turns]

        if not to_compress:
            return

        # Format turns for compressor
        history_text = "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
            for t in to_compress
        )

        log.info(
            "[memory:%s] Compressing %d turns (%d chars) into summary...",
            self.session_id, len(to_compress), len(history_text),
        )

        new_summary = self._compressor.compress(
            history_text=history_text,
            current_query=current_query,
            existing_summary=self.summary,
        )

        self.summary = new_summary
        self.summary_covers_up_to += len(to_compress)

        log.info(
            "[memory:%s] Compression done. Summary now %d chars (~%d tokens). "
            "Covers up to turn %d.",
            self.session_id, len(self.summary), len(self.summary) // 4,
            self.summary_covers_up_to,
        )
