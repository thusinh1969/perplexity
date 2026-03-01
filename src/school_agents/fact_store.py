"""
fact_store.py — Lightweight knowledge graph extracted from conversations.

Domain-agnostic. Extracts and maintains:
  - Entities:      {id, type, name, attributes}
  - Relations:     {subject, predicate, object}
  - Facts:         {key, value}

All extracted by LLM after each turn, merged/deduped automatically.
Injected into prompt as structured Tier-1 context that NEVER gets compressed.

Usage:
    store = FactStore(bank, session_id="123")
    store.load()

    # After each turn, extract new knowledge
    store.extract(
        llm_client=oai,
        model="qwen/qwen3-coder-next",
        conversation_text="User: điểm Nguyễn Văn A?\nAssistant: Toán 8.5...",
    )

    # Build context for prompt injection
    context = store.to_context_string()
    # → "[Known entities]\n- Nguyễn Văn A (person): student_id=20210345\n..."

    store.save()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import json_repair

from .memory_bank import MemoryBankBase
from .llm_utils import strip_think_tags, extract_json, NO_THINK_SYSTEM, no_think_extra_body

log = logging.getLogger("school_agents.fact_store")


# ── Data Models ──────────────────────────────────────────────────────

@dataclass
class Entity:
    """A named thing mentioned in conversation."""
    id: str              # normalized key, e.g. "nguyen_van_a", "python_3.13"
    type: str            # e.g. "person", "software", "organization", "policy", "course"
    name: str            # display name
    attributes: dict[str, str] = field(default_factory=dict)  # key-value pairs
    last_updated: float = 0.0

    def merge(self, other: "Entity") -> None:
        """Merge another entity's attributes into this one (newer wins)."""
        self.attributes.update(other.attributes)
        self.name = other.name or self.name
        self.type = other.type or self.type
        self.last_updated = max(self.last_updated, other.last_updated)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "name": self.name,
            "attributes": self.attributes, "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(**d)


@dataclass
class Relation:
    """A relationship between two entities."""
    subject: str         # entity id
    predicate: str       # e.g. "enrolled_in", "scored", "authored", "is_part_of"
    object: str          # entity id or literal value
    detail: str = ""     # optional extra info
    last_updated: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.object}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "predicate": self.predicate,
            "object": self.object, "detail": self.detail,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(**d)


@dataclass
class Fact:
    """A standalone piece of knowledge not tied to a specific entity."""
    key: str             # normalized topic, e.g. "retake_policy", "python_3.13_release_date"
    value: str           # the fact itself
    confidence: str = "stated"  # "stated" (user said), "retrieved" (tool found), "inferred"
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key, "value": self.value,
            "confidence": self.confidence, "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**d)


# ── Extraction Prompt ────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
Extract key knowledge from the conversation below as a JSON object.

Format:
{{
  "entities": [{{"id": "snake_case", "type": "person|org|software|policy|course|location|event|other", "name": "Name", "attributes": {{"key": "value"}}}}],
  "relations": [{{"subject": "id", "predicate": "verb", "object": "id_or_value", "detail": ""}}],
  "facts": [{{"key": "topic", "value": "concise fact", "confidence": "stated|retrieved|inferred"}}]
}}

CRITICAL rules:
- Max 10 entities, max 10 relations, max 10 facts per extraction.
- Per entity: max 5 attributes. Combine related scores into one attribute.
- Keep total JSON under 800 tokens. Be concise but thorough.
- Same language as conversation. Merge with existing knowledge (don't duplicate).
- Return ONLY JSON, no markdown fences, no explanation.
- If nothing new: {{"entities":[],"relations":[],"facts":[]}}
{existing_knowledge_block}
Conversation:
{conversation_text}

JSON:"""


# ── Fact Store ───────────────────────────────────────────────────────

class FactStore:
    """
    Lightweight knowledge graph for a conversation session.

    Stores entities, relations, and facts extracted from conversation turns.
    Persists via MemoryBankBase alongside ConversationMemory.
    """

    def __init__(self, bank: MemoryBankBase, session_id: str):
        self._bank = bank
        self.session_id = session_id
        self.entities: dict[str, Entity] = {}      # id → Entity
        self.relations: dict[str, Relation] = {}    # key → Relation
        self.facts: dict[str, Fact] = {}            # key → Fact
        self._extraction_count: int = 0

    # ── Persistence ──

    def load(self) -> None:
        """Load from memory bank."""
        data = self._bank.get(f"facts:{self.session_id}")
        if data is None:
            log.info("[facts:%s] New session (no prior facts).", self.session_id)
            return
        for e in data.get("entities", []):
            ent = Entity.from_dict(e)
            self.entities[ent.id] = ent
        for r in data.get("relations", []):
            rel = Relation.from_dict(r)
            self.relations[rel.key] = rel
        for f in data.get("facts", []):
            fact = Fact.from_dict(f)
            self.facts[fact.key] = fact
        self._extraction_count = data.get("extraction_count", 0)
        log.info(
            "[facts:%s] Loaded: %d entities, %d relations, %d facts",
            self.session_id, len(self.entities), len(self.relations), len(self.facts),
        )

    def save(self) -> None:
        """Persist to memory bank."""
        data = {
            "session_id": self.session_id,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "facts": [f.to_dict() for f in self.facts.values()],
            "extraction_count": self._extraction_count,
            "updated_at": time.time(),
        }
        self._bank.put(f"facts:{self.session_id}", data)
        log.debug("[facts:%s] Saved.", self.session_id)

    # ── Extraction ──

    def extract(
        self,
        llm_client: Any,
        model: str,
        conversation_text: str,
        max_tokens: int = 16384,
        extra_body: dict | None = None,
    ) -> dict:
        """
        Extract entities/relations/facts from conversation text using LLM.

        Args:
            llm_client: OpenAI-compatible client
            model: Model name
            conversation_text: Recent turn(s) to extract from
            max_tokens: Max tokens for extraction response

        Returns:
            Dict with counts: {"entities": N, "relations": N, "facts": N}
        """
        existing_block = ""
        if self.entities or self.facts:
            existing_block = f"Existing knowledge (update/merge, don't duplicate):\n{self.to_context_string()}\n"

        prompt = EXTRACTION_PROMPT.format(
            existing_knowledge_block=existing_block,
            conversation_text=conversation_text,
        )

        log.info("[facts:%s] Extracting from %d chars of conversation...",
                 self.session_id, len(conversation_text))

        raw = ""
        try:
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NO_THINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=0.95,
                extra_body=no_think_extra_body(extra_body),
            )
            raw = resp.choices[0].message.content.strip()
            log.debug("[facts:%s] Raw LLM output (%d chars): %s",
                      self.session_id, len(raw), raw[:300])

            # Extract JSON from potentially messy output (handles thinking text, fences, etc.)
            cleaned = extract_json(raw)
            if not cleaned:
                log.warning("[facts:%s] No JSON found in LLM output", self.session_id)
                return {"entities": 0, "relations": 0, "facts": 0}

            extracted = json_repair.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.error("[facts:%s] Failed to parse extraction JSON: %s\nRaw (%d chars): %s",
                      self.session_id, exc, len(raw), raw[:500])
            return {"entities": 0, "relations": 0, "facts": 0}
        except Exception as exc:
            log.error("[facts:%s] Extraction LLM call failed: %s", self.session_id, exc, exc_info=True)
            return {"entities": 0, "relations": 0, "facts": 0}

        counts = self._merge(extracted)
        self._extraction_count += 1
        log.info("[facts:%s] Extracted: +%d entities, +%d relations, +%d facts (total: %d/%d/%d)",
                 self.session_id,
                 counts["entities"], counts["relations"], counts["facts"],
                 len(self.entities), len(self.relations), len(self.facts))
        return counts

    def _merge(self, extracted: dict) -> dict:
        """Merge extracted data into existing store. Returns counts of new/updated items."""
        now = time.time()
        counts = {"entities": 0, "relations": 0, "facts": 0}

        for e_raw in extracted.get("entities", []):
            try:
                e = Entity(
                    id=e_raw["id"],
                    type=e_raw.get("type", "other"),
                    name=e_raw.get("name", e_raw["id"]),
                    attributes=e_raw.get("attributes", {}),
                    last_updated=now,
                )
                if e.id in self.entities:
                    self.entities[e.id].merge(e)
                else:
                    self.entities[e.id] = e
                counts["entities"] += 1
            except (KeyError, TypeError) as exc:
                log.debug("[facts] Skipping malformed entity: %s", exc)

        for r_raw in extracted.get("relations", []):
            try:
                r = Relation(
                    subject=r_raw["subject"],
                    predicate=r_raw["predicate"],
                    object=r_raw["object"],
                    detail=r_raw.get("detail", ""),
                    last_updated=now,
                )
                self.relations[r.key] = r  # overwrite if same key
                counts["relations"] += 1
            except (KeyError, TypeError) as exc:
                log.debug("[facts] Skipping malformed relation: %s", exc)

        for f_raw in extracted.get("facts", []):
            try:
                f = Fact(
                    key=f_raw["key"],
                    value=f_raw["value"],
                    confidence=f_raw.get("confidence", "stated"),
                    last_updated=now,
                )
                self.facts[f.key] = f  # overwrite if same key
                counts["facts"] += 1
            except (KeyError, TypeError) as exc:
                log.debug("[facts] Skipping malformed fact: %s", exc)

        return counts

    # ── Context Building ──

    def to_context_string(self) -> str:
        """
        Format stored knowledge as a string for prompt injection.

        This is Tier-1 context — structured, NEVER compressed, NEVER truncated.
        """
        if not self.entities and not self.relations and not self.facts:
            return ""

        parts = []

        # Entities
        if self.entities:
            ent_lines = []
            for e in sorted(self.entities.values(), key=lambda x: x.last_updated, reverse=True):
                attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
                attr_str = f" ({attrs})" if attrs else ""
                ent_lines.append(f"- {e.name} [{e.type}]{attr_str}")
            parts.append("[Known entities]\n" + "\n".join(ent_lines))

        # Relations
        if self.relations:
            rel_lines = []
            for r in sorted(self.relations.values(), key=lambda x: x.last_updated, reverse=True):
                detail = f" — {r.detail}" if r.detail else ""
                rel_lines.append(f"- {r.subject} → {r.predicate} → {r.object}{detail}")
            parts.append("[Known relations]\n" + "\n".join(rel_lines))

        # Facts
        if self.facts:
            fact_lines = []
            for f in sorted(self.facts.values(), key=lambda x: x.last_updated, reverse=True):
                fact_lines.append(f"- {f.key}: {f.value}")
            parts.append("[Known facts]\n" + "\n".join(fact_lines))

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "facts": len(self.facts),
            "extractions": self._extraction_count,
            "context_chars": len(self.to_context_string()),
            "context_tokens_est": len(self.to_context_string()) // 4,
        }

    def clear(self) -> None:
        """Reset all knowledge."""
        self.entities.clear()
        self.relations.clear()
        self.facts.clear()
        self._extraction_count = 0
        self.save()

    # ── Query helpers ──

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def find_entities(self, type_filter: str = "") -> list[Entity]:
        """Find entities, optionally filtered by type."""
        if not type_filter:
            return list(self.entities.values())
        return [e for e in self.entities.values() if e.type == type_filter]

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        """Get all relations where entity is subject or object."""
        return [
            r for r in self.relations.values()
            if r.subject == entity_id or r.object == entity_id
        ]
