"""
fact_store.py — Evidence-only knowledge graph extracted from verified sources.

TRUST MODEL: Facts ONLY come from verified evidence (web search, RAG, APIs).
             NEVER from LLM conversation output (can hallucinate).
             NEVER from user statements (can be wrong or malicious).

Each fact tracks its source:
  - source_type: "web" | "rag" | "api"
  - source_ref:  URL or document ID
  - source_title: human-readable source name
  - turn_id:     which conversation turn produced it

Reconciliation: when a newer verified fact contradicts an older one,
the newer one REPLACES it. Priority: api > rag > web.
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

SOURCE_PRIORITY = {"web": 1, "rag": 2, "api": 3}


# ── Data Models ──────────────────────────────────────────────────────

@dataclass
class Entity:
    id: str
    type: str
    name: str
    attributes: dict[str, str] = field(default_factory=dict)
    source_type: str = "web"
    source_ref: str = ""
    source_title: str = ""
    turn_id: int = 0
    last_updated: float = 0.0

    def merge(self, other: "Entity") -> None:
        old_pri = SOURCE_PRIORITY.get(self.source_type, 0)
        new_pri = SOURCE_PRIORITY.get(other.source_type, 0)
        if new_pri > old_pri or (new_pri == old_pri and other.turn_id >= self.turn_id):
            self.attributes.update(other.attributes)
            self.name = other.name or self.name
            self.type = other.type or self.type
            self.source_type = other.source_type
            self.source_ref = other.source_ref
            self.source_title = other.source_title
            self.turn_id = other.turn_id
            self.last_updated = other.last_updated
        else:
            for k, v in other.attributes.items():
                if k not in self.attributes:
                    self.attributes[k] = v

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "name": self.name,
            "attributes": self.attributes,
            "source_type": self.source_type, "source_ref": self.source_ref,
            "source_title": self.source_title, "turn_id": self.turn_id,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    detail: str = ""
    source_type: str = "web"
    source_ref: str = ""
    source_title: str = ""
    turn_id: int = 0
    last_updated: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.object}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "predicate": self.predicate,
            "object": self.object, "detail": self.detail,
            "source_type": self.source_type, "source_ref": self.source_ref,
            "source_title": self.source_title, "turn_id": self.turn_id,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Fact:
    key: str
    value: str
    source_type: str = "web"
    source_ref: str = ""
    source_title: str = ""
    turn_id: int = 0
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key, "value": self.value,
            "source_type": self.source_type, "source_ref": self.source_ref,
            "source_title": self.source_title, "turn_id": self.turn_id,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Extraction Prompt ────────────────────────────────────────────────

EVIDENCE_EXTRACTION_PROMPT = """\
Extract key knowledge from the VERIFIED EVIDENCE below as JSON.

This evidence comes from {source_type_label}. Every fact you extract is backed by real sources.
Treat all information as VERIFIED.

Format:
{{
  "entities": [{{"id": "snake_case", "type": "person|org|software|policy|location|event|other", "name": "Name", "attributes": {{"key": "value"}}}}],
  "relations": [{{"subject": "id", "predicate": "verb", "object": "id_or_value", "detail": ""}}],
  "facts": [{{"key": "topic_snake_case", "value": "concise factual statement"}}]
}}

RULES:
- Max 10 entities, 10 relations, 10 facts.
- Per entity: max 5 attributes.
- Extract ONLY factual claims present in the evidence. Do NOT infer or speculate.
- Include specific numbers, dates, percentages when present in evidence.
- Same language as the evidence. Be concise but thorough.
- Return ONLY JSON, no markdown fences, no explanation.
- If nothing extractable: {{"entities":[],"relations":[],"facts":[]}}
{existing_knowledge_block}
User question: {user_question}

Verified evidence ({source_type_label}):
{evidence_text}

JSON:"""


# ── Fact Store ───────────────────────────────────────────────────────

class FactStore:
    """
    Evidence-only knowledge graph.

    TRUST MODEL:
    - Only extract facts from verified evidence (web, RAG, API)
    - Never from LLM output or user claims
    - Each fact carries source provenance
    - Newer verified facts replace older conflicting ones
    """

    def __init__(self, bank: MemoryBankBase, session_id: str):
        self._bank = bank
        self.session_id = session_id
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.facts: dict[str, Fact] = {}
        self._extraction_count: int = 0

    # ── Persistence ──

    def load(self) -> None:
        data = self._bank.get(f"facts:{self.session_id}")
        if data is None:
            log.info("[facts:%s] New session.", self.session_id)
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
        log.info("[facts:%s] Loaded: %dE %dR %dF",
                 self.session_id, len(self.entities), len(self.relations), len(self.facts))

    def save(self) -> None:
        data = {
            "session_id": self.session_id,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "facts": [f.to_dict() for f in self.facts.values()],
            "extraction_count": self._extraction_count,
            "updated_at": time.time(),
        }
        self._bank.put(f"facts:{self.session_id}", data)

    # ── Evidence-Based Extraction ──

    def extract_from_evidence(
        self,
        llm_client: Any,
        model: str,
        evidence_text: str,
        source_type: str,
        sources: list[dict] | None = None,
        user_question: str = "",
        turn_id: int = 0,
        max_tokens: int = 16384,
        extra_body: dict | None = None,
    ) -> dict:
        """Extract facts from VERIFIED evidence only."""
        if not evidence_text or not evidence_text.strip():
            log.info("[facts:%s] No evidence (turn %d), skipping.", self.session_id, turn_id)
            return {"entities": 0, "relations": 0, "facts": 0}

        source_labels = {
            "web": "web search results",
            "rag": "internal document search (RAG)",
            "api": "internal API data",
        }
        source_type_label = source_labels.get(source_type, source_type)

        existing_block = ""
        if self.entities or self.facts:
            existing_block = f"\nExisting knowledge (update/merge, don't duplicate):\n{self.to_context_string()}\n"

        prompt = EVIDENCE_EXTRACTION_PROMPT.format(
            source_type_label=source_type_label,
            existing_knowledge_block=existing_block,
            user_question=user_question,
            evidence_text=evidence_text[:6000],
        )

        log.info("[facts:%s] Extracting from %s evidence (%d chars, turn %d)",
                 self.session_id, source_type, len(evidence_text), turn_id)

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

            cleaned = extract_json(raw)
            if not cleaned:
                log.warning("[facts:%s] No JSON in extraction output", self.session_id)
                return {"entities": 0, "relations": 0, "facts": 0}

            extracted = json_repair.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.error("[facts:%s] JSON parse failed: %s", self.session_id, exc)
            return {"entities": 0, "relations": 0, "facts": 0}
        except Exception as exc:
            log.error("[facts:%s] Extraction failed: %s", self.session_id, exc, exc_info=True)
            return {"entities": 0, "relations": 0, "facts": 0}

        # Default source ref from first source
        default_ref = ""
        default_title = ""
        if sources:
            default_ref = sources[0].get("url", "")
            default_title = sources[0].get("title", "")

        counts = self._merge_evidence(
            extracted, source_type=source_type,
            source_ref=default_ref, source_title=default_title,
            turn_id=turn_id,
        )
        self._extraction_count += 1

        log.info("[facts:%s] +%dE +%dR +%dF (total: %d/%d/%d, src=%s, turn=%d)",
                 self.session_id,
                 counts["entities"], counts["relations"], counts["facts"],
                 len(self.entities), len(self.relations), len(self.facts),
                 source_type, turn_id)
        return counts

    def _merge_evidence(self, extracted: dict, source_type: str,
                        source_ref: str, source_title: str, turn_id: int) -> dict:
        now = time.time()
        counts = {"entities": 0, "relations": 0, "facts": 0}

        for e_raw in extracted.get("entities", []):
            try:
                e = Entity(
                    id=e_raw["id"], type=e_raw.get("type", "other"),
                    name=e_raw.get("name", e_raw["id"]),
                    attributes=e_raw.get("attributes", {}),
                    source_type=source_type, source_ref=source_ref,
                    source_title=source_title, turn_id=turn_id,
                    last_updated=now,
                )
                if e.id in self.entities:
                    self.entities[e.id].merge(e)
                else:
                    self.entities[e.id] = e
                counts["entities"] += 1
            except (KeyError, TypeError) as exc:
                log.debug("[facts] Bad entity: %s", exc)

        for r_raw in extracted.get("relations", []):
            try:
                r = Relation(
                    subject=r_raw["subject"], predicate=r_raw["predicate"],
                    object=r_raw["object"], detail=r_raw.get("detail", ""),
                    source_type=source_type, source_ref=source_ref,
                    source_title=source_title, turn_id=turn_id,
                    last_updated=now,
                )
                existing = self.relations.get(r.key)
                if existing:
                    old_pri = SOURCE_PRIORITY.get(existing.source_type, 0)
                    new_pri = SOURCE_PRIORITY.get(r.source_type, 0)
                    if new_pri > old_pri or (new_pri == old_pri and r.turn_id >= existing.turn_id):
                        self.relations[r.key] = r
                else:
                    self.relations[r.key] = r
                counts["relations"] += 1
            except (KeyError, TypeError) as exc:
                log.debug("[facts] Bad relation: %s", exc)

        for f_raw in extracted.get("facts", []):
            try:
                f = Fact(
                    key=f_raw["key"], value=f_raw["value"],
                    source_type=source_type, source_ref=source_ref,
                    source_title=source_title, turn_id=turn_id,
                    last_updated=now,
                )
                existing = self.facts.get(f.key)
                if existing:
                    old_pri = SOURCE_PRIORITY.get(existing.source_type, 0)
                    new_pri = SOURCE_PRIORITY.get(f.source_type, 0)
                    if new_pri > old_pri or (new_pri == old_pri and f.turn_id >= existing.turn_id):
                        log.info("[facts:%s] REPLACE '%s': '%s'(%s,t%d) → '%s'(%s,t%d)",
                                 self.session_id, f.key,
                                 existing.value[:50], existing.source_type, existing.turn_id,
                                 f.value[:50], f.source_type, f.turn_id)
                        self.facts[f.key] = f
                else:
                    self.facts[f.key] = f
                counts["facts"] += 1
            except (KeyError, TypeError) as exc:
                log.debug("[facts] Bad fact: %s", exc)

        return counts

    # ── Context Building ──

    def to_context_string(self) -> str:
        if not self.entities and not self.relations and not self.facts:
            return ""

        parts = []

        if self.entities:
            lines = []
            for e in sorted(self.entities.values(), key=lambda x: x.last_updated, reverse=True):
                attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
                attr_str = f" ({attrs})" if attrs else ""
                lines.append(f"- {e.name} [{e.type}]{attr_str} [{e.source_type}]")
            parts.append("[Verified entities]\n" + "\n".join(lines))

        if self.relations:
            lines = []
            for r in sorted(self.relations.values(), key=lambda x: x.last_updated, reverse=True):
                detail = f" — {r.detail}" if r.detail else ""
                lines.append(f"- {r.subject} → {r.predicate} → {r.object}{detail} [{r.source_type}]")
            parts.append("[Verified relations]\n" + "\n".join(lines))

        if self.facts:
            lines = []
            for f in sorted(self.facts.values(), key=lambda x: x.last_updated, reverse=True):
                lines.append(f"- {f.key}: {f.value} [{f.source_type}]")
            parts.append("[Verified facts]\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "facts": len(self.facts),
            "extractions": self._extraction_count,
        }

    def clear(self) -> None:
        self.entities.clear()
        self.relations.clear()
        self.facts.clear()
        self._extraction_count = 0
        self.save()

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def find_entities(self, type_filter: str = "") -> list[Entity]:
        if not type_filter:
            return list(self.entities.values())
        return [e for e in self.entities.values() if e.type == type_filter]

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        return [
            r for r in self.relations.values()
            if r.subject == entity_id or r.object == entity_id
        ]
