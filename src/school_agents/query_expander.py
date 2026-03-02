"""
query_expander.py — Generate smarter search queries from user input.

Techniques:
  1. Query Rewrite:   Fix typos, add specificity, translate intent
  2. Query Expansion: Generate 2-3 related queries for broader coverage
  3. Result Fusion:   Merge + dedup results from multiple queries via reciprocal rank

Used in two modes:
  - Auto:        Expand silently, merge results (API / non-interactive)
  - Interactive:  Show expanded queries, ask user to confirm before searching

Usage:
    expander = QueryExpander(openai_client, model="qwen/qwen3-coder-next")

    # Generate expanded queries
    queries = expander.expand("python mới có gì hay")
    # → ["Python 3.13 new features official",
    #    "Python latest release improvements 2024",
    #    "Python 3.13 vs 3.12 comparison changes"]

    # Search all queries and merge results
    merged = expander.search_expanded(queries, search_fn=tavily_search)
    # → deduplicated, ranked results from all queries
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import json_repair

from .llm_utils import strip_think_tags, extract_json, NO_THINK_SYSTEM, no_think_extra_body

log = logging.getLogger("school_agents.query_expander")


EXPAND_PROMPT = """\
{today}

You are a search query optimizer. Given a user's question, generate exactly 3 search queries \
that together will find the most comprehensive and accurate answer from the web.

STRICT RULES:
1. Query 1: An ENGLISH search query — precise, well-formed, using standard search terms. \
   MANDATORY English regardless of user's language (English sources are more comprehensive).
2. Query 2: A query in the user's original language (Vietnamese if they wrote Vietnamese). \
   This captures local-language sources and perspectives.
3. Query 3: An ENGLISH query from a different angle — analytical, comparative, or predictive. \
   Example: if user asks about a war, this could be "expert analysis [topic] forecast".

ENTITY DISAMBIGUATION (CRITICAL):
- When the query mentions a specific entity (company, person, stock ticker, product, place), \
  ALWAYS add disambiguating context to EVERY query to avoid near-name confusion.
- Infer context from: user's language, conversation topic, domain, country.
  Examples of disambiguation:
  - Stock ticker in Vietnamese context → add "Vietnam" or "HOSE" or "HNX"
  - Company name that exists in multiple countries → add country
  - Person name → add role/field/context
  - Medical term → add specialty context
  - Ambiguous abbreviation → spell out + add context
- NEVER generate a bare ticker/name that could match a different entity in another market/country/domain.

FORMAT RULES:
- Each query: 5-12 words. Not too short (vague), not too long (over-specific).
- Include time context when relevant: add year/month, "latest", "today" for current events.
- Return ONLY a valid JSON array of exactly 3 strings.
- Do NOT include any explanation, numbering, or markdown. Just the JSON array.
- Start your response with [ and end with ]

EXAMPLES:
User: "Tình hình chiến tranh Mỹ Iran thế nào?"
→ ["US Iran war latest military updates March 2026", "tình hình chiến tranh Mỹ Iran mới nhất", "US Iran conflict expert analysis escalation forecast"]

User: "Bitcoin giá bao nhiêu?"
→ ["Bitcoin price today March 2026 USD", "giá Bitcoin hôm nay", "Bitcoin price prediction short term analysis"]

User question: {user_query}

JSON array:"""


class QueryExpander:
    """
    LLM-powered query expansion for better search coverage.

    Generates multiple search queries from a single user question,
    then merges results using reciprocal rank fusion.
    """

    def __init__(
        self,
        openai_client: Any,
        model: str,
        max_queries: int = 3,
        temperature: float = 0.3,
        extra_body: dict | None = None,
        max_tokens: int = 16384,
    ):
        self._oai = openai_client
        self._model = model
        self.max_queries = max_queries
        self._temperature = temperature
        self._extra_body = extra_body or {}
        self._max_tokens = max_tokens

    def expand(self, user_query: str, progress_callback=None) -> list[str]:
        """
        Generate expanded search queries from user's question.

        Args:
            user_query: Original user question
            progress_callback: Optional callable(token_count: int) called during streaming
                               to show progress. If None, no progress shown.

        Returns:
            List of 3 search queries (original rewrite + 2 expansions)
        """
        # Inject current date so LLM generates time-aware queries
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        _now = _dt.now(_ZI("Asia/Ho_Chi_Minh"))
        date_context = f"Today is {_now:%A, %B %d, %Y}."

        prompt = EXPAND_PROMPT.format(user_query=user_query, today=date_context)

        log.info("[expander] Expanding query: %r", user_query)
        t0 = time.perf_counter()

        try:
            # Stream the LLM call so user sees progress during thinking
            stream = self._oai.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": NO_THINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=0.95,
                extra_body=no_think_extra_body(self._extra_body),
                stream=True,
            )

            # Collect all tokens, report progress periodically
            chunks = []
            token_count = 0
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
                    token_count += 1
                    if progress_callback and token_count % 20 == 0:
                        progress_callback(token_count)

            raw = "".join(chunks).strip()
            log.debug("[expander] Raw LLM output (%d tokens, %d chars): %s",
                      token_count, len(raw), raw[:300])

            # Extract JSON from potentially messy output
            cleaned = extract_json(raw)
            if not cleaned:
                log.warning("[expander] No JSON found in LLM output")
                return [user_query]

            queries = json_repair.loads(cleaned)

            # Validate
            if not isinstance(queries, list):
                raise ValueError(f"Expected list, got {type(queries)}")
            queries = [str(q).strip() for q in queries if str(q).strip()]

            # Filter out garbage queries (thinking text leaked into results)
            valid = []
            for q in queries:
                # Skip if too long (>80 chars), too short (<3 chars), or contains markdown
                if len(q) < 3 or len(q) > 80:
                    log.debug("[expander] Skipping bad query (len=%d): %s", len(q), q[:50])
                    continue
                if any(marker in q.lower() for marker in ["**", "analyze", "thinking", "step ", "draft", "query 1", "query 2"]):
                    log.debug("[expander] Skipping thinking-leak query: %s", q[:50])
                    continue
                valid.append(q)

            queries = valid[:self.max_queries]
            if not queries:
                log.warning("[expander] All queries filtered as invalid — falling back")
                return [user_query]

            elapsed = time.perf_counter() - t0
            log.info("[expander] Generated %d queries in %.2fs: %s",
                     len(queries), elapsed, queries)
            return queries

        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("[expander] Parse failed: %s — falling back to original query", exc)
            return [user_query]
        except Exception as exc:
            log.error("[expander] LLM call failed: %s — falling back to original query", exc)
            return [user_query]

    def search_expanded(
        self,
        queries: list[str],
        search_fn: Callable[..., dict],
        max_results_per_query: int = 5,
        max_merged_results: int | None = None,
        **search_kwargs,
    ) -> dict:
        """
        Search all expanded queries and merge results via reciprocal rank fusion.

        Args:
            queries: List of search queries (from expand())
            search_fn: Function that takes (query, max_results, **kwargs) → dict with "results" key
            max_results_per_query: Results per individual query
            max_merged_results: Max results after fusion (default: max_results_per_query * len(queries), capped at 20)
            **search_kwargs: Extra kwargs passed to search_fn

        Returns:
            Merged dict with:
              - "queries": list of queries searched
              - "results": deduplicated, re-ranked results
              - "total_raw": total results before dedup
              - "fusion_method": "reciprocal_rank"
        """
        if max_merged_results is None:
            # Default: keep most results but cap at reasonable limit
            max_merged_results = min(max_results_per_query * len(queries), 20)
        all_results: list[dict] = []  # (result_dict, query_index, rank)
        query_results_map: list[list[dict]] = []

        log.info("[expander] Searching %d queries, %d results each...",
                 len(queries), max_results_per_query)
        t0 = time.perf_counter()

        for i, q in enumerate(queries):
            try:
                data = search_fn(
                    query=q,
                    max_results=max_results_per_query,
                    **search_kwargs,
                )
                results = data.get("results", [])
                query_results_map.append(results)
                log.info("[expander] Q%d %r → %d results", i + 1, q, len(results))
            except Exception as exc:
                log.warning("[expander] Q%d %r failed: %s", i + 1, q, exc)
                query_results_map.append([])

        # Reciprocal rank fusion
        merged = self._reciprocal_rank_fusion(query_results_map, max_results=max_merged_results)

        elapsed = time.perf_counter() - t0
        total_raw = sum(len(qr) for qr in query_results_map)
        log.info("[expander] Fusion done in %.2fs: %d raw → %d merged results",
                 elapsed, total_raw, len(merged))

        return {
            "queries": queries,
            "results": merged,
            "total_raw": total_raw,
            "fusion_method": "reciprocal_rank",
        }

    @staticmethod
    def _reciprocal_rank_fusion(
        query_results: list[list[dict]],
        k: int = 60,
        max_results: int = 15,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) to merge results from multiple queries.

        Score for each doc = sum over queries of: 1 / (k + rank)
        where k=60 (standard constant), rank is 1-indexed position.

        Deduplicates by URL.
        """
        # url → {score, best_result_dict}
        scored: dict[str, dict[str, Any]] = {}

        for query_idx, results in enumerate(query_results):
            for rank, result in enumerate(results, start=1):
                url = result.get("url", "")
                if not url:
                    # No URL — use title+content hash as key
                    url = f"_no_url_{hash(result.get('title', '') + result.get('content', ''))}"

                rrf_score = 1.0 / (k + rank)

                if url in scored:
                    scored[url]["score"] += rrf_score
                    # Keep the result with more content
                    existing_content = len(scored[url]["result"].get("content", ""))
                    new_content = len(result.get("content", ""))
                    if new_content > existing_content:
                        scored[url]["result"] = result
                else:
                    scored[url] = {"score": rrf_score, "result": result}

        # Sort by score descending, return top N
        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)

        merged = []
        for item in ranked[:max_results]:
            r = item["result"].copy()
            r["_rrf_score"] = round(item["score"], 6)
            merged.append(r)

        return merged

    def format_for_confirmation(self, original_query: str, expanded: list[str]) -> str:
        """
        Format expanded queries for interactive confirmation display.

        Returns a string like:
            Original: "python mới có gì hay"
            Expanded queries:
              1. Python 3.13 new features official
              2. Python latest release improvements 2024
              3. Python 3.13 vs 3.12 comparison changes
        """
        lines = [f'Original: "{original_query}"', "Expanded queries:"]
        for i, q in enumerate(expanded, 1):
            lines.append(f"  {i}. {q}")
        return "\n".join(lines)
