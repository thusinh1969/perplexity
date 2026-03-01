"""
context_compressor.py — Compress conversation history to fit context windows.

Two strategies (can be combined):
  1. LLM Summary:   Ask the LLM to summarize older turns (semantic, high quality)
  2. LLMLingua:      Token-level compression via Microsoft LLMLingua-2 (fast, no LLM call)
  3. Hybrid:         LLM summary → LLMLingua post-compression (maximum compression)

Usage:
    compressor = ContextCompressor(strategy="llm_summary")
    compressor = ContextCompressor(strategy="llmlingua")
    compressor = ContextCompressor(strategy="hybrid")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger("school_agents.compressor")

# ── LLMLingua availability ──
_LLMLINGUA_AVAILABLE = False
try:
    from llmlingua import PromptCompressor as _LinguaCompressor
    _LLMLINGUA_AVAILABLE = True
except ImportError:
    _LinguaCompressor = None


SUMMARY_PROMPT_TEMPLATE = """\
Summarize this conversation preserving ALL important details.

RULES:
- Keep: key topics, questions asked, answers given, decisions made, facts, names, numbers, URLs
- Keep: technical details, code snippets mentioned, configurations discussed
- Drop: greetings, filler words, excessive formatting, repeated information
- Write in the SAME LANGUAGE as the conversation (Vietnamese → Vietnamese, English → English)
- Scale your summary to the conversation length:
  * Short conversation (2-4 turns): 3-5 sentences
  * Medium conversation (5-10 turns): 1-2 paragraphs
  * Long conversation (10+ turns): 2-4 paragraphs organized by topic
- Use bullet points for distinct topics if the conversation covers multiple subjects
- Include specific numbers, dates, names — these are critical context for future turns
{existing_summary_block}
Conversation to summarize:
{history_text}

Summary:"""


class ContextCompressor:
    """
    Compress conversation history using configurable strategy.

    Args:
        strategy: "llm_summary" | "llmlingua" | "hybrid"
        openai_client: OpenAI-compatible client for LLM summary calls
        model: Model name for summary calls
        lingua_target_ratio: Token retention ratio for LLMLingua (0.0-1.0)
        lingua_device: "cpu" or "cuda"
        lingua_model: HuggingFace model for LLMLingua-2
    """

    def __init__(
        self,
        strategy: str = "llm_summary",
        openai_client: "OpenAI | None" = None,
        model: str = "",
        lingua_target_ratio: float = 0.4,
        lingua_device: str = "cpu",
        lingua_model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        extra_body: dict | None = None,
        max_tokens: int = 16384,
    ):
        if strategy not in ("llm_summary", "llmlingua", "hybrid"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self.strategy = strategy
        self._oai = openai_client
        self._model = model
        self._extra_body = extra_body or {}
        self._max_tokens = max_tokens
        self._lingua_ratio = lingua_target_ratio
        self._lingua_device = lingua_device
        self._lingua_model_name = lingua_model
        self._lingua: _LinguaCompressor | None = None  # lazy init

        # Validate dependencies
        if strategy in ("llmlingua", "hybrid") and not _LLMLINGUA_AVAILABLE:
            log.warning(
                "[compressor] LLMLingua not installed. "
                "Falling back to 'llm_summary'. Install: pip install llmlingua"
            )
            self.strategy = "llm_summary"

        if strategy in ("llm_summary", "hybrid") and openai_client is None:
            raise ValueError(
                f"Strategy '{strategy}' requires openai_client. "
                "Pass an OpenAI-compatible client."
            )

        log.info("[compressor] Initialized: strategy=%s", self.strategy)

    # ── Public API ──

    def compress(
        self,
        history_text: str,
        current_query: str = "",
        existing_summary: str = "",
        max_summary_tokens: int = 300,
    ) -> str:
        """
        Compress conversation history.

        Args:
            history_text: Formatted older turns to compress
            current_query: Current user question (helps LLMLingua preserve relevance)
            existing_summary: Previous summary to prepend/merge
            max_summary_tokens: Target max tokens for result

        Returns:
            Compressed summary string
        """
        if not history_text.strip():
            return existing_summary

        if self.strategy == "llm_summary":
            return self._compress_llm(history_text, existing_summary)

        elif self.strategy == "llmlingua":
            return self._compress_lingua(
                history_text, current_query, existing_summary
            )

        elif self.strategy == "hybrid":
            # Stage 1: LLM summary
            summary = self._compress_llm(history_text, existing_summary)
            # Stage 2: LLMLingua post-compression if still too long
            approx_tokens = len(summary) // 4
            if approx_tokens > max_summary_tokens:
                log.info(
                    "[compressor:hybrid] Summary ~%d tokens > %d limit, applying LLMLingua",
                    approx_tokens, max_summary_tokens,
                )
                summary = self._compress_lingua(summary, current_query, "")
            return summary

        return history_text  # fallback, should never reach

    # ── LLM Summary ──

    def _compress_llm(self, history_text: str, existing_summary: str) -> str:
        """Use LLM to produce semantic summary."""
        existing_block = ""
        if existing_summary:
            existing_block = f"\nPrevious summary to incorporate:\n{existing_summary}\n"

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            existing_summary_block=existing_block,
            history_text=history_text,
        )

        log.info(
            "[compressor:llm] Summarizing %d chars of history (existing summary: %d chars)",
            len(history_text), len(existing_summary),
        )

        try:
            from .llm_utils import strip_think_tags, no_think_extra_body
            resp = self._oai.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a thorough conversation summarizer. Capture all important details, decisions, and context. Output ONLY the summary text. No thinking tags, no reasoning, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=0.1,  # deterministic summary
                top_p=0.95,
                extra_body=no_think_extra_body(self._extra_body),
            )
            raw = resp.choices[0].message.content.strip()
            # Remove <think>...</think> blocks (Qwen3, etc.)
            summary = strip_think_tags(raw)
            # If think tags consumed everything, try to salvage text after </think>
            if not summary.strip() and "</think>" in raw:
                summary = raw.split("</think>")[-1].strip()
            if not summary.strip():
                log.warning("[compressor:llm] Summary empty after strip — using raw truncated")
                summary = raw[:500]  # better than nothing
            log.info("[compressor:llm] Summary: %d chars", len(summary))
            return summary
        except Exception as exc:
            log.error("[compressor:llm] Failed: %s — returning existing summary", exc)
            return existing_summary or history_text[:1000]

    # ── LLMLingua Token Compression ──

    def _get_lingua(self) -> _LinguaCompressor:
        """Lazy-init LLMLingua model."""
        if self._lingua is None:
            log.info(
                "[compressor:lingua] Loading model %s on %s (first call, may take a moment)...",
                self._lingua_model_name, self._lingua_device,
            )
            self._lingua = _LinguaCompressor(
                model_name=self._lingua_model_name,
                use_llmlingua2=True,
                device_map=self._lingua_device,
            )
            log.info("[compressor:lingua] Model loaded.")
        return self._lingua

    def _compress_lingua(
        self,
        text: str,
        current_query: str,
        existing_summary: str,
    ) -> str:
        """Token-level compression via LLMLingua-2."""
        lingua = self._get_lingua()

        full_context = text
        if existing_summary:
            full_context = f"Previous context: {existing_summary}\n\n{text}"

        log.info("[compressor:lingua] Compressing %d chars (ratio=%.2f)", len(full_context), self._lingua_ratio)

        try:
            result = lingua.compress_prompt(
                context=[full_context],
                question=current_query or "",
                rate=self._lingua_ratio,
                force_tokens=["\n", "?", ".", "!", ","],
                drop_consecutive=True,
                use_token_level_filter=True,
                token_budget_ratio=1.4,  # extra budget for non-English
            )
            compressed = result["compressed_prompt"]
            log.info(
                "[compressor:lingua] %d → %d chars (%.1fx compression)",
                len(full_context), len(compressed), len(full_context) / max(len(compressed), 1),
            )
            return compressed
        except Exception as exc:
            log.error("[compressor:lingua] Failed: %s — returning raw text", exc)
            return full_context
