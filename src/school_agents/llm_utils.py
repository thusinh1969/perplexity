"""
llm_utils.py — Shared LLM output post-processing.

Central place for cleaning raw LLM output. Used by every module that
consumes LLM responses (crew_runner, compressor, fact_store, query_expander).
"""
from __future__ import annotations

import re

# Pre-compiled patterns — avoid re.compile on every call
_RE_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL)

# System message to disable thinking in Qwen3-style models for structured output
# Works with Qwen3/Qwen3.5 via LMStudio/vLLM — tells model to skip reasoning
NO_THINK_SYSTEM = (
    "You are a precise JSON extractor. "
    "Output ONLY valid JSON. No thinking, no reasoning, no explanation, no markdown fences. "
    "Start your response with { or [."
)


def no_think_extra_body(extra_body: dict | None = None) -> dict:
    """Merge extra_body with Qwen3 thinking-disable flags.

    Qwen3/Qwen3.5 models support disabling thinking via:
      - chat_template_kwargs.enable_thinking = false (vLLM/LMStudio)
      - Also reduces wasted tokens on internal reasoning for structured output tasks.

    Args:
        extra_body: Existing extra_body dict (top_k, repetition_penalty, etc.)

    Returns:
        New dict with thinking disabled.
    """
    merged = dict(extra_body or {})
    merged["chat_template_kwargs"] = {"enable_thinking": False}
    return merged


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output.

    Handles both:
      - Properly closed: <think>reasoning here</think>actual output
      - Unclosed (max_tokens cutoff): <think>reasoning here...

    Safe to call on text that has no think tags — returns unchanged.

    Args:
        text: Raw LLM output string.

    Returns:
        Cleaned string with all think blocks removed.
    """
    if "<think>" not in text:
        return text
    text = _RE_THINK_CLOSED.sub("", text)
    text = _RE_THINK_UNCLOSED.sub("", text)
    return text.strip()


def extract_json(text: str) -> str:
    """Extract JSON from messy LLM output that may contain thinking text.

    Handles:
      - <think>...</think> tags (Qwen3 thinking format)
      - "Thinking Process: ..." plain text (Qwen3.5 style)
      - Markdown code fences (```json ... ```)
      - Mixed text + JSON

    Tries in order:
      1. Strip <think> tags, return if valid-looking JSON remains
      2. Find last JSON object {...} in text
      3. Find last JSON array [...] in text
      4. Return empty string if nothing found

    Args:
        text: Raw LLM output string.

    Returns:
        Cleaned string containing just the JSON, or "" if not found.
    """
    if not text or not text.strip():
        return ""

    # 1. Try stripping think tags first
    cleaned = strip_think_tags(text)
    if cleaned.strip():
        # Remove markdown fences
        if cleaned.strip().startswith("```"):
            inner = cleaned.strip().split("```")
            if len(inner) >= 2:
                cleaned = inner[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        # Check if it looks like JSON
        s = cleaned.strip()
        if s.startswith("{") or s.startswith("["):
            return s

    # 2. Search for JSON object in raw text (last occurrence — most likely the actual output)
    # Use greedy match from last { to find the outermost object
    objects = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL))
    if objects:
        # Return the largest match (likely the full JSON)
        best = max(objects, key=lambda m: len(m.group()))
        return best.group()

    # 3. Search for JSON array
    arrays = list(re.finditer(r'\[.*?\]', text, re.DOTALL))
    if arrays:
        best = max(arrays, key=lambda m: len(m.group()))
        return best.group()

    return ""
