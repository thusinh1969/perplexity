"""
image_context.py — Module-level image storage for multimodal CrewAI pipeline.

Uses a simple module-level list instead of threading.local() because
CrewAI may execute LLM calls in different threads than the caller.

Usage:
    from .image_context import set_images, get_images, clear_images

    set_images([{"b64": "...", "mime": "image/jpeg"}, ...])
    # ... CrewAI runs, litellm patch auto-injects images into LLM calls ...
    clear_images()
"""
from __future__ import annotations
import logging

log = logging.getLogger("school_agents.image_context")

# Module-level storage — visible from ANY thread in the process.
# Safe for CLI (single request). For server, set/clear wraps each request.
_current_images: list[dict] = []


def set_images(images: list[dict]) -> None:
    """Store images for current request.

    Args:
        images: List of {"b64": base64_string, "mime": "image/jpeg|image/png|..."}
    """
    global _current_images
    _current_images = list(images)
    log.info("[image_ctx] Set %d image(s)", len(images))


def get_images() -> list[dict]:
    """Get images for current request. Returns [] if none."""
    return _current_images


def clear_images() -> None:
    """Clear after request completes."""
    global _current_images
    _current_images = []
