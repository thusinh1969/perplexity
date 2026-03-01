"""
image_context.py — Thread-safe image storage for multimodal CrewAI pipeline.

Usage:
    from .image_context import set_images, get_images, clear_images

    set_images([{"b64": "...", "mime": "image/jpeg"}, ...])
    # ... CrewAI runs, litellm patch auto-injects images into LLM calls ...
    clear_images()
"""
from __future__ import annotations
import threading
import logging

log = logging.getLogger("school_agents.image_context")

_local = threading.local()


def set_images(images: list[dict]) -> None:
    """Store images for current request thread.

    Args:
        images: List of {"b64": base64_string, "mime": "image/jpeg|image/png|..."}
    """
    _local.images = list(images)
    log.info("[image_ctx] Set %d image(s)", len(images))


def get_images() -> list[dict]:
    """Get images for current request. Returns [] if none."""
    return getattr(_local, "images", [])


def clear_images() -> None:
    """Clear after request completes."""
    _local.images = []
