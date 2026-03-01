from __future__ import annotations
from typing import Any, Dict

_TOOL_CFG: Dict[str, Any] | None = None

def set_tool_config(cfg: Dict[str, Any]) -> None:
    global _TOOL_CFG
    _TOOL_CFG = cfg

def get_tool_config() -> Dict[str, Any]:
    if _TOOL_CFG is None:
        raise RuntimeError("Tool config not set. Call set_tool_config(cfg) at startup.")
    return _TOOL_CFG
