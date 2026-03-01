from __future__ import annotations
import json
import httpx
from typing import Any, Dict, Optional

from crewai.tools import tool
from ..tool_context import get_tool_config

def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

@tool("rag_query_policy")
def rag_query_policy(query: str, domain: str = "other") -> str:
    """Call your internal RAG service (hybrid dense+BM25 + rerank). Returns JSON string."""
    cfg = get_tool_config()["rag"]
    base = cfg["base_url"].rstrip("/")
    api_key = cfg["api_key"]
    defaults = cfg.get("defaults", {})
    payload: Dict[str, Any] = {
        "query": query,
        "domain": domain,
        "retrieval": {
            "use_dense": bool(defaults.get("use_dense", True)),
            "use_bm25": bool(defaults.get("use_bm25", True)),
            "alpha": float(defaults.get("alpha", 0.5)),
            "top_k": int(defaults.get("top_k", 20)),
        },
        "rerank": defaults.get("rerank", {"enabled": False}),
    }

    # Adjust to your actual RAG endpoint path
    url = f"{base}/query"
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, headers=_auth_headers(api_key), json=payload)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)
