from __future__ import annotations
import json
import httpx
from typing import Any, Dict, Optional

from crewai.tools import tool
from ..tool_context import get_tool_config

def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

def _format_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")

@tool("student_get_profile")
def student_get_profile(student_id: str) -> str:
    """Fetch a student's profile by ID. Returns JSON string."""
    cfg = get_tool_config()["student_apis"]
    base = cfg["base_url"]
    api_key = cfg["api_key"]
    ep = cfg["endpoints"]["profile"].format(student_id=student_id)
    url = _format_url(base, ep)
    with httpx.Client(timeout=20.0) as client:
        r = client.get(url, headers=_auth_headers(api_key))
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)

@tool("student_get_grades")
def student_get_grades(student_id: str, from_date: str, to_date: str) -> str:
    """Fetch student grades for a date range. Returns JSON string."""
    cfg = get_tool_config()["student_apis"]
    base = cfg["base_url"]
    api_key = cfg["api_key"]
    ep = cfg["endpoints"]["grades"].format(student_id=student_id)
    url = _format_url(base, ep)
    params = {"from": from_date, "to": to_date}
    with httpx.Client(timeout=20.0) as client:
        r = client.get(url, headers=_auth_headers(api_key), params=params)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)

@tool("student_get_attendance")
def student_get_attendance(student_id: str, from_date: str, to_date: str) -> str:
    """Fetch student attendance records for a date range. Returns JSON string."""
    cfg = get_tool_config()["student_apis"]
    base = cfg["base_url"]
    api_key = cfg["api_key"]
    ep = cfg["endpoints"]["attendance"].format(student_id=student_id)
    url = _format_url(base, ep)
    params = {"from": from_date, "to": to_date}
    with httpx.Client(timeout=20.0) as client:
        r = client.get(url, headers=_auth_headers(api_key), params=params)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)
