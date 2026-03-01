from __future__ import annotations
import base64
import json
import httpx
from typing import Dict

from crewai.tools import tool
from ..tool_context import get_tool_config

def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}

@tool("speech_stt")
def speech_stt(audio_b64: str, mime_type: str = "audio/wav") -> str:
    """Optional STT. Expects base64 audio, returns JSON {text: ...}."""
    cfg = get_tool_config().get("audio", {})
    if not cfg.get("enabled"):
        return json.dumps({"error": "audio disabled"}, ensure_ascii=False)

    stt = cfg["stt"]
    url = stt["base_url"].rstrip("/") + "/stt"
    api_key = stt["api_key"]

    audio_bytes = base64.b64decode(audio_b64)
    files = {"file": ("audio", audio_bytes, mime_type)}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=_auth_headers(api_key), files=files)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)

@tool("speech_tts")
def speech_tts(text: str, voice: str = "default") -> str:
    """Optional TTS. Returns JSON {audio_b64: ..., mime_type: ...} or {url: ...}."""
    cfg = get_tool_config().get("audio", {})
    if not cfg.get("enabled"):
        return json.dumps({"error": "audio disabled"}, ensure_ascii=False)

    tts = cfg["tts"]
    url = tts["base_url"].rstrip("/") + "/tts"
    api_key = tts["api_key"]

    payload = {"text": text, "voice": voice}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers={**_auth_headers(api_key), "Content-Type": "application/json"}, json=payload)
        r.raise_for_status()
        return json.dumps(r.json(), ensure_ascii=False)
