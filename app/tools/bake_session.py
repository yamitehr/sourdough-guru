"""Supabase session management using direct HTTP calls (httpx).

Uses the PostgREST API directly to avoid supabase-py key format issues.
"""

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.config import get_settings

logger = logging.getLogger("sourdough.supabase")


def _headers() -> dict:
    settings = get_settings()
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _rest_url(table: str) -> str:
    settings = get_settings()
    return f"{settings.SUPABASE_URL}/rest/v1/{table}"


# ---- Session listing ----

def list_all_sessions() -> list[dict]:
    """List all chat sessions, most recent first."""
    url = _rest_url("chat_sessions") + "?select=session_id,updated_at,messages&order=updated_at.desc&limit=50"
    resp = httpx.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    sessions = []
    for row in resp.json():
        msgs = row.get("messages", [])
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # Extract first user message as preview
        preview = ""
        for m in msgs:
            if m.get("role") == "user":
                preview = m["content"][:80]
                break
        sessions.append({
            "session_id": row["session_id"],
            "updated_at": row.get("updated_at", ""),
            "preview": preview,
            "message_count": len(msgs),
        })
    return sessions


def delete_session(session_id: str) -> None:
    """Delete a chat session and its associated bake session."""
    headers = _headers()
    # Delete chat session
    resp = httpx.delete(
        _rest_url("chat_sessions") + f"?session_id=eq.{session_id}",
        headers=headers, timeout=10,
    )
    resp.raise_for_status()
    # Also delete any associated bake session
    resp = httpx.delete(
        _rest_url("bake_sessions") + f"?session_id=eq.{session_id}",
        headers=headers, timeout=10,
    )
    resp.raise_for_status()
    logger.info(f"[Supabase] Deleted session {session_id}")


# ---- Chat session persistence ----

def save_session(session_id: str, messages: list[dict]) -> None:
    """Save or update conversation messages for a session."""
    payload = {
        "session_id": session_id,
        "messages": json.dumps(messages),
        "updated_at": datetime.now(ZoneInfo("Asia/Jerusalem")).replace(tzinfo=None).isoformat(),
    }
    headers = _headers()
    # Upsert: use Prefer: resolution=merge-duplicates
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    resp = httpx.post(_rest_url("chat_sessions"), json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    logger.info(f"[Supabase] Saved chat session {session_id}")


def load_session(session_id: str) -> list[dict]:
    """Load conversation messages for a session. Returns [] if not found."""
    url = _rest_url("chat_sessions") + f"?session_id=eq.{session_id}&select=messages&limit=1"
    resp = httpx.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data:
        raw = data[0]["messages"]
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    return []


# ---- Bake plan persistence ----

def save_bake_plan(session_id: str, plan_data: dict) -> str:
    """Save a bake plan (timeline + details) linked to a session.

    Deletes any existing plan for this session first to guarantee a clean replacement.
    """
    headers = _headers()
    # Remove old plan so the new one is always the single active record
    httpx.delete(
        _rest_url("bake_sessions") + f"?session_id=eq.{session_id}",
        headers=headers,
        timeout=10,
    )
    payload = {
        "session_id": session_id,
        "plan_data": json.dumps(plan_data),
        "created_at": datetime.now(ZoneInfo("Asia/Jerusalem")).replace(tzinfo=None).isoformat(),
        "active": True,
    }
    headers["Prefer"] = "return=representation"
    resp = httpx.post(_rest_url("bake_sessions"), json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    logger.info(f"[Supabase] Saved bake plan for session {session_id}")
    return session_id


def get_bake_status(session_id: str) -> dict:
    """Get current bake status for the polling endpoint."""
    url = _rest_url("bake_sessions") + f"?session_id=eq.{session_id}&select=plan_data,active&limit=1"
    resp = httpx.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return {"session_id": session_id, "active": False, "steps": []}

    row = data[0]
    plan_data = row["plan_data"]
    if isinstance(plan_data, str):
        plan_data = json.loads(plan_data)

    timeline = plan_data.get("timeline", [])
    now = datetime.now(ZoneInfo("Asia/Jerusalem")).replace(tzinfo=None)

    current_step = None
    next_step = None
    time_remaining = None

    for i, step in enumerate(timeline):
        start = datetime.fromisoformat(step["start_time"])
        end = datetime.fromisoformat(step["end_time"])
        if start <= now <= end:
            current_step = step["name"]
            time_remaining = (end - now).total_seconds() / 60
            if i + 1 < len(timeline):
                next_step = timeline[i + 1]["name"]
            break
        elif now < start:
            next_step = step["name"]
            time_remaining = (start - now).total_seconds() / 60
            break

    return {
        "session_id": session_id,
        "active": row.get("active", False),
        "current_step": current_step,
        "next_step": next_step,
        "time_remaining_minutes": round(time_remaining, 1) if time_remaining else None,
        "steps": timeline,
    }
