from __future__ import annotations

import time
from typing import Optional

from flask import Blueprint, jsonify, request, current_app

from .auth import token_required

system_prompt_bp = Blueprint("system_prompt", __name__)

# Keys used in Supabase table `system_prompts`
PROMPT_KEY_LIFELINES_RETRIEVER = "lifelines_retriever"
PROMPT_KEY_BLUELINES_RETRIEVER = "bluelines_retriever"
PROMPT_KEY_REPORT_GENERATION = "report_generation"

# Simple in-process cache to avoid hitting Supabase on every request
_PROMPT_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 60.0


def _cache_get(key: str) -> Optional[str]:
    entry = _PROMPT_CACHE.get(key)
    if not entry:
        return None
    value, expires_at = entry
    if time.time() >= expires_at:
        _PROMPT_CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: str) -> None:
    _PROMPT_CACHE[key] = (value, time.time() + _CACHE_TTL_SECONDS)


def _cache_invalidate(key: str) -> None:
    _PROMPT_CACHE.pop(key, None)


def get_system_prompt(key: str, fallback: str = None) -> str:
    """
    Returns prompt from Supabase `system_prompts` table.
    Raises ValueError if prompt is not found in Supabase.
    
    The fallback parameter is kept for backwards compatibility but is not used.
    All prompts must be stored in Supabase.
    """
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        rec = (
            current_app.supabase.table("system_prompts")
            .select("prompt")
            .eq("key", key)
            .limit(1)
            .execute()
        )
        if rec.data and rec.data[0].get("prompt"):
            value = rec.data[0]["prompt"]
            _cache_set(key, value)
            return value
        else:
            raise ValueError(f"System prompt '{key}' not found in Supabase. Please configure it in the database.")
    except Exception as e:
        if "not found in Supabase" in str(e):
            raise
        raise ValueError(f"Failed to fetch system prompt '{key}' from Supabase: {str(e)}")


def set_system_prompt(key: str, prompt: str, updated_by: Optional[str] = None) -> None:
    """
    Stores prompt into Supabase `system_prompts` (insert or update).
    """
    payload = {
        "key": key,
        "prompt": prompt,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if updated_by is not None:
        payload["updated_by"] = updated_by

    # Insert or update (manual upsert pattern)
    existing = (
        current_app.supabase.table("system_prompts")
        .select("key")
        .eq("key", key)
        .limit(1)
        .execute()
    )
    if existing.data:
        current_app.supabase.table("system_prompts").update(payload).eq("key", key).execute()
    else:
        current_app.supabase.table("system_prompts").insert(payload).execute()

    _cache_invalidate(key)


def _get_prompt_key_or_404(name: str) -> str:
    normalized = (name or "").strip().lower()
    if normalized in ("lifelines", "lifelines_retriever", "lifelines-retriever"):
        return PROMPT_KEY_LIFELINES_RETRIEVER
    if normalized in ("bluelines", "bluelines_retriever", "bluelines-retriever"):
        return PROMPT_KEY_BLUELINES_RETRIEVER
    if normalized in ("report", "report_generation", "report-generation", "reportgen"):
        return PROMPT_KEY_REPORT_GENERATION
    return ""


@system_prompt_bp.route("/<name>", methods=["GET"])
@token_required
def get_prompt(user, name: str):
    """
    Get the stored system prompt for a subsystem.
    URL param `name`:
      - lifelines
      - bluelines
      - report
    """
    key = _get_prompt_key_or_404(name)
    if not key:
        return jsonify({"error": "Unknown prompt name"}), 404

    # Fetch stored prompt from Supabase
    try:
        rec = (
            current_app.supabase.table("system_prompts")
            .select("key, prompt, updated_at, updated_by")
            .eq("key", key)
            .limit(1)
            .execute()
        )
        if rec.data:
            stored = rec.data[0].get("prompt")
            if stored:
                return jsonify({
                    "ok": True,
                    "key": key,
                    "prompt": stored,
                    "stored_prompt": stored,
                    "using_default": False,
                    "updated_at": rec.data[0].get("updated_at"),
                    "updated_by": rec.data[0].get("updated_by"),
                }), 200
    except Exception:
        pass

    # If no prompt found in Supabase, return error
    return jsonify({
        "ok": False,
        "key": key,
        "error": f"System prompt '{key}' not configured in Supabase. Please configure it first.",
        "stored_prompt": None,
        "using_default": False,
        "updated_at": None,
        "updated_by": None,
    }), 404


@system_prompt_bp.route("/<name>", methods=["PUT"])
@token_required
def update_prompt(user, name: str):
    """
    Update the stored system prompt for a subsystem.
    Body JSON:
      { "prompt": "..." }
    """
    key = _get_prompt_key_or_404(name)
    if not key:
        return jsonify({"error": "Unknown prompt name"}), 404

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Field 'prompt' is required"}), 400

    try:
        user_id = getattr(user, "id", None)
        set_system_prompt(key=key, prompt=prompt, updated_by=str(user_id) if user_id else None)
        return jsonify({"ok": True, "key": key}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update system prompt: {str(e)}"}), 500
