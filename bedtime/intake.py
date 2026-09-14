"""Intake: turn free-text requests and feedback into safe, structured instructions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import prompts
from .llm import LLM, call_json
from .models import CATEGORIES, StoryRequest


class NotAStoryRequest(ValueError):
    pass


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _clamp_age(value: Any, default: int = 7) -> int:
    try:
        return min(max(int(value), 5), 10)
    except (TypeError, ValueError):
        return default


def parse_request(llm: LLM, raw: str, age_override: Optional[int] = None) -> StoryRequest:
    data: Dict[str, Any] = call_json(
        llm,
        prompts.intake_messages(raw),
        temperature=0.0,
        max_tokens=600,
        required_keys=("kind", "appropriate", "brief", "category"),
    )
    if data.get("kind") == "not_a_story":
        raise NotAStoryRequest(raw)

    category = data.get("category")
    characters = [
        {"name": str(c.get("name", "")), "description": str(c.get("description", ""))}
        for c in _as_list(data.get("characters"))
        if isinstance(c, dict)
    ]
    return StoryRequest(
        raw=raw,
        brief=str(data.get("brief") or raw),
        category=category if category in CATEGORIES else "calming",
        age=_clamp_age(age_override if age_override is not None else data.get("age")),
        length=data.get("length") if data.get("length") in ("short", "medium", "long") else "medium",
        characters=characters,
        setting=str(data.get("setting") or ""),
        must_include=[str(x) for x in _as_list(data.get("must_include"))],
        adjustment_note="" if data.get("appropriate") else str(data.get("adjustment_note") or ""),
    )


def screen_feedback(llm: LLM, req: StoryRequest, feedback: str) -> Tuple[str, str]:
    """Return (change instruction for the writer, note for the parent or "")."""
    data = call_json(
        llm,
        prompts.feedback_screen_messages(req.brief, feedback),
        temperature=0.0,
        max_tokens=300,
        required_keys=("appropriate", "change"),
    )
    note = "" if data.get("appropriate") else str(data.get("note") or "")
    return str(data.get("change") or feedback), note
