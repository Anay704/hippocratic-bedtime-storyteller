"""Planner and storyteller: outline the arc, then write and revise prose."""

from __future__ import annotations

import re
from typing import List

from . import prompts
from .llm import LLM, call_json
from .models import Outline, StoryRequest

PLAN_TEMPERATURE = 0.7
WRITE_TEMPERATURE = 0.8  # creative, but low enough that gpt-3.5 still follows the plan
REVISE_TEMPERATURE = 0.6  # revisions should fix notes, not reinvent the story


def _validate_plan(data: dict):
    beats = data.get("beats")
    if not isinstance(beats, list) or len(beats) < 4:
        return "beats must be a list of 6 objects with 'beat' and 'summary'"
    return None


def plan_story(llm: LLM, req: StoryRequest) -> Outline:
    words = prompts.target_words(req.age, req.length)
    strategy = prompts.CATEGORY_STRATEGIES[req.category]
    data = call_json(
        llm,
        prompts.planner_messages(req, strategy, words),
        temperature=PLAN_TEMPERATURE,
        max_tokens=900,
        required_keys=("title", "logline", "beats"),
        validate=_validate_plan,
    )
    beats = [
        {"beat": str(b.get("beat", "")), "summary": str(b.get("summary", ""))}
        for b in data["beats"]
        if isinstance(b, dict)
    ]
    return Outline(
        title=str(data["title"]),
        logline=str(data["logline"]),
        beats=beats,
        heart=str(data.get("heart", "")),
        refrain=str(data.get("refrain", "")),
        reflection_question=str(data.get("reflection_question", "")),
    )


def _max_tokens(words: int) -> int:
    # ~1.35 tokens per English word, plus headroom so the ending is never truncated.
    return min(int(words * 1.35 * 1.6) + 200, 3000)


def clean_story(text: str, fallback_title: str) -> str:
    text = text.strip().strip("`").strip()
    # Drop the "## beat" markers the writer uses to pace each scene.
    text = re.sub(r"(?m)^#{2,}[^\n]*\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text.startswith("# "):
        text = f"# {fallback_title}\n\n{text}"
    # Keep the title on its own line even if the model runs straight into the story.
    title, _, body = text.partition("\n")
    return f"{title.strip()}\n\n{body.strip()}"


def write_story(llm: LLM, req: StoryRequest, outline: Outline) -> str:
    words = prompts.target_words(req.age, req.length)
    raw = llm.chat(
        prompts.write_messages(req, outline, words),
        temperature=WRITE_TEMPERATURE,
        max_tokens=_max_tokens(words),
    )
    return clean_story(raw, outline.title)


def revise_story(
    llm: LLM,
    req: StoryRequest,
    outline: Outline,
    story: str,
    notes: List[str],
    parent_change: str = "",
) -> str:
    words = prompts.target_words(req.age, req.length)
    raw = llm.chat(
        prompts.revise_messages(req, outline, words, story, notes, parent_change),
        temperature=REVISE_TEMPERATURE,
        max_tokens=_max_tokens(words),
    )
    return clean_story(raw, outline.title)
