"""LLM judge: scores a story against the rubric and returns actionable revisions."""

from __future__ import annotations

from typing import Any, Dict

from . import prompts, readability
from .llm import LLM, call_json
from .models import RUBRIC, StoryRequest, Verdict


def _validate(data: Dict[str, Any]):
    scores = data.get("scores")
    if not isinstance(scores, dict):
        return "scores must be an object"
    missing = [k for k in RUBRIC if k not in scores]
    if missing:
        return f"scores is missing: {', '.join(missing)}"
    return None


def _to_score(value: Any) -> int:
    try:
        return min(max(int(round(float(value))), 1), 10)
    except (TypeError, ValueError):
        return 1


def _as_note(item: Any) -> str:
    """gpt-3.5 sometimes returns a revision as {"instruction": "example"}; flatten it."""
    if isinstance(item, dict):
        return "; ".join(f"{k} (for example: {v})" if v else str(k) for k, v in item.items())
    return str(item).strip()


def judge_story(llm: LLM, req: StoryRequest, story: str) -> Verdict:
    words = prompts.target_words(req.age, req.length)
    data = call_json(
        llm,
        prompts.judge_messages(req, story, readability.describe(story, req.age, words)),
        temperature=0.0,  # grading should be repeatable
        max_tokens=900,
        required_keys=("scores", "safety_ok", "revisions"),
        validate=_validate,
    )
    scores = {k: _to_score(data["scores"][k]) for k in RUBRIC}
    # Safety is decided twice: the model's flag and its own age score must agree it is fine.
    safety_ok = bool(data.get("safety_ok")) and scores["age_appropriateness"] > 3

    return Verdict(
        scores=scores,
        safety_ok=safety_ok,
        strengths=[str(s) for s in data.get("strengths") or []],
        revisions=[n for n in (_as_note(r) for r in data.get("revisions") or []) if n],
        summary=str(data.get("summary", "")),
        # Code-measured problems block a pass on their own, whatever the judge scored.
        measured_issues=readability.check(story, req.age, words),
        continuity_problems=[n for n in (_as_note(p) for p in data.get("continuity_problems") or []) if n],
    )
