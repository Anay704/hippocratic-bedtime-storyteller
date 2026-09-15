"""Orchestrates the agents: intake -> plan -> write -> (judge -> revise)* -> feedback."""

from __future__ import annotations

from typing import Callable, List, Optional

from . import intake, judge, storyteller
from .llm import LLM
from .models import RUBRIC, Draft, StoryRequest, StoryResult
from .prompts import RUBRIC_GUIDE

EventHandler = Callable[[str, dict], None]


def _silent(event: str, payload: dict) -> None:
    pass


class StoryPipeline:
    def __init__(self, llm: LLM, max_revisions: int = 2, on_event: Optional[EventHandler] = None):
        self.llm = llm
        self.max_revisions = max_revisions
        self.emit = on_event or _silent

    # -- stage 1 ----------------------------------------------------------- #
    def understand(self, raw: str, age_override: Optional[int] = None) -> StoryRequest:
        self.emit("stage", {"name": "Reading your request"})
        req = intake.parse_request(self.llm, raw, age_override)
        self.emit("request", {"request": req})
        return req

    # -- stages 2-4 -------------------------------------------------------- #
    def create(self, req: StoryRequest) -> StoryResult:
        self.emit("stage", {"name": "Planning the story arc"})
        outline = storyteller.plan_story(self.llm, req)
        self.emit("outline", {"outline": outline})

        self.emit("stage", {"name": "Writing the first draft"})
        story = storyteller.write_story(self.llm, req, outline)

        result = StoryResult(request=req, outline=outline)
        self._refine(result, story, first_label="draft 1")
        return result

    # -- stage 5 ----------------------------------------------------------- #
    def apply_feedback(self, result: StoryResult, feedback: str) -> StoryResult:
        req = result.request
        change, note = intake.screen_feedback(self.llm, req, feedback)
        if note:
            self.emit("notice", {"message": note})
        req.feedback.append(change)

        n = len(req.feedback)
        # Re-plan first so the change becomes part of the arc instead of a paragraph bolted onto the end.
        self.emit("stage", {"name": f"Updating the plan: {change}"})
        result.outline = storyteller.plan_story(self.llm, req, previous=result.outline, change=change)
        self.emit("outline", {"outline": result.outline})

        # Write fresh from the updated plan: revising the old text made gpt-3.5 bolt the change onto the end.
        self.emit("stage", {"name": "Retelling the story"})
        story = storyteller.write_story(self.llm, req, result.outline)
        # The judge now also checks the change was honored (requested_changes in its brief).
        self._refine(result, story, first_label=f"feedback {n}", parent_change=change)
        return result

    # -- judge/revise loop ------------------------------------------------- #
    def _refine(
        self, result: StoryResult, story: str, first_label: str, parent_change: str = ""
    ) -> None:
        req, outline = result.request, result.outline
        round_drafts: List[Draft] = []
        label = first_label

        for attempt in range(self.max_revisions + 1):
            self.emit("stage", {"name": f"Editor reviewing {label}"})
            verdict = judge.judge_story(self.llm, req, story)
            draft = Draft(text=story, verdict=verdict, label=label)
            round_drafts.append(draft)
            self.emit("verdict", {"draft": draft})

            if verdict.passed or attempt == self.max_revisions:
                break

            notes = verdict.notes or _fallback_notes(verdict.scores)
            label = f"{first_label} rev {attempt + 1}"
            self.emit("stage", {"name": f"Revising ({len(notes)} editor notes)"})
            story = storyteller.revise_story(
                self.llm, req, outline, story, notes, parent_change=parent_change
            )

        best = max(round_drafts, key=lambda d: d.verdict.rank_key())
        result.drafts.extend(round_drafts)
        result.final = best
        self.emit("final", {"draft": best, "rounds": len(round_drafts)})


def _fallback_notes(scores: dict) -> List[str]:
    """If the judge scored low but gave no revisions, target its weakest dimensions."""
    weakest = sorted(RUBRIC, key=lambda k: scores.get(k, 0))[:2]
    return [f"Strengthen {k.replace('_', ' ')}: {RUBRIC_GUIDE[k]}" for k in weakest]

