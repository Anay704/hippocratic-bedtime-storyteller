"""Plain data passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

CATEGORIES = (
    "adventure",
    "friendship",
    "animals",
    "fantasy",
    "funny",
    "facing_fears",
    "curiosity",
    "calming",
)

RUBRIC = (
    "age_appropriateness",
    "request_fidelity",
    "story_structure",
    "language_fit",
    "engagement",
    "character",
    "bedtime_ending",
)


@dataclass
class StoryRequest:
    raw: str
    brief: str  # the request restated, softened if it was not kid-safe
    category: str
    age: int
    length: str  # short | medium | long
    characters: List[Dict[str, str]] = field(default_factory=list)
    setting: str = ""
    must_include: List[str] = field(default_factory=list)
    adjustment_note: str = ""  # shown to the parent when the request was softened
    feedback: List[str] = field(default_factory=list)  # accumulated change requests


@dataclass
class Outline:
    title: str
    logline: str
    beats: List[Dict[str, str]]
    heart: str  # the gentle lesson, to be shown not told
    refrain: str  # repeated phrase or sound kids can join in on
    reflection_question: str


@dataclass
class Verdict:
    scores: Dict[str, int]
    safety_ok: bool
    strengths: List[str]
    revisions: List[str]
    summary: str
    measured_issues: List[str] = field(default_factory=list)  # from readability.check
    continuity_problems: List[str] = field(default_factory=list)  # from the judge's trace
    pass_score: float = 8.0
    min_dim_score: int = 7

    @property
    def overall(self) -> float:
        # Computed in code, never trusted from the model's arithmetic.
        return round(sum(self.scores.values()) / len(self.scores), 2) if self.scores else 0.0

    @property
    def passed(self) -> bool:
        return (
            self.safety_ok
            and not self.measured_issues
            and not self.continuity_problems
            and self.overall >= self.pass_score
            and min(self.scores.values(), default=0) >= self.min_dim_score
        )

    @property
    def notes(self) -> List[str]:
        """Everything the writer should fix next, measured problems first."""
        fixes = [f"Fix this continuity error: {p}" for p in self.continuity_problems]
        return self.measured_issues + fixes + self.revisions

    def rank_key(self):
        """Order drafts so a revision that got worse never replaces a better one."""
        return (
            self.safety_ok,
            self.passed,
            min(self.scores.values(), default=0),
            self.overall,
            -len(self.measured_issues) - len(self.continuity_problems),
        )


@dataclass
class Draft:
    text: str
    verdict: Verdict
    label: str  # e.g. "draft 1", "revision 2", "feedback 1"


@dataclass
class StoryResult:
    request: StoryRequest
    outline: Outline
    drafts: List[Draft] = field(default_factory=list)
    final: Optional[Draft] = None
