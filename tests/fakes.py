"""A scripted stand-in for the OpenAI client, routed by which role is speaking."""

from __future__ import annotations

import json
from typing import Callable, Dict, List

from bedtime import prompts
from bedtime.models import RUBRIC

GOOD_STORY = "# Rex and the Night Light\n\n" + " ".join(
    ["Rex the little dinosaur saw the moon glow soft and round."] * 45
)
SHORT_STORY = "# Rex\n\nRex was scared. Then he was not. The end."


def scores(value: int) -> Dict[str, int]:
    return {k: value for k in RUBRIC}


def intake_reply(**overrides) -> str:
    data = {
        "kind": "story",
        "appropriate": True,
        "brief": "A story about a dinosaur named Rex who is scared of the dark.",
        "adjustment_note": "",
        "category": "facing_fears",
        "age": 6,
        "length": "medium",
        "characters": [{"name": "Rex", "description": "a small dinosaur"}],
        "setting": "",
        "must_include": ["Rex"],
    }
    data.update(overrides)
    return json.dumps(data)


PLAN_REPLY = json.dumps(
    {
        "title": "Rex and the Night Light",
        "logline": "Rex wants to sleep but the dark feels too big.",
        "beats": [{"beat": f"beat {i}", "summary": "something happens"} for i in range(6)],
        "heart": "brave can be small",
        "refrain": "Stomp, stomp, glow",
        "reflection_question": "What helps you feel cozy at night?",
    }
)


def judge_reply(value: int, safety_ok: bool = True, revisions=None) -> str:
    return json.dumps(
        {
            "evidence": {"strongest_line": "a", "weakest_line": "b"},
            "scores": scores(value),
            "safety_ok": safety_ok,
            "strengths": ["warm"],
            "revisions": ["Add a sound word when Rex stomps."] if revisions is None else revisions,
            "summary": "ok",
        }
    )


class FakeLLM:
    """Each role pops its next scripted reply; every call is recorded for assertions."""

    def __init__(self, **scripts: List[str]):
        self.scripts = {role: list(replies) for role, replies in scripts.items()}
        self.calls: List[Dict] = []

    @staticmethod
    def role_of(messages) -> str:
        system = messages[0]["content"]
        table: Dict[str, Callable[[str], bool]] = {
            "intake": lambda s: s == prompts.INTAKE_SYSTEM,
            "feedback": lambda s: s == prompts.FEEDBACK_SCREEN_SYSTEM,
            "plan": lambda s: s == prompts.PLANNER_SYSTEM,
            "write": lambda s: s == prompts.STORYTELLER_SYSTEM,
            "judge": lambda s: s == prompts.judge_system(),
        }
        for role, match in table.items():
            if match(system):
                return role
        raise AssertionError("unknown system prompt")

    def chat(self, messages, *, temperature, max_tokens, json_mode=False) -> str:
        role = self.role_of(messages)
        self.calls.append({"role": role, "messages": messages, "temperature": temperature})
        replies = self.scripts.get(role)
        if not replies:
            raise AssertionError(f"no scripted reply left for {role}")
        return replies.pop(0)

    def count(self, role: str) -> int:
        return sum(1 for c in self.calls if c["role"] == role)
