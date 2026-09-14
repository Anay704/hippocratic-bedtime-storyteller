"""Bedtime storyteller for ages 5-10, with an LLM editor in the loop.

Usage:
    python main.py                                  # interactive
    python main.py -r "A story about a brave snail" --age 6
    python main.py --verbose --save stories/        # show the editor's scorecards, save a transcript

Before submitting the assignment, describe here in a few sentences what you would have built next if
you spent 2 more hours on this project:

1. An offline eval harness: a fixed set of ~30 requests (including tricky ones like "a scary zombie
   story" or prompt-injection attempts), run the pipeline on each, and track judge scores, pass rate,
   number of revisions, and safety adjustments per commit, so prompt changes are measured, not guessed.
2. Calibrate the judge against people: have a few parents and teachers rank pairs of stories, then
   tune the rubric anchors until the judge's pairwise preferences agree with theirs. Also check for
   self-preference by swapping in a different judge prompt and comparing verdicts.
3. Pairwise judging for revisions: instead of scoring each draft alone, ask "is B better than A on
   these notes?" which is more reliable than absolute scores and stops revisions that drift.
4. A read-aloud mode using text-to-speech with slower pacing in the wind-down, and a "series" memory
   so a child's recurring characters come back night after night.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import openai

from bedtime.intake import NotAStoryRequest
from bedtime.llm import LLMFormatError, OpenAIChat
from bedtime.models import RUBRIC, StoryResult
from bedtime.pipeline import StoryPipeline

EXAMPLE_REQUEST = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."
WIDTH = 78


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so the project needs no extra dependency."""
    env_file = Path(__file__).resolve().parent / path
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_story(text: str) -> str:
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if block.startswith("# "):
            title = block[2:].strip()
            out.append(f"{title}\n{'~' * len(title)}")
        else:
            out.append(textwrap.fill(" ".join(block.split()), WIDTH))
    return "\n\n".join(out)


def render_scorecard(draft) -> str:
    v = draft.verdict
    bars = "\n".join(
        f"    {k.replace('_', ' '):<21} {'#' * v.scores[k]:<10} {v.scores[k]}" for k in RUBRIC
    )
    status = "PASS" if v.passed else ("SAFETY FAIL" if not v.safety_ok else "needs revision")
    lines = [f"  [{draft.label}] overall {v.overall}/10, {status}", bars]
    if v.summary:
        lines.append(textwrap.fill(v.summary, WIDTH, initial_indent="    ", subsequent_indent="    "))
    for note in v.notes:
        lines.append(textwrap.fill(note, WIDTH, initial_indent="    - ", subsequent_indent="      "))
    return "\n".join(lines)


class ConsoleReporter:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, event: str, payload: dict) -> None:
        if event == "stage":
            print(f"  ... {payload['name']}")
        elif event == "notice":
            print(f"\n  Note: {payload['message']}\n")
        elif event == "request":
            req = payload["request"]
            if req.adjustment_note:
                print(f"\n  Note: {req.adjustment_note}\n")
            if self.verbose:
                print(f"      category={req.category} age={req.age} length={req.length}")
        elif event == "outline" and self.verbose:
            outline = payload["outline"]
            print(f"      plan: {outline.logline}")
        elif event == "verdict" and self.verbose:
            print(render_scorecard(payload["draft"]))
        elif event == "final":
            d = payload["draft"]
            print(
                f"  ... Editor's pick: {d.label} "
                f"({d.verdict.overall}/10 after {payload['rounds']} review round(s))"
            )


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #


def save_transcript(result: StoryResult, directory: str) -> Path:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result.outline.title.lower()).strip("-")[:40] or "story"
    path = folder / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"

    req, outline = result.request, result.outline
    parts = [
        result.final.text,
        f"\n*Something to wonder about:* {outline.reflection_question}",
        "\n---\n## How this story was made",
        f"**Request:** {req.raw}  \n**Brief:** {req.brief}  \n"
        f"**Category:** {req.category} | **Age:** {req.age} | **Length:** {req.length}",
    ]
    if req.adjustment_note:
        parts.append(f"**Safety adjustment:** {req.adjustment_note}")
    if req.feedback:
        parts.append("**Requested changes:** " + "; ".join(req.feedback))
    parts.append("\n### Plan\n" + "\n".join(f"{i}. **{b['beat']}**: {b['summary']}" for i, b in enumerate(outline.beats, 1)))
    parts.append("\n### Editor rounds")
    for d in result.drafts:
        mark = " (final)" if d is result.final else ""
        parts.append(f"\n```\n{render_scorecard(d)}\n```{mark}")
    path.write_text("\n".join(parts) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bedtime stories for ages 5-10, reviewed by an LLM editor.")
    p.add_argument("-r", "--request", help="story request (otherwise you will be asked)")
    p.add_argument("--age", type=int, choices=range(5, 11), metavar="{5..10}", help="listener's age")
    p.add_argument("--rounds", type=int, default=2, help="max editor revision rounds (default 2)")
    p.add_argument("-v", "--verbose", action="store_true", help="show the plan and editor scorecards")
    p.add_argument("--save", metavar="DIR", help="save a markdown transcript to DIR")
    p.add_argument("--no-feedback", action="store_true", help="skip the change-request loop")
    return p.parse_args(argv)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def run_session(pipeline: StoryPipeline, args) -> Optional[StoryResult]:
    request = args.request
    while True:
        if not request:
            request = ask(
                f"What kind of story do you want to hear?\n(press Enter for: {EXAMPLE_REQUEST})\n> "
            ) or EXAMPLE_REQUEST
        try:
            req = pipeline.understand(request, age_override=args.age)
            break
        except NotAStoryRequest:
            print("\nThat doesn't sound like a story request. Try something like: a dragon who is afraid of heights.\n")
            if args.request:
                return None
            request = None

    result = pipeline.create(req)
    while True:
        print("\n" + "=" * WIDTH + "\n")
        print(render_story(result.final.text))
        if result.outline.reflection_question:
            print(f"\nSomething to wonder about: {result.outline.reflection_question}")
        print("\n" + "=" * WIDTH)

        if args.no_feedback or not sys.stdin.isatty():
            return result
        feedback = ask("\nWant to change anything? (e.g. 'make it funnier', 'add a dragon'; Enter to finish)\n> ")
        if not feedback:
            return result
        pipeline.apply_feedback(result, feedback)


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    try:
        llm = OpenAIChat()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    pipeline = StoryPipeline(llm, max_revisions=max(args.rounds, 0), on_event=ConsoleReporter(args.verbose))

    try:
        result = run_session(pipeline, args)
    except LLMFormatError as exc:
        print(f"\nSorry, the storyteller got tangled up: {exc}", file=sys.stderr)
        return 1
    except openai.APIError as exc:
        print(f"\nOpenAI API error: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nGoodnight!")
        return 130
    if result is None:
        return 1

    if args.save:
        print(f"\nSaved transcript to {save_transcript(result, args.save)}")
    print("\nSweet dreams!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
