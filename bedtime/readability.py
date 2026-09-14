"""Deterministic checks that ground the LLM judge.

LLMs are bad at counting words and estimating reading level, so code measures
those and hands the judge exact numbers. The checks also produce their own
revision notes, which means a too-long or too-complex story gets fixed even if
the judge overlooks it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?:\s|$)")

LESSON_TELLS = (
    "the moral of",
    "learned a valuable lesson",
    "learned an important lesson",
    "the lesson is",
    "and that's why you should",
)


def count_syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", word)))


@dataclass
class TextStats:
    words: int
    sentences: int
    avg_sentence_words: float
    fk_grade: float


def story_body(story: str) -> str:
    """Drop the markdown title line so it does not skew the counts."""
    lines = story.strip().splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def measure(story: str) -> TextStats:
    body = story_body(story)
    words = _WORD.findall(body)
    n_words = len(words)
    n_sentences = max(1, len(_SENTENCE_END.findall(body)))
    if n_words == 0:
        return TextStats(0, 0, 0.0, 0.0)
    syllables = sum(count_syllables(w) for w in words)
    avg = n_words / n_sentences
    grade = 0.39 * avg + 11.8 * (syllables / n_words) - 15.59
    return TextStats(n_words, n_sentences, round(avg, 1), round(max(grade, 0.0), 1))


def max_grade_for_age(age: int) -> float:
    # Stories are heard, not read, so listeners handle text well above their own
    # reading level: a 5-year-old ~ grade 3, a 10-year-old ~ grade 8. The syllable
    # heuristic also runs high on short dialogue-heavy text, so leave headroom.
    return float(age - 2)


def check(story: str, age: int, target_words: int) -> List[str]:
    """Return concrete revision notes for measurable problems (empty if none)."""
    stats = measure(story)
    notes: List[str] = []
    low, high = int(target_words * 0.7), int(target_words * 1.35)
    if stats.words < low:
        notes.append(
            f"The story is only {stats.words} words and {stats.sentences} sentences; it must be about {target_words} words, "
            f"roughly {round(target_words / max(stats.avg_sentence_words, 1))} sentences. Expand every beat into a full scene: in 'Trying', show each "
            "attempt with dialogue and what the characters see and hear; slow the 'Wind-down' with soft sensory details. "
            "Do not add new plot."
        )
    elif stats.words > high:
        notes.append(
            f"The story is {stats.words} words; trim it to about {target_words} by cutting repeated description."
        )
    limit = max_grade_for_age(age)
    if stats.fk_grade > limit + 1:
        notes.append(
            f"Reading level is grade {stats.fk_grade}, too hard for a {age}-year-old listener (target <= {limit:g}). "
            "Split long sentences and swap long words for short ones."
        )
    lowered = story.lower()
    for tell in LESSON_TELLS:
        if tell in lowered:
            notes.append(
                f"Remove the stated lesson ('{tell}...'); let the ending show it through what the characters do."
            )
            break
    return notes


def describe(story: str, age: int, target_words: int) -> str:
    s = measure(story)
    return (
        f"- word count: {s.words} (target about {target_words})\n"
        f"- average sentence length: {s.avg_sentence_words} words\n"
        f"- Flesch-Kincaid grade: {s.fk_grade} (comfortable max for a {age}-year-old listener: {max_grade_for_age(age):g})"
    )
