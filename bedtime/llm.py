"""Thin wrapper around the OpenAI chat API.

Every model call in the project goes through an object with a `chat` method, so
the model name lives in exactly one place and tests can swap in a fake.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Protocol

MODEL = "gpt-3.5-turbo"  # fixed by the assignment: do not change

Message = Dict[str, str]


class LLM(Protocol):
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str: ...


class OpenAIChat:
    """Production client. Retries on transient errors are handled by the SDK."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        from openai import OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or copy .env.example to .env."
            )
        self._client = OpenAI(api_key=key, max_retries=max_retries)

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        kwargs: Dict[str, Any] = dict(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class LLMFormatError(RuntimeError):
    pass


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Models sometimes wrap JSON in prose or code fences; grab the outer object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level value is not an object", text, 0)
    return data


def call_json(
    llm: LLM,
    messages: List[Message],
    *,
    temperature: float,
    max_tokens: int,
    required_keys: tuple = (),
    validate: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    attempts: int = 2,
) -> Dict[str, Any]:
    """Call the model in JSON mode and return a dict, repairing bad output once.

    On a parse or schema failure the model is shown its own reply and the exact
    problem, which fixes gpt-3.5's occasional slips far more reliably than a
    blind retry.
    """
    convo = list(messages)
    problem = ""
    for _ in range(attempts):
        raw = llm.chat(convo, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        try:
            data = _parse_json_object(raw)
            missing = [k for k in required_keys if k not in data]
            problem = f"missing keys: {', '.join(missing)}" if missing else ""
            if not problem and validate:
                problem = validate(data) or ""
            if not problem:
                return data
        except json.JSONDecodeError as exc:
            problem = f"invalid JSON ({exc.msg})"
        convo = convo + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": f"That reply was not usable: {problem}. "
                "Reply again with only the corrected JSON object.",
            },
        ]
    raise LLMFormatError(f"Model did not return usable JSON: {problem}")
