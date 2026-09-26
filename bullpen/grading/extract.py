"""Answer extraction from a model's response.

Each function returns a candidate answer (or ``None``); :mod:`bullpen.grading.graders`
decides correctness. :func:`think_first` runs an extractor on the text after
``</think>`` and falls back to the whole response.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from bullpen.grading.adapters import LETTERS
from bullpen.grading.legacy import extract_boxed, unwrap_latex_text

_THINK_END = re.compile(r"</think>", re.IGNORECASE)

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

#: Chat-template special tokens (``<|im_end|>``, ``<|eot_id|>``, ...).
_SPECIAL_TOKENS = re.compile(r"<\|[^|]+\|>")


def strip_special_tokens(text: str) -> str:
    """Remove chat-template special tokens from a raw decode."""
    return _SPECIAL_TOKENS.sub("", text).strip() if text else ""


def after_think(text: str) -> str:
    """The response after the last ``</think>``, or the whole text if nothing follows it."""
    parts = _THINK_END.split(text or "")
    tail = parts[-1] if len(parts) > 1 else ""
    return tail if tail.strip() else (text or "")


def think_first(fn: Callable[..., str | None], text: str, *args: object) -> str | None:
    """Run ``fn`` on the post-``</think>`` region, falling back to the full text."""
    tail = after_think(text)
    got = fn(tail, *args)
    if got is not None:
        return got
    return fn(text, *args) if tail != text else None


def extract_mcq(text: str, n_choices: int = len(LETTERS)) -> str | None:
    """Extract an option letter among the first ``n_choices`` letters.

    Preference order: ``\\boxed{X}``, "the answer is X", a trailing ``(X)``, then the
    last parenthesised letter anywhere.
    """
    text = unwrap_latex_text(text)
    n = max(1, min(n_choices, len(LETTERS)))
    rng = LETTERS[:n] + LETTERS[:n].lower()
    cls = f"[{rng}]"
    m = re.search(rf"\\boxed\{{\s*\(?\s*({cls})\s*\)?[^}}]*\}}", text)
    if m:
        return m.group(1).upper()
    for pat in (
        rf"(?:final\s+)?answer\s+is[:\s]*\**\(?\s*({cls})\s*\)?\b",
        rf"answer[:\s]*\**\(?\s*({cls})\s*\)?\b",
        rf"\(({cls})\)\s*$",
        rf"^\(?({cls})\)?[.\s]*$",
    ):
        m = re.search(pat, text.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    found = re.findall(rf"\(({cls})\)", text)
    return found[-1].upper() if found else None


def extract_free(text: str) -> str | None:
    """The final free-form answer: boxed, "the answer is ...", else the last non-empty line."""
    boxed = extract_boxed(text or "")
    if boxed is not None:
        return boxed
    for pat in (
        r"(?:the\s+)?final answer is[:\s]*(.+?)(?:\.\s|\n|$)",
        r"(?:the\s+)?answer is[:\s]*(.+?)(?:\.\s|\n|$)",
    ):
        m = re.search(pat, text or "", re.IGNORECASE)
        if m:
            return m.group(1)
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def extract_code(text: str) -> str:
    """The last fenced code block, preferring the post-``</think>`` region.

    Falls back to an unterminated fence, then to the raw text.
    """
    tail = after_think(text)
    scopes = ([tail] if tail != (text or "") else []) + [text or ""]
    for scope in scopes:
        blocks = _FENCE.findall(scope)
        if blocks:
            return blocks[-1]
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text or "", re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else (text or "")
