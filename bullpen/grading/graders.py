"""MCQ letter match, free-form normalised match and IFEval strict checks, plus :func:`grade_item`.

:func:`grade_item` routes on (benchmark, model) through
:data:`~bullpen.grading.adapters.ADAPTER_TABLE`; an item's own ``format`` field wins
when the two disagree, keeping the think-first variant for reasoning models.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

from loguru import logger

from bullpen.grading.adapters import (
    ADAPTERS,
    DEFAULT_REASONING_MARKERS,
    LETTERS,
    Adapter,
    resolve_adapter,
)
from bullpen.grading.extract import (
    extract_code,
    extract_free,
    extract_mcq,
    strip_special_tokens,
    think_first,
)
from bullpen.grading.legacy import grade_legacy, unwrap_latex_text
from bullpen.grading.sandbox import DEFAULT_SANDBOX, Sandbox, grade_code


# --------------------------------------------------------------------------- #
# format="mcq"
# --------------------------------------------------------------------------- #
def grade_mcq(output: str, gold: str | None, n_choices: int = len(LETTERS)) -> bool:
    """Correct iff the extracted option letter equals the gold letter."""
    pred = extract_mcq(output or "", n_choices)
    return pred is not None and pred == (gold or "").strip().upper()


# --------------------------------------------------------------------------- #
# format="free"
# --------------------------------------------------------------------------- #
#: A trailing parenthesised unit, as in asdiv's golds (``"9 (apples)"``); not part of the answer.
_UNIT_PAREN = re.compile(r"\s*\([^)]*\)\s*$")

#: Tolerance of the numeric fallback in :func:`free_match` (so ``204.`` matches ``204``).
_FREE_NUMERIC_TOL = 1e-6


def norm_free_answer(value: str | None) -> str:
    """Normalise a free-form answer for exact match (``\\text{...}`` keeps its contents)."""
    if value is None:
        return ""
    s = unwrap_latex_text(value)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("$", "").replace("**", "")
    s = s.strip().strip("\"'`").strip()
    s = s.rstrip(".").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return " ".join(_UNIT_PAREN.sub("", s).strip().lower().split())


def free_match(pred: str | None, gold: str | None) -> bool:
    """Normalised equality, then a numeric comparison."""
    p, g = norm_free_answer(pred), norm_free_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p.replace(",", "")) - float(g.replace(",", ""))) < _FREE_NUMERIC_TOL
    except ValueError:
        return False


def grade_free(output: str, gold: str | None) -> bool:
    """Correct iff the extracted final answer matches the gold after normalisation."""
    pred = extract_free(output)
    return pred is not None and free_match(pred, gold)


# --------------------------------------------------------------------------- #
# format="ifeval" — strict prompt-level constraint checking
# --------------------------------------------------------------------------- #
def ifeval_gold(instruction_id_list: Sequence[str], kwargs: Sequence[Mapping[str, Any]]) -> str:
    """Serialise one IFEval item's constraints into the string ``gold`` field."""
    clean = [{k: v for k, v in kw.items() if v is not None} for kw in kwargs]
    return json.dumps({"instruction_id_list": list(instruction_id_list), "kwargs": clean})


def _rel(count: int, relation: str | None, target: int) -> bool:
    """IFEval's two comparisons. Anything but ``"less than"`` means "at least"."""
    if relation == "less than":
        return count < target
    return count >= target


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


@cache
def _warn_langdetect_missing() -> None:
    """Warn once that the language items cannot be checked."""
    logger.warning(
        "langdetect not installed — ifeval 'language:response_language' items grade as "
        "incorrect (`uv add langdetect`)"
    )


def _detect_language(text: str) -> str | None:
    """The response's language, or ``None`` (which fails the check) if undetectable."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return str(detect(text))
    except ImportError:
        _warn_langdetect_missing()
        return None
    except Exception:  # noqa: BLE001 — the detector throws on empty/symbol-only text
        return None


def check_instruction(iid: str, kw: Mapping[str, Any], resp: str) -> bool:
    """Whether one IFEval instruction was followed (strict); an unknown id grades as failed."""
    r = resp
    if iid == "punctuation:no_comma":
        return "," not in r
    if iid == "change_case:english_lowercase":
        return r == r.lower()
    if iid == "change_case:english_capital":
        return r == r.upper()
    if iid == "change_case:capital_word_frequency":
        n = sum(1 for w in _words(r) if w.isupper() and w.isalpha())
        return _rel(n, kw.get("capital_relation"), int(kw.get("capital_frequency", 0)))
    if iid == "length_constraints:number_words":
        return _rel(len(_words(r)), kw.get("relation"), int(kw.get("num_words", 0)))
    if iid == "length_constraints:number_sentences":
        return _rel(len(_sentences(r)), kw.get("relation"), int(kw.get("num_sentences", 0)))
    if iid == "length_constraints:number_paragraphs":
        # IFEval splits paragraphs on the '***' divider, not on a blank line.
        paras = [p for p in re.split(r"\s*\*\*\*\s*", r) if p.strip()]
        return len(paras) == kw.get("num_paragraphs", 0)
    if iid == "length_constraints:nth_paragraph_first_word":
        paras = _paragraphs(r)
        n = int(kw.get("nth_paragraph", 1))
        if len(paras) < max(n, int(kw.get("num_paragraphs", n))):
            return False
        first = _words(paras[n - 1])
        return bool(first) and first[0].lower() == str(kw.get("first_word", "")).lower()
    if iid == "keywords:existence":
        return all(
            re.search(rf"\b{re.escape(str(k))}\b", r, re.IGNORECASE) for k in kw.get("keywords", [])
        )
    if iid == "keywords:forbidden_words":
        return not any(
            re.search(rf"\b{re.escape(str(k))}\b", r, re.IGNORECASE)
            for k in kw.get("forbidden_words", [])
        )
    if iid == "keywords:frequency":
        n = len(re.findall(rf"\b{re.escape(str(kw.get('keyword', '')))}\b", r, re.IGNORECASE))
        return _rel(n, kw.get("relation"), int(kw.get("frequency", 0)))
    if iid == "keywords:letter_frequency":
        n = r.lower().count(str(kw.get("letter", "")).lower())
        return _rel(n, kw.get("let_relation"), int(kw.get("let_frequency", 0)))
    if iid == "detectable_content:number_placeholders":
        return len(re.findall(r"\[[^\[\]]+\]", r)) >= int(kw.get("num_placeholders", 0))
    if iid == "detectable_content:postscript":
        marker = str(kw.get("postscript_marker", "P.S."))
        return bool(re.search(rf"(?m)^\s*{re.escape(marker)}", r, re.IGNORECASE))
    if iid == "detectable_format:number_bullet_lists":
        return len(re.findall(r"(?m)^\s*[\*\-]\s+\S", r)) == kw.get("num_bullets", 0)
    if iid == "detectable_format:number_highlighted_sections":
        # `**bold**` counts as one highlight, so collapse it to `*bold*` first.
        n = len(re.findall(r"\*[^\*\n]+\*", r.replace("**", "*")))
        return n >= int(kw.get("num_highlights", 0))
    if iid == "detectable_format:title":
        return bool(re.search(r"<<[^\n<>]+>>", r))
    if iid == "detectable_format:multiple_sections":
        spliter = str(kw.get("section_spliter", "Section"))
        n = len(re.findall(rf"(?m)^\s*{re.escape(spliter)}\s*\d+", r))
        return n >= int(kw.get("num_sections", 0))
    if iid == "detectable_format:json_format":
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.strip(), flags=re.IGNORECASE)
        try:
            json.loads(body)
        except ValueError:
            return False
        return True
    if iid == "detectable_format:constrained_response":
        return any(o in r for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))
    if iid == "startend:quotation":
        s = r.strip()
        return len(s) >= 2 and s.startswith('"') and s.endswith('"')
    if iid == "startend:end_checker":
        return r.strip().lower().endswith(str(kw.get("end_phrase", "")).strip().lower())
    if iid == "combination:two_responses":
        return len([p for p in r.split("******") if p.strip()]) == 2
    if iid == "combination:repeat_prompt":
        want = str(kw.get("prompt_to_repeat", "")).strip().lower()
        return bool(want) and r.strip().lower().startswith(want)
    if iid == "language:response_language":
        lang = _detect_language(r)
        return lang is not None and lang == str(kw.get("language", "")).lower()
    logger.warning(f"ifeval: unhandled instruction_id {iid!r} — grading as failed")
    return False


#: Every instruction id :func:`check_instruction` implements. An IFEval item
#: naming one outside this set grades as failed.
IFEVAL_INSTRUCTION_IDS: frozenset[str] = frozenset(
    {
        "punctuation:no_comma",
        "change_case:english_lowercase",
        "change_case:english_capital",
        "change_case:capital_word_frequency",
        "length_constraints:number_words",
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:letter_frequency",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:title",
        "detectable_format:multiple_sections",
        "detectable_format:json_format",
        "detectable_format:constrained_response",
        "startend:quotation",
        "startend:end_checker",
        "combination:two_responses",
        "combination:repeat_prompt",
        "language:response_language",
    }
)


def grade_ifeval(output: str, gold: str) -> bool:
    """Strict prompt-level accuracy: every instruction in the item must hold."""
    spec = json.loads(gold)
    return all(
        check_instruction(iid, kw, output or "")
        for iid, kw in zip(spec["instruction_id_list"], spec["kwargs"], strict=False)
    )


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def extract_answer(adapter: Adapter, response: str, n_choices: int = len(LETTERS)) -> str | None:
    """The answer ``adapter`` reads out of ``response`` (the whole response for ifeval)."""
    if adapter.fmt == "mcq":
        if adapter.strip_think:
            return think_first(extract_mcq, response, n_choices)
        return extract_mcq(response, n_choices)
    if adapter.fmt == "code":
        return extract_code(response)
    if adapter.fmt == "ifeval":
        return response
    if adapter.strip_think:
        return think_first(extract_free, response)
    return extract_free(response)


def grade_with(
    adapter: Adapter,
    bench: str,
    response: str,
    item: Mapping[str, Any],
    sandbox: Sandbox = DEFAULT_SANDBOX,
) -> bool:
    """Grade one response with an already-resolved adapter."""
    gold = item.get("gold")
    if adapter.fmt == "legacy":
        return grade_legacy(bench, response, gold)
    if adapter.fmt == "code":
        return grade_code(response, gold, sandbox)
    if adapter.fmt == "ifeval":
        return grade_ifeval(response, gold)
    n_choices = int(item.get("n_choices") or len(LETTERS))
    pred = extract_answer(adapter, response, n_choices)
    if adapter.fmt == "mcq":
        return pred is not None and pred == (gold or "").strip().upper()
    return pred is not None and free_match(pred, gold)


#: Item ``format`` values that may override the table's routing.
_ITEM_FORMATS = ("mcq", "free", "code", "ifeval")


def grade_item(
    bench: str,
    output: str,
    item: Mapping[str, Any],
    model: str | None = None,
    sandbox: Sandbox = DEFAULT_SANDBOX,
    markers: Sequence[str] = DEFAULT_REASONING_MARKERS,
) -> bool:
    """Grade one response against one probe item: the entry point for collection code.

    An item's explicit ``format`` overrides the table, except for legacy benchmarks.
    """
    output = strip_special_tokens(output)
    adapter = resolve_adapter(bench, model, markers)
    fmt = item.get("format")
    if adapter.fmt != "legacy" and fmt in _ITEM_FORMATS and fmt != adapter.fmt:
        think = f"{fmt}_think"
        adapter = ADAPTERS[think if adapter.strip_think and think in ADAPTERS else str(fmt)]
    return grade_with(adapter, bench, output, item, sandbox)
