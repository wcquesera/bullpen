"""The graders of math, gpqa, bbh and gsm8k (the ``legacy`` adapter).

Kept exactly as the paper's collection graded these benchmarks; wrapped answers such
as ``\\boxed{\\text{A}}`` are unwrapped so the wrapping convention is not graded.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

#: ``\text{X}`` and similar wrappers around an answer.
_LATEX_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|mbox|textbf|textit)\s*\{([^{}]*)\}")

#: How many nested wrapper layers :func:`unwrap_latex_text` peels.
_UNWRAP_PASSES = 3

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

#: Tolerance on a numeric gsm8k match (so ``42.0`` matches ``42``).
_NUMERIC_TOL = 1e-4


def last_number(text: str) -> str | None:
    """The last number in ``text``, commas stripped."""
    nums = _NUMBER.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def unwrap_latex_text(value: str | None) -> str:
    """Replace ``\\text{X}``-style wrappers with their contents (nested ones too)."""
    if value is None:
        return ""
    out = str(value)
    for _ in range(_UNWRAP_PASSES):
        out, n = _LATEX_WRAP.subn(r"\1", out)
        if not n:
            break
    return out


def extract_boxed(text: str) -> str | None:
    """The contents of the last ``\\boxed{...}``, matched brace by brace."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def extract_gsm8k_pred(text: str) -> str | None:
    """The model's final number: boxed, then stated, then the last one written."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        n = last_number(m.group(1))
        if n is not None:
            return n
    for pat in (
        r"answer is[:\s]*\$?(-?\d[\d,]*\.?\d*)",
        r"answer[:\s]*\$?(-?\d[\d,]*\.?\d*)",
        r"####\s*(-?\d[\d,]*\.?\d*)",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")
    return last_number(text)


def grade_gsm8k(output: str, gold: str | None) -> bool:
    """Numeric match on the final number."""
    pred = extract_gsm8k_pred(output)
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < _NUMERIC_TOL
    except ValueError:
        return pred.strip() == gold.strip()


def _norm_math(value: str | None) -> str:
    """Normalise a LaTeX answer for exact match (``\\text{...}`` is deleted, e.g. units)."""
    if value is None:
        return ""
    s = value.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace(" ", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def _sympy_equivalent(pred: str, gold: str) -> bool:
    """Whether two LaTeX expressions simplify to the same thing (``False`` if sympy fails)."""
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex

        def parse(expr: str) -> Any:
            try:
                return parse_latex(expr)
            except Exception:  # noqa: BLE001 — any parse failure means "try sympify"
                return sympify(expr)

        return bool(simplify(parse(pred) - parse(gold)) == 0)
    except Exception:  # noqa: BLE001 — sympy absent, or the pair is not parseable
        return False


def grade_math(output: str, gold: str | None) -> bool:
    """Boxed answer, then a stated one, then the last number; then equivalence."""
    pred = extract_boxed(output)
    if pred is None:
        # plain-text finals ("The answer is 42"), before the last-number fallback
        for pat in (
            r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([^$\n]+)\$?",
            r"[Aa]nswer[:\s]*\$?\\?boxed\{([^}]+)\}",
        ):
            m = re.search(pat, output)
            if m:
                pred = m.group(1).strip()
                break
    if pred is None:
        pred = last_number(output)
    gold_boxed = extract_boxed(gold) if "\\boxed" in (gold or "") else gold
    if pred is None or gold_boxed is None:
        return False
    if _norm_math(pred) == _norm_math(gold_boxed):
        return True
    return _sympy_equivalent(pred, gold_boxed)


def extract_mc(text: str) -> str | None:
    """An A-D option letter. Some sources label their options lowercase."""
    text = unwrap_latex_text(text)
    m = re.search(r"\\boxed\{\s*\(?\s*([A-Da-d])\s*\)?\s*[.,]?\s*\}", text)
    if m:
        return m.group(1).upper()
    for pat in (
        r"answer is[:\s]*\(?([A-Da-d])\)?",
        r"answer[:\s]*\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\)\s*$",
        r"^\(?([A-Da-d])\)?$",
    ):
        m = re.search(pat, text.strip(), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"\b([A-Da-d])\b", text)
    return letters[-1].upper() if letters else None


def grade_gpqa(output: str, gold: str) -> bool:
    """Multiple choice over four options."""
    pred = extract_mc(output)
    return pred is not None and pred.upper() == gold.upper()


def extract_mc_bbh(text: str) -> str | None:
    """An A-Z option letter — BBH's option lists run well past D."""
    text = unwrap_latex_text(text)
    m = re.search(r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?\s*[.,]?\s*\}", text)
    if m:
        return m.group(1).upper()
    for pat in (
        r"answer is[:\s]*\(?\s*([A-Z])\s*\)?",
        r"answer[:\s]*\(?\s*([A-Z])\s*\)?",
        r"\(([A-Z])\)\s*$",
        r"^\(?([A-Z])\)?$",
    ):
        m = re.search(pat, text.strip(), re.MULTILINE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"\(([A-Z])\)", text)
    return letters[-1].upper() if letters else None


def _norm_free(value: str | None) -> str:
    """Normalise a free-form answer for exact match."""
    if value is None:
        return ""
    s = unwrap_latex_text(value)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("$", "").replace("**", "")
    s = s.strip().strip("\"'` ")
    s = s.strip().rstrip(".").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return " ".join(s.lower().split())


def _bbh_free_pred(output: str) -> str:
    """The model's final free-form answer, pulled out of a chain of thought."""
    boxed = extract_boxed(output)
    if boxed is not None:
        return boxed
    for pat in (r"answer is[:\s]*(.+?)(?:\.|\n|$)", r"final answer[:\s]*(.+?)(?:\.|\n|$)"):
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            return m.group(1)
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else output


def grade_bbh(output: str, gold: str | None) -> bool:
    """Letter match if the gold is a bare capital letter, else normalised exact match."""
    gold = (gold or "").strip()
    if re.fullmatch(r"[A-Z]", gold):
        pred = extract_mc_bbh(output)
        return pred is not None and pred == gold
    return _norm_free(_bbh_free_pred(output)) == _norm_free(gold)


#: The legacy graders by benchmark name.
LEGACY_GRADERS: dict[str, Callable[[str, Any], bool]] = {
    "gsm8k": grade_gsm8k,
    "math": grade_math,
    "gpqa": grade_gpqa,
    "bbh": grade_bbh,
}


def grade_legacy(benchmark: str, output: str, gold: str | None) -> bool:
    """Grade one answer with the legacy grader named by ``benchmark`` (``KeyError`` if none)."""
    return bool(LEGACY_GRADERS[benchmark](output, gold))
