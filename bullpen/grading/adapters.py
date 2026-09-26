"""The benchmark x model -> answer-protocol lookup table.

An adapter bundles one (benchmark, model) cell's protocol: the prompt suffix, whether
to read the post-``</think>`` region first, and the grader family::

    fmt="mcq"     N-way multiple choice, gold is an option letter (A..Z)
    fmt="free"    gold is a literal string matched after normalisation
    fmt="code"    gold is JSON test code, run against the model's code in a sandbox
    fmt="ifeval"  gold is JSON (instruction_id_list, kwargs), checked programmatically
    fmt="legacy"  defer to :mod:`bullpen.grading.legacy`
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bullpen.grading.legacy import LEGACY_GRADERS

#: Option labels, in order. Index i is the label of the i-th choice.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Prompt suffixes, kept beside the graders that read the answers they elicit.
MCQ_INSTR = (
    "\n\nThink step by step, then give the final answer as the option letter inside \\boxed{}."
)
FREE_INSTR = "\n\nThink step by step, then give the final answer inside \\boxed{}."
CODE_INSTR = (
    "\n\nWrite the complete function. Put your final solution in a single ```python code block."
)


@dataclass(frozen=True)
class Adapter:
    """One (benchmark, model) answer protocol.

    ``strip_think`` selects :func:`bullpen.grading.extract.think_first` (post-``</think>``
    region first, then the full text).
    """

    name: str
    fmt: str
    #: the answer-eliciting text appended to the prompt at prep time
    suffix: str
    strip_think: bool


_ADAPTER_SPECS: dict[str, dict[str, object]] = {
    # name          format    prompt suffix  prefer the post-</think> region?
    "mcq": {"fmt": "mcq", "suffix": MCQ_INSTR, "strip_think": False},
    "mcq_think": {"fmt": "mcq", "suffix": MCQ_INSTR, "strip_think": True},
    "free": {"fmt": "free", "suffix": FREE_INSTR, "strip_think": False},
    "free_think": {"fmt": "free", "suffix": FREE_INSTR, "strip_think": True},
    "code": {"fmt": "code", "suffix": CODE_INSTR, "strip_think": False},
    "ifeval": {"fmt": "ifeval", "suffix": "", "strip_think": False},
    # `legacy` defers to bullpen.grading.legacy (math/gpqa/bbh/gsm8k)
    "legacy": {"fmt": "legacy", "suffix": "", "strip_think": False},
}

ADAPTERS: dict[str, Adapter] = {
    name: Adapter(
        name=name,
        fmt=str(spec["fmt"]),
        suffix=str(spec["suffix"]),
        strip_think=bool(spec["strip_think"]),
    )
    for name, spec in _ADAPTER_SPECS.items()
}

#: benchmark -> its model-independent base adapter.
BENCH_ADAPTER: dict[str, str] = {
    "mmlu": "mcq",
    "mmlu_pro": "mcq",
    "mmlu_med": "mcq",
    "arc_challenge": "mcq",
    "hellaswag": "mcq",
    "winogrande": "mcq",
    "musr": "mcq",
    "truthfulqa": "mcq",
    "medqa": "mcq",
    "medmcqa": "mcq",
    "piqa": "mcq",
    "social_iqa": "mcq",
    "logiqa": "mcq",
    "mathqa": "mcq",
    "aime_2024": "free",
    "pubmedqa": "free",
    "asdiv": "free",
    "humaneval": "code",
    "mbpp": "code",
    "ifeval": "ifeval",
    "gpqa_diamond": "mcq",
    "gpqa_extended": "mcq",
    "gpqa_main": "mcq",
    "math": "legacy",
    "gpqa": "legacy",
    "bbh": "legacy",
    "gsm8k": "legacy",
}

#: (benchmark selector, model selector) -> {base adapter: replacement adapter}.
MODEL_ADAPTER_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    # reasoning models: read the post-</think> region first (not applied to legacy benches)
    ("*", "@reasoning"): {"mcq": "mcq_think", "free": "free_think"},
}

#: Prefixes whose sub-benchmarks inherit a parent's adapter (``mmlu_anatomy`` -> ``mmlu``);
#: an exact :data:`BENCH_ADAPTER` entry wins.
_BENCH_PREFIX_INHERIT: dict[str, str] = {"mmlu_": "mmlu", "gpqa_": "gpqa"}


def _build_adapter_table() -> dict[str, dict[str, str]]:
    """benchmark -> {model selector -> adapter name}.

    Model selectors, resolved most-specific-first by :func:`adapter_name`:
    an exact model key, then ``@reasoning``/``@instruct``, then ``*``.
    """
    table: dict[str, dict[str, str]] = {}
    for bench, base in BENCH_ADAPTER.items():
        row = {"*": base}
        for (bench_sel, model_sel), remap in MODEL_ADAPTER_OVERRIDES.items():
            if bench_sel in ("*", bench) and base in remap:
                row[model_sel] = remap[base]
        table[bench] = row
    return table


ADAPTER_TABLE: dict[str, dict[str, str]] = _build_adapter_table()

#: benchmark -> format string, for the non-legacy benchmarks.
BENCH_FORMAT: dict[str, str] = {
    bench: ADAPTERS[name].fmt for bench, name in BENCH_ADAPTER.items() if name != "legacy"
}

#: Case-insensitive substrings of a model id that mark it as a reasoning model (for
#: grading; distinct from ``config/collect.yaml``'s ``thinking_markers``, which set the
#: generation budget).
DEFAULT_REASONING_MARKERS: tuple[str, ...] = (
    "r1-distill",
    "deepseek-r1",
    "qwen3",
    "qwq",
    "reasoning",
    "nemotron",
    "magistral",
    "openthinker",
    "sky-t1",
    "skywork-or1",
    "mimo",
    "s1.1",
    "exaone-deep",
    "glm-z1",
    "marco-o1",
)


def model_class(
    model: str | None, markers: Sequence[str] = DEFAULT_REASONING_MARKERS
) -> str | None:
    """``'@reasoning'`` / ``'@instruct'`` for a model key, or ``None`` if no model is named."""
    if not model:
        return None
    low = model.lower()
    return "@reasoning" if any(m.lower() in low for m in markers) else "@instruct"


def adapter_name(
    bench: str,
    model: str | None = None,
    markers: Sequence[str] = DEFAULT_REASONING_MARKERS,
) -> str:
    """The adapter name for one (benchmark, model) cell.

    An unrecognised benchmark falls back to ``legacy`` if a legacy grader is named
    after it, else to ``free``.
    """
    row = ADAPTER_TABLE.get(bench)
    if row is None:
        for prefix, parent in _BENCH_PREFIX_INHERIT.items():
            if bench.startswith(prefix) and parent in ADAPTER_TABLE:
                row = ADAPTER_TABLE[parent]
                break
    if row is None:
        return "legacy" if bench in LEGACY_GRADERS else "free"
    for selector in (model, model_class(model, markers)):
        if selector and selector in row:
            return row[selector]
    return row["*"]


def resolve_adapter(
    bench: str,
    model: str | None = None,
    markers: Sequence[str] = DEFAULT_REASONING_MARKERS,
) -> Adapter:
    """(benchmark, model) -> the adapter that handles it."""
    return ADAPTERS[adapter_name(bench, model, markers)]


def is_routed(bench: str) -> bool:
    """Whether ``bench`` has a table entry rather than hitting the fallback."""
    if bench in ADAPTER_TABLE:
        return True
    return any(
        bench.startswith(prefix) and parent in ADAPTER_TABLE
        for prefix, parent in _BENCH_PREFIX_INHERIT.items()
    )


def unrouted(benches: Iterable[str]) -> list[str]:
    """Those of ``benches`` this table does not route, sorted and deduplicated."""
    return sorted({b for b in benches if not is_routed(b)})
