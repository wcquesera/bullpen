"""Benchmark graders: one response + one probe item -> right or wrong.

``grade_item(bench, output, item, model)`` is the entry point::

    adapters.py  the benchmark x model -> protocol table, and the lookup over it
    extract.py   answer extraction, think-trace and special-token handling
    graders.py   MCQ / free / IFEval grading, and the grade_item dispatch
    sandbox.py   the code sandbox, and the execute-the-model's-code grader
    legacy.py    the graders of math / gpqa / bbh / gsm8k

SAFETY: grading a ``code`` benchmark executes model-generated Python. See
:mod:`bullpen.grading.sandbox`; :meth:`~bullpen.grading.sandbox.Sandbox.backend`
reports which containment is in force.
"""

from bullpen.grading.adapters import (
    ADAPTER_TABLE,
    ADAPTERS,
    BENCH_ADAPTER,
    BENCH_FORMAT,
    CODE_INSTR,
    DEFAULT_REASONING_MARKERS,
    FREE_INSTR,
    LETTERS,
    MCQ_INSTR,
    Adapter,
    adapter_name,
    is_routed,
    model_class,
    resolve_adapter,
    unrouted,
)
from bullpen.grading.extract import (
    after_think,
    extract_code,
    extract_free,
    extract_mcq,
    strip_special_tokens,
    think_first,
)
from bullpen.grading.graders import (
    IFEVAL_INSTRUCTION_IDS,
    check_instruction,
    extract_answer,
    free_match,
    grade_free,
    grade_ifeval,
    grade_item,
    grade_mcq,
    grade_with,
    ifeval_gold,
    norm_free_answer,
)
from bullpen.grading.legacy import (
    LEGACY_GRADERS,
    extract_boxed,
    grade_bbh,
    grade_gpqa,
    grade_gsm8k,
    grade_legacy,
    grade_math,
    unwrap_latex_text,
)
from bullpen.grading.sandbox import (
    CODE_GUARD,
    DEFAULT_SANDBOX,
    Sandbox,
    build_code_program,
    bwrap_available,
    code_gold,
    grade_code,
)

__all__ = [
    "ADAPTERS",
    "ADAPTER_TABLE",
    "BENCH_ADAPTER",
    "BENCH_FORMAT",
    "CODE_GUARD",
    "CODE_INSTR",
    "DEFAULT_REASONING_MARKERS",
    "DEFAULT_SANDBOX",
    "FREE_INSTR",
    "IFEVAL_INSTRUCTION_IDS",
    "LEGACY_GRADERS",
    "LETTERS",
    "MCQ_INSTR",
    "Adapter",
    "Sandbox",
    "adapter_name",
    "after_think",
    "build_code_program",
    "bwrap_available",
    "check_instruction",
    "code_gold",
    "extract_answer",
    "extract_boxed",
    "extract_code",
    "extract_free",
    "extract_mcq",
    "free_match",
    "grade_bbh",
    "grade_code",
    "grade_free",
    "grade_gpqa",
    "grade_gsm8k",
    "grade_ifeval",
    "grade_item",
    "grade_legacy",
    "grade_math",
    "grade_mcq",
    "grade_with",
    "ifeval_gold",
    "is_routed",
    "model_class",
    "norm_free_answer",
    "resolve_adapter",
    "strip_special_tokens",
    "think_first",
    "unrouted",
    "unwrap_latex_text",
]
