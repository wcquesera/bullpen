"""Readouts supervised by published leaderboard columns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from bullpen.config import load_evaluate
from bullpen.evaluation.battery.constants import UNIT_MODELS
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _absent,
    _folds_over,
    _labelled_rows,
    _oof_logistic,
    _oof_scalar,
    _permuted_null,
    _scalar_transfer,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.shared import _cluster_ci, _or_nan, _resample
from bullpen.evaluation.groups import STRUCTURAL
from bullpen.evaluation.metrics import r2_score, spearman_rho

# --------------------------------------------------------------------------- #
# published leaderboards
# --------------------------------------------------------------------------- #
#: Metric every published-leaderboard readout headlines.
EXTERNAL_PRIMARY: str = "spearman"


@dataclass(frozen=True)
class ExternalSpec:
    """One published column and the battery task supervised by it."""

    task: str
    #: key into ``ctx.external``, from the label tables beside the slice
    table: str
    #: noun phrase naming the column, read into the generated docstring
    what: str


#: The published-leaderboard readouts: one per published column.
EXTERNAL_SPECS: tuple[ExternalSpec, ...] = (
    ExternalSpec("chatbot_arena_elo", "chatbot_arena_elo", "the LMSYS Chatbot Arena Elo rating"),
    ExternalSpec(
        "alpacaeval_winrate",
        "alpacaeval_winrate",
        "the AlpacaEval 2 length-controlled win rate",
    ),
    ExternalSpec(
        "open_llm_leaderboard", "open_llm_leaderboard", "the Open LLM Leaderboard v1 average"
    ),
    ExternalSpec(
        "open_llm_leaderboard_v2",
        "open_llm_leaderboard_v2",
        "the Open LLM Leaderboard v2 average",
    ),
    ExternalSpec("i_math_score", "open_llm_v2_math", "the Open LLM v2 MATH Lvl 5 sub-score"),
    ExternalSpec("i_gpqa_score", "open_llm_v2_gpqa", "the Open LLM v2 GPQA sub-score"),
    ExternalSpec("instruction_following", "open_llm_v2_ifeval", "the published IFEval score"),
    # scraped boards (external_signals.json beside the slice)
    ExternalSpec("ext_arena_rating", "lmarena_style_rating_overall", "the LMArena rating"),
    ExternalSpec(
        "ext_arena_rating_coding", "lmarena_style_rating_coding", "the LMArena coding rating"
    ),
    ExternalSpec(
        "ext_arena_style_delta",
        "lmarena_style_style_delta_overall",
        "the LMArena style-control delta (rating minus style-controlled rating)",
    ),
    ExternalSpec(
        "ext_arena_style_delta_creative",
        "lmarena_style_style_delta_creative_writing",
        "the LMArena creative-writing style-control delta",
    ),
    ExternalSpec(
        "ext_arena_style_delta_math",
        "lmarena_style_style_delta_math",
        "the LMArena math style-control delta",
    ),
    ExternalSpec("ext_ayumi_rp_rank", "ayumi_rp_v3_rank_score", "the Ayumi role-play rank score"),
    ExternalSpec("ext_ayumi_rp_iq", "ayumi_rp_v3_iq_score", "the Ayumi role-play IQ score"),
    ExternalSpec(
        "ext_canaicode_junior",
        "canaicode_junior_v2_pass_rate_best",
        "the can-ai-code junior pass rate",
    ),
    ExternalSpec(
        "ext_canaicode_senior",
        "canaicode_senior_pass_rate_best",
        "the can-ai-code senior pass rate",
    ),
    ExternalSpec(
        "ext_livebench_language", "livebench_category_language", "the LiveBench language score"
    ),
    ExternalSpec(
        "ext_moral_machine_human_distance",
        "moral_machine_distance_to_human",
        "the Moral Machine distance to human preferences",
    ),
    ExternalSpec("ext_llmmap_refusal", "llmmap_refusal_rate", "the LLMmap probe refusal rate"),
)

#: ``task -> the published column it reads``, for the coverage check before scoring.
EXTERNAL_TABLE_BY_TASK: dict[str, str] = {spec.task: spec.table for spec in EXTERNAL_SPECS}

#: The generated docstring of every published-leaderboard readout.
_EXTERNAL_DOC = """Predict {what} from the embedding, out of fold.

    A ridge readout scored by Spearman, beside a model-permuted null and a
    competence-only (mean accuracy) baseline.
    """


def external_coverage_floor() -> int:
    """Slice models a published board must score before its readout is a number."""
    return load_evaluate().min_external_coverage


#: The headline of every closed-set readout registered off a label table — the model cards and
#: the external survey.
CLOSED_SET_PRIMARY: str = "macro_f1"


def _labelled_scalar_transfer(
    ctx: TaskContext, y: np.ndarray, rows: np.ndarray, groups: np.ndarray | None = None
) -> TaskMetrics:
    """Rank transfer to one per-model label, with the competence baseline beside it."""
    out = _scalar_transfer(ctx, y, rows, primary=EXTERNAL_PRIMARY, groups=groups)
    target = y[rows]
    by_competence = _oof_scalar(
        ctx.A[rows].mean(axis=1)[:, None], target, int(out["n_folds"]), ctx.seed, groups
    )
    out["spearman_competence_only"] = _or_nan(spearman_rho, target, by_competence)
    out["r2_competence_only"] = _or_nan(r2_score, target, by_competence)
    return out


def _closed_set_transfer(
    ctx: TaskContext,
    column: np.ndarray,
    rows: np.ndarray,
    classes: Sequence[str],
    groups: np.ndarray | None = None,
) -> TaskMetrics:
    """Out-of-fold closed-set decode of one per-model categorical label."""
    predicted, _ = _oof_logistic(ctx, column, rows, groups)
    truth = column[rows]
    labels = list(classes)
    n_folds = _folds_over(rows, ctx, groups)

    def macro_f1(against: np.ndarray) -> float:
        return float(f1_score(against, predicted, average="macro", labels=labels, zero_division=0))

    def macro_f1_over(counts: np.ndarray) -> float:
        """Macro-F1 on one resample of the labelled models, over the same class set."""
        kept = _resample(counts)
        return float(
            f1_score(truth[kept], predicted[kept], average="macro", labels=labels, zero_division=0)
        )

    return {
        CLOSED_SET_PRIMARY: macro_f1(truth),
        f"{CLOSED_SET_PRIMARY}_permuted_null": _permuted_null(truth, macro_f1, ctx.seed),
        "accuracy": float((truth == predicted).mean()),
        "n_classes": len(labels),
        "n_labelled": int(rows.size),
        # the two columns a reader needs to tell a grouped row from a random one:
        # a grouped split is capped by the number of families, not by the rows
        "n_folds": n_folds,
        "grouped_cv": float(groups is not None),
        **_cluster_ci(macro_f1_over, macro_f1(truth), int(rows.size), UNIT_MODELS, ctx.seed),
    }


def _not_a_closed_set(task: str, n_classes: int, min_members: int) -> TaskMetrics:
    """The shared note for a block whose class set collapsed under its own floor."""
    return _absent(
        CLOSED_SET_PRIMARY,
        f"{n_classes} class(es) of {task} clear the {min_members}-member floor on this "
        f"block; that is not a closed-set problem this cut can pose",
    )


def _external_transfer(ctx: TaskContext, table: str) -> TaskMetrics:
    """The shared body of every published-leaderboard readout."""
    column = ctx.external.get(table)
    if column is None:
        return _absent(
            EXTERNAL_PRIMARY,
            f"the cut carries no {table} column",
        )
    y = np.asarray(column, dtype=np.float64)
    rows = _labelled_rows(np.isfinite(y))
    floor = external_coverage_floor()
    if rows.size < floor:
        return _absent(
            EXTERNAL_PRIMARY,
            f"{table} lists {rows.size} of the {ctx.n_models} scored models, below the "
            f"{floor}-model coverage floor; a ridge fitted out of fold on that many rows "
            f"reports its own shrinkage, not the bank",
        )
    out = _labelled_scalar_transfer(ctx, y, rows)
    out["n_unlisted"] = int(ctx.n_models - rows.size)
    return out


def _register_external(spec: ExternalSpec) -> None:
    """Register one published-leaderboard readout from its spec."""

    def readout(ctx: TaskContext) -> TaskMetrics:
        return _external_transfer(ctx, spec.table)

    readout.__name__ = spec.task
    readout.__doc__ = _EXTERNAL_DOC.format(what=spec.what, table=spec.table)
    task(spec.task, primary=EXTERNAL_PRIMARY, group=STRUCTURAL)(readout)


for _spec in EXTERNAL_SPECS:
    _register_external(_spec)
