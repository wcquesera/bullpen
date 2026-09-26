"""Readouts on the item (question) axis."""

from __future__ import annotations

import numpy as np

from bullpen.evaluation.battery.constants import (
    NDCG_K,
    UNIT_BENCHMARK_CLUSTERS,
    UNIT_MODELS,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _absent,
    _labelled_rows,
    _oof_logistic,
    _permuted_null,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.shared import (
    _cluster_ci,
    _mean_or_nan,
    _or_nan,
    _resample,
)
from bullpen.evaluation.groups import DECISION, STRUCTURAL
from bullpen.evaluation.metrics import row_auroc, spearman_rho

# --------------------------------------------------------------------------- #
# item-axis readouts
# --------------------------------------------------------------------------- #


#: Published boards the top-K shortlist readout will use, best coverage first.
TOPK_BOARDS: tuple[str, ...] = (
    "open_llm_leaderboard",
    "open_llm_leaderboard_v2",
    "livebench_aggregate",
    "chatbot_arena_elo",
)

#: Models a board must list before a top-K shortlist is a classification.
TOPK_COVERAGE_FLOOR: int = 20

#: Shortlist size — a deployment shortlist, :data:`NDCG_K`'s reason and its value.
TOPK_K: int = NDCG_K


def _leave_column_out_ability(M: np.ndarray) -> np.ndarray:
    """[M, Q] each model's mean accuracy EXCLUDING the column it is paired with."""
    values = np.asarray(M, dtype=np.float64)
    n_other = max(values.shape[1] - 1, 1)
    return (values.sum(axis=1, keepdims=True) - values) / n_other


def _item_discrimination(M: np.ndarray) -> np.ndarray:
    """[Q] point-biserial of each column against ability — the 2PL ``a`` proxy."""
    values = np.asarray(M, dtype=np.float64)
    theta = _leave_column_out_ability(values)
    x = values - values.mean(axis=0)
    t = theta - theta.mean(axis=0)
    den = np.sqrt((x**2).sum(axis=0) * (t**2).sum(axis=0))
    out = np.full(values.shape[1], np.nan)
    ok = den > 0
    out[ok] = (x * t).sum(axis=0)[ok] / den[ok]
    return out


def _item_fisher(M: np.ndarray) -> np.ndarray:
    """[Q] 2PL item information at the cohort's mean ability, ``a^2 p (1 - p)``."""
    values = np.asarray(M, dtype=np.float64)
    p = np.clip(values.mean(axis=0), 0.0, 1.0)
    return _item_discrimination(values) ** 2 * p * (1.0 - p)


def _question_rank_ci(ctx: TaskContext, truth: np.ndarray, predicted: np.ndarray) -> TaskMetrics:
    """Interval of a Spearman taken ACROSS questions, clustered on their benchmarks."""
    a = np.asarray(truth, dtype=np.float64)
    b = np.asarray(predicted, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    bench = np.asarray(ctx.bench, dtype=int)[keep]

    def rho(counts: np.ndarray) -> float:
        kept = _resample(counts, bench)
        return _or_nan(spearman_rho, a[kept], b[kept])

    return _cluster_ci(
        rho,
        _or_nan(spearman_rho, a, b),
        ctx.n_benchmarks,
        UNIT_BENCHMARK_CLUSTERS,
        ctx.seed,
    )


def _rank_agreement(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Spearman over the questions where BOTH sides are measured, NaN if too few."""
    a = np.asarray(truth, dtype=np.float64)
    b = np.asarray(predicted, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2:
        return float("nan")
    return _or_nan(spearman_rho, a[keep], b[keep])


@task("item_information", primary="spearman_discrimination", group=STRUCTURAL)
def item_information(ctx: TaskContext) -> TaskMetrics:
    """Rank the QUESTIONS by how much they discriminate — decoded from the bank alone."""
    truth_disc = _item_discrimination(ctx.R)
    truth_fisher = _item_fisher(ctx.R)
    difficulty = ctx.A.mean(axis=0)
    decoded = ctx.decoded_cells
    predicted_disc = _item_discrimination(decoded)
    return {
        "spearman_discrimination": _rank_agreement(truth_disc, predicted_disc),
        "spearman_discrimination_permuted_null": _mean_or_nan(
            [_rank_agreement(truth_disc, _item_discrimination(P)) for P in ctx.permuted_decodes]
        ),
        "spearman_fisher": _rank_agreement(truth_fisher, _item_fisher(decoded)),
        "spearman_fisher_permuted_null": _mean_or_nan(
            [_rank_agreement(truth_fisher, _item_fisher(P)) for P in ctx.permuted_decodes]
        ),
        # arm-independent, and the baseline the two above are only worth reading
        # against: how much of the item ordering is difficulty and nothing else
        "spearman_difficulty_only": _rank_agreement(truth_fisher, difficulty * (1.0 - difficulty)),
        "n_questions": int(np.isfinite(truth_disc).sum()),
        **_question_rank_ci(ctx, truth_disc, predicted_disc),
    }


@task("external_topk", primary="auroc_topk", group=DECISION)
def external_topk(ctx: TaskContext) -> TaskMetrics:
    """Pick the shortlist: will this unseen model land in a published board's top K?"""
    for name in TOPK_BOARDS:
        column = ctx.external.get(name)
        if column is None:
            continue
        rows = _labelled_rows(np.isfinite(np.asarray(column, dtype=np.float64)))
        if rows.size >= TOPK_COVERAGE_FLOOR:
            break
    else:
        return _absent(
            "auroc_topk",
            f"no published board on this cut lists {TOPK_COVERAGE_FLOOR} of the "
            f"{ctx.n_models} scored models; a {TOPK_K}-positive shortlist below that is "
            "a classification over a handful of rows",
        )
    y = np.asarray(column, dtype=np.float64)
    # the shortlist never exceeds a third of the labelled block: a top-10 of 30 is
    # a shortlist, a top-10 of 12 is the board
    k = min(TOPK_K, rows.size // 3)
    shortlist = rows[np.argsort(-y[rows])[:k]]
    labels = np.zeros(ctx.n_models, dtype=int)
    labels[shortlist] = 1
    label = labels[rows]
    _, score = _oof_logistic(ctx, labels, rows)
    competence = ctx.A[rows].mean(axis=1)
    picked = np.argsort(-score)[:k]
    auroc = _or_nan(row_auroc, label, score)
    return {
        "auroc_topk": auroc,
        "auroc_permuted_null": _permuted_null(
            label, lambda shuffled: _or_nan(row_auroc, shuffled, score), ctx.seed
        ),
        "auroc_competence_only": _or_nan(row_auroc, label, competence),
        "precision_at_k": float(label[picked].mean()),
        "precision_at_k_chance": float(k / rows.size),
        "board": name,
        "k": int(k),
        "n_labelled": int(rows.size),
        **_cluster_ci(
            lambda counts: row_auroc(*(a[_resample(counts)] for a in (label, score))),
            auroc,
            int(rows.size),
            UNIT_MODELS,
            ctx.seed,
        ),
    }
