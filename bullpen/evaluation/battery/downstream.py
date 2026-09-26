"""Downstream tasks: specialisation, cold start, probes, knapsack."""

from __future__ import annotations

import numpy as np
from loguru import logger

from bullpen.evaluation.battery.constants import (
    HALF_SPLIT_DRAWS,
    HEADLINE_BUDGET_GB,
    KNAPSACK_DRAWS,
    KNAPSACK_NULL_SQUADS,
    KNAPSACK_PERMUTATIONS,
    NEIGHBOUR_METRIC,
    RELIABILITY_DRAWS,
    TRANSFER_PERMUTATIONS,
    UNIT_BENCHMARK_CLUSTERS,
    UNIT_MODELS,
    VRAM_BUDGETS_GB,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics, _permutation_orders
from bullpen.evaluation.battery.helpers import (
    _competence_r2,
    _greedy_knapsack,
    _permuted_null,
    _portfolio_row,
    _random_squad,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.shared import (
    _cluster_ci,
    _clustered_question_ci,
    _headroom_key,
    _mean_or_nan,
    _neighbour_k,
    _no_ci,
    _or_nan,
    _resample,
    _retention_key,
    _upward_headroom,
)
from bullpen.evaluation.groups import DOWNSTREAM
from bullpen.evaluation.metrics import (
    column_auroc,
    neighbour_predict,
    oof_ridge_predict,
    pool_mean_rows,
    profile_target,
    profile_target_reliability,
    r2_score,
)


# --------------------------------------------------------------------------- #
# downstream tasks
# --------------------------------------------------------------------------- #
@task("specialisation", primary="r2_profile_pc2plus", group=DOWNSTREAM)
def specialisation(ctx: TaskContext) -> TaskMetrics:
    """Predict the competence-removed specialisation profile."""
    r2s: list[float] = []
    competence: list[float] = []
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for draw in range(HALF_SPLIT_DRAWS):
        _, target_cols = ctx.question_halves(draw)
        try:
            target = profile_target(ctx.A[:, target_cols])
        except ValueError as exc:
            logger.debug("specialisation draw {} has no profile target: {}", draw, exc)
            r2s.append(float("nan"))
            competence.append(_or_nan(_competence_r2, ctx, target_cols))
            continue
        predicted = oof_ridge_predict(ctx.Z, target, n_folds=ctx.n_folds, seed=ctx.seed)
        r2s.append(_or_nan(r2_score, target, predicted))
        parts.append((target, predicted))
        competence.append(_or_nan(_competence_r2, ctx, target_cols))
    ceiling = profile_target_reliability(ctx.A, n_draws=RELIABILITY_DRAWS, seed=ctx.seed)

    def permuted_r2(order: np.ndarray) -> float:
        """The same mean over half-splits, each target fitted from the bank in ``order``."""
        return _mean_or_nan(
            [
                _or_nan(
                    r2_score,
                    target,
                    oof_ridge_predict(ctx.Z[order], target, n_folds=ctx.n_folds, seed=ctx.seed),
                )
                for target, _ in parts
            ]
        )

    def r2_over(counts: np.ndarray) -> float:
        """The same mean over half-splits, on one resample of the models."""
        kept = _resample(counts)
        return _mean_or_nan(
            [_or_nan(r2_score, target[kept], predicted[kept]) for target, predicted in parts]
        )

    return {
        "r2_profile_pc2plus": float(np.nanmean(r2s)),
        # the anchor: an out-of-fold R^2 on a bank with no model information is
        # negative, not 0, for the reason _scalar_transfer's null is
        "r2_profile_pc2plus_permuted_null": _permuted_null(
            np.arange(ctx.n_models), permuted_r2, ctx.seed, TRANSFER_PERMUTATIONS
        ),
        "draw_lo": float(np.nanmin(r2s)),
        "draw_hi": float(np.nanmax(r2s)),
        "r2_competence": float(np.nanmean(competence)),
        "n_splits": HALF_SPLIT_DRAWS,
        # the ceiling, not decoration: quote it beside the R^2 above
        "target_reliability": ceiling.reliability,
        "target_reliability_lo": ceiling.lo,
        "target_reliability_hi": ceiling.hi,
        **_cluster_ci(r2_over, float(np.nanmean(r2s)), ctx.n_models, UNIT_MODELS, ctx.seed),
    }


@task("correctness_forecasting", primary="column_auroc", group=DOWNSTREAM)
def correctness_forecasting(ctx: TaskContext) -> TaskMetrics:
    """Rank models on a single question — the difficulty-controlled routing readout."""
    P = neighbour_predict(ctx.Z, ctx.A, k=_neighbour_k(ctx.n_models), metric=NEIGHBOUR_METRIC)
    per_question = column_auroc(P, ctx.R)
    null = column_auroc(pool_mean_rows(ctx.A, seed=ctx.seed), ctx.R)
    return {
        "column_auroc": float(np.nanmean(per_question)),
        "column_auroc_null": float(np.nanmean(null)),
        "n_questions": int(np.isfinite(per_question).sum()),
        "metric": NEIGHBOUR_METRIC,
        **_clustered_question_ci(ctx, per_question),
    }


@task("knapsack", primary=_headroom_key(HEADLINE_BUDGET_GB), group=DOWNSTREAM)
def knapsack(ctx: TaskContext) -> TaskMetrics:
    """Pick a model portfolio under a VRAM budget using predicted rows."""
    if ctx.vram_gb is None:
        return {
            _headroom_key(HEADLINE_BUDGET_GB): float("nan"),
            "note": "vram_gb not supplied; this task will not invent model weights",
            **_no_ci(),
        }
    gb = np.asarray(ctx.vram_gb, dtype=np.float64)
    weighted = np.isfinite(gb)
    if not weighted.any():
        return {
            _headroom_key(HEADLINE_BUDGET_GB): float("nan"),
            "note": "no model carries a finite vram_gb; there is nothing to pack",
            "n_models_weighted": 0,
            **_no_ci(),
        }
    if not weighted.all():
        logger.info(
            "{} of {} models carry no vram_gb and are excluded from every portfolio",
            int((~weighted).sum()),
            gb.size,
        )
    retentions: dict[int, list[float]] = {budget: [] for budget in VRAM_BUDGETS_GB}
    headrooms: dict[int, list[float]] = {budget: [] for budget in VRAM_BUDGETS_GB}
    chances: dict[int, list[float]] = {budget: [] for budget in VRAM_BUDGETS_GB}
    #: ``(scored question ids, predicted row, reference row, random-squad row)`` per draw at the
    #: HEADLINE budget — the unaggregated form of the three portfolio values, kept so the
    #: interval can be clustered on the benchmarks those questions belong to.
    headline_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    #: headline headroom of the same pick from a model-permuted bank, per draw
    permuted_headrooms: list[float] = []
    rng = np.random.default_rng(ctx.seed)
    orders = _permutation_orders(ctx.n_models, ctx.seed, KNAPSACK_PERMUTATIONS)
    for draw in range(KNAPSACK_DRAWS):
        select_cols, score_cols = ctx.question_halves(draw)
        A_select, A_score = ctx.A[:, select_cols], ctx.A[:, score_cols]
        P = neighbour_predict(
            ctx.Z, A_select, k=_neighbour_k(ctx.n_models), metric=NEIGHBOUR_METRIC
        )
        for budget in VRAM_BUDGETS_GB:
            squad = _greedy_knapsack(A_select, gb, budget)
            reference_row = _portfolio_row(A_score, squad)
            reference = float(reference_row.mean()) if squad else 0.0
            if reference <= 0:
                logger.debug("budget {}GB admits no model on draw {}", budget, draw)
                continue
            predicted_row = _portfolio_row(A_score, _greedy_knapsack(P, gb, budget))
            predicted = float(predicted_row.mean())
            # the same budget and the same squad size, picked without reading a row
            chance_row = np.mean(
                [
                    _portfolio_row(A_score, _random_squad(rng, gb, budget, len(squad)))
                    for _ in range(KNAPSACK_NULL_SQUADS)
                ],
                axis=0,
            )
            chance = float(chance_row.mean())
            retentions[budget].append(predicted / reference)
            headrooms[budget].append(_upward_headroom(predicted, chance, reference))
            chances[budget].append(chance / reference)
            if budget == HEADLINE_BUDGET_GB:
                headline_rows.append((score_cols, predicted_row, reference_row, chance_row))
                for order in orders:
                    P_null = neighbour_predict(
                        ctx.Z[order],
                        A_select,
                        k=_neighbour_k(ctx.n_models),
                        metric=NEIGHBOUR_METRIC,
                    )
                    null_row = _portfolio_row(A_score, _greedy_knapsack(P_null, gb, budget))
                    permuted_headrooms.append(
                        _upward_headroom(float(null_row.mean()), chance, reference)
                    )
    out: TaskMetrics = {
        _headroom_key(budget): _mean_or_nan(values) for budget, values in headrooms.items()
    }
    out.update(
        {_retention_key(budget): _mean_or_nan(values) for budget, values in retentions.items()}
    )
    out.update(
        {
            f"random_retention_at_{budget}gb": _mean_or_nan(values)
            for budget, values in chances.items()
        }
    )
    out[f"{_headroom_key(HEADLINE_BUDGET_GB)}_permuted_null"] = _mean_or_nan(permuted_headrooms)
    out["n_draws"] = KNAPSACK_DRAWS
    out["n_null_squads"] = KNAPSACK_NULL_SQUADS
    out["n_models_weighted"] = int(weighted.sum())
    bench = np.asarray(ctx.bench, dtype=int)

    def headroom_over(counts: np.ndarray) -> float:
        """The headline headroom, re-averaged over the questions a benchmark resample keeps."""
        return _mean_or_nan(
            [
                _upward_headroom(
                    float(predicted[kept].mean()),
                    float(chance[kept].mean()),
                    float(reference[kept].mean()),
                )
                for cols, predicted, reference, chance in headline_rows
                if (kept := _resample(counts, bench[cols])).size
            ]
        )

    out.update(
        _cluster_ci(
            headroom_over,
            float(out[_headroom_key(HEADLINE_BUDGET_GB)]),
            ctx.n_benchmarks,
            UNIT_BENCHMARK_CLUSTERS,
            ctx.seed,
        )
    )
    return out
