"""Shared heads, guards, interval helpers and metric-key builders."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from loguru import logger
from sklearn.linear_model import RidgeCV

from bullpen.evaluation.battery.constants import (
    CI_BOOT,
    MIN_CI_DRAWS,
    MIN_HEADROOM,
    NEIGHBOUR_K,
    STD_FLOOR,
    UNIT_BENCHMARK_CLUSTERS,
    UNIT_NONE,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.metrics import ALPHA, ALPHAS, bootstrap_ci


# --------------------------------------------------------------------------- #
# shared heads and guards
# --------------------------------------------------------------------------- #
def _or_nan(fn: Callable[..., float], *args: object, **kwargs: object) -> float:
    """Run a readout, returning NaN where its input is degenerate rather than raising."""
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        logger.debug("{} returned NaN on a degenerate unit: {}", fn.__name__, exc)
        return float("nan")


def _mean_or_nan(values: Sequence[float]) -> float:
    """Mean of the finite entries of ``values``, NaN when there are none."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


# --------------------------------------------------------------------------- #
# intervals
# --------------------------------------------------------------------------- #
def _no_ci(unit: str = UNIT_NONE) -> TaskMetrics:
    """The interval keys for a cell that has no measurement behind it."""
    return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_units": 0, "resample_unit": unit}


def _mean_ci(values: Sequence[float] | np.ndarray, unit: str, seed: int) -> TaskMetrics:
    """Interval for a primary that IS a mean over one score per independent unit."""
    try:
        ci = bootstrap_ci(np.asarray(values, dtype=np.float64), n_boot=CI_BOOT, seed=seed)
    except ValueError as exc:
        logger.debug("no interval over {}: {}", unit, exc)
        return _no_ci(unit)
    return {"ci_lo": ci.lo, "ci_hi": ci.hi, "n_units": ci.n, "resample_unit": unit}


def _resample(counts: np.ndarray, unit_of: np.ndarray | None = None) -> np.ndarray:
    """Element indices for one cluster resample: each element repeated by its unit's count."""
    units = np.arange(counts.size) if unit_of is None else np.asarray(unit_of, dtype=int)
    return np.repeat(np.arange(units.size), counts[units])


def _resample_pairs(counts: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Pair indices for one cluster resample over the pairs' two endpoint MODELS."""
    return np.repeat(np.arange(pairs.shape[0]), counts[pairs[:, 0]] * counts[pairs[:, 1]])


def _cluster_ci(
    statistic: Callable[[np.ndarray], float],
    point: float,
    n_units: int,
    unit: str,
    seed: int,
) -> TaskMetrics:
    """Percentile interval of ``statistic`` over a cluster bootstrap of ``n_units``."""
    if n_units < 2 or not np.isfinite(point):
        return _no_ci(unit)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(CI_BOOT):
        counts = np.bincount(rng.integers(0, n_units, size=n_units), minlength=n_units)
        try:
            value = float(statistic(counts))
        except ValueError as exc:  # a resample that left a degenerate readout
            logger.debug("interval draw over {} was degenerate: {}", unit, exc)
            continue
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < MIN_CI_DRAWS:
        logger.debug(
            "only {} of {} interval draws over {} were finite; reporting no interval",
            len(draws),
            CI_BOOT,
            unit,
        )
        return _no_ci(unit)
    lo, hi = np.quantile(draws, [ALPHA / 2, 1.0 - ALPHA / 2])
    return {
        "ci_lo": float(min(lo, point)),
        "ci_hi": float(max(hi, point)),
        "n_units": int(n_units),
        "resample_unit": unit,
    }


def _clustered_question_ci(ctx: TaskContext, per_question: np.ndarray) -> TaskMetrics:
    """Interval for a mean of per-question scores, clustered on the questions' benchmarks."""
    values = np.asarray(per_question, dtype=np.float64)
    bench = np.asarray(ctx.bench, dtype=int)
    return _cluster_ci(
        lambda counts: _mean_or_nan(values[_resample(counts, bench)]),
        _mean_or_nan(values),
        ctx.n_benchmarks,
        UNIT_BENCHMARK_CLUSTERS,
        ctx.seed,
    )


def _headroom(value: float, chance: float, best: float) -> float:
    """``(value - chance) / (best - chance)``: 0 at ``chance``, 1 at ``best``."""
    span = best - chance
    if not np.isfinite(span) or abs(span) <= MIN_HEADROOM:
        return float("nan")
    return float((value - chance) / span)


def _upward_headroom(value: float, chance: float, best: float) -> float:
    """:func:`_headroom` for a ceiling that must sit ABOVE chance, else NaN."""
    if not np.isfinite(best - chance) or best - chance <= MIN_HEADROOM:
        return float("nan")
    return _headroom(value, chance, best)


def _ridge_fit_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> np.ndarray:
    """Fit a ridge on one block of rows and predict another. Returns [n_test, -1]."""
    mu = X_train.mean(axis=0)
    sd = np.maximum(X_train.std(axis=0), STD_FLOOR)
    model = RidgeCV(alphas=alphas).fit((X_train - mu) / sd, Y_train)
    return model.predict((X_test - mu) / sd).reshape(len(X_test), -1)


def _neighbour_k(n_rows: int) -> int:
    """``NEIGHBOUR_K``, lowered when a fold does not hold that many models."""
    return max(1, min(NEIGHBOUR_K, n_rows - 1))


def _retention_key(budget_gb: int) -> str:
    """Metric key for portfolio retention under a ``budget_gb`` budget."""
    return f"retention_at_{budget_gb}gb"


def _headroom_key(budget_gb: int) -> str:
    """Metric key for the headroom-normalised portfolio score under a ``budget_gb`` budget."""
    return f"headroom_at_{budget_gb}gb"
