"""Reward-model readouts: mean reward and reward style on the probe cells."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bullpen.data.group_h_labels import (
    correctness_residual,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _absent,
    _labelled_rows,
    _scalar_transfer,
    _too_few,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.traits import TRAIT_PRIMARY
from bullpen.evaluation.groups import STRUCTURAL

# --------------------------------------------------------------------------- #
# reward-model readouts
# --------------------------------------------------------------------------- #
#: Every reward-model readout, for ``eval.py``'s coverage decision.
GROUP_H_TASKS: tuple[str, ...] = ("rm_mean", "rm_style")


#: The note every reward-model readout returns on a cut whose labels were never built.
_GROUP_H_ABSENT = "the cut carries no reward labels (group_h_labels.npz)"


def _group_h_rows(ctx: TaskContext) -> np.ndarray:
    """Models with enough reward-scored cells to carry a reward label."""
    if ctx.group_h is None:
        return np.empty(0, dtype=int)
    return _labelled_rows(ctx.group_h.covered())


def _oof_by_model(ctx: TaskContext, build: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """``build``'s output, every row taken from a fit that did not see that row."""
    out: np.ndarray | None = None
    for train, test in ctx.model_folds():
        block = np.asarray(build(train), dtype=np.float64)
        if out is None:
            out = np.full(block.shape, np.nan)
        out[test] = block[test]
    if out is None:  # pragma: no cover — model_folds always yields at least two folds
        raise RuntimeError("model_folds produced no fold")
    return out


def _oof_residual(ctx: TaskContext) -> np.ndarray:
    """[M, Q] reward score with the per-question correctness bit removed, out of fold."""
    labels = ctx.group_h
    assert labels is not None
    return _oof_by_model(ctx, lambda train: correctness_residual(labels.rm, labels.correct, train))


@task("rm_mean", primary=TRAIT_PRIMARY, group=STRUCTURAL)
def rm_mean(ctx: TaskContext) -> TaskMetrics:
    """Predict the model's mean reward-model score — the competence-loaded half."""
    if ctx.group_h is None:
        return _absent(TRAIT_PRIMARY, _GROUP_H_ABSENT)
    rows = _group_h_rows(ctx)
    if _too_few(rows):
        return _absent(
            TRAIT_PRIMARY,
            f"rm_mean: {rows.size} of the {ctx.n_models} scored models carry reward scores",
        )
    with np.errstate(invalid="ignore"):
        y = np.nanmean(ctx.group_h.rm, axis=1)
    out = _scalar_transfer(ctx, y, rows, primary=TRAIT_PRIMARY)
    out["n_unscored"] = int(ctx.n_models - rows.size)
    return out


@task("rm_style", primary=TRAIT_PRIMARY, group=STRUCTURAL)
def rm_style(ctx: TaskContext) -> TaskMetrics:
    """Predict the reward premium that survives removing the correctness bit."""
    if ctx.group_h is None:
        return _absent(TRAIT_PRIMARY, _GROUP_H_ABSENT)
    rows = _group_h_rows(ctx)
    if _too_few(rows):
        return _absent(
            TRAIT_PRIMARY,
            f"rm_style: {rows.size} of the {ctx.n_models} scored models carry reward scores",
        )
    with np.errstate(invalid="ignore"):
        y = np.nanmean(_oof_residual(ctx), axis=1)
    rows = rows[np.isfinite(y[rows])]
    if _too_few(rows):
        return _absent(
            TRAIT_PRIMARY, f"rm_style: the residual is defined for only {rows.size} models"
        )
    return _scalar_transfer(ctx, y, rows, primary=TRAIT_PRIMARY)
