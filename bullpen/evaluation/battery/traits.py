"""Behaviour: text-trait readouts and model authorship."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bullpen.evaluation.battery.constants import (
    MIN_LABELLED_FIT,
    MIN_LABELLED_SCORE,
    STD_FLOOR,
    UNIT_MODELS,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _absent,
    _folds_over,
    _labelled_rows,
    _scalar_transfer,
    _too_few,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.shared import _mean_ci
from bullpen.evaluation.groups import STRUCTURAL
from bullpen.evaluation.metrics import oof_ridge_predict

# --------------------------------------------------------------------------- #
# behaviour: text-trait readouts
# --------------------------------------------------------------------------- #
#: Metric the three single-channel trait readouts headline.
TRAIT_PRIMARY: str = "spearman"


@dataclass(frozen=True)
class TraitSpec:
    """One text-trait channel and the battery task supervised by it."""

    task: str
    #: column name in ``ctx.trait_names``, from text_traits.npz
    channel: str
    #: noun phrase naming the channel, read into the generated docstring
    what: str


#: The three single-channel trait readouts.
TRAIT_SPECS: tuple[TraitSpec, ...] = (
    TraitSpec("length_profile", "log_n_tok", "how long this model's answers run"),
    TraitSpec("hedge_style", "hedge_rate", "how often this model hedges"),
    TraitSpec("refusal_rate", "refusal_rate", "how often this model declines to answer"),
)

#: ``task -> the trait channel it reads``, plus the two readouts that need the whole profile
#: rather than one column.
TRAIT_CHANNEL_BY_TASK: dict[str, str] = {spec.task: spec.channel for spec in TRAIT_SPECS}
TRAIT_TASKS: tuple[str, ...] = (
    *TRAIT_CHANNEL_BY_TASK,
    "model_authorship",
)

#: The generated docstring for the three. One text, because one caveat covers all
#: three and a per-task paraphrase is three chances to weaken it.
_TRAIT_DOC = """Predict {what} from the embedding, out of fold (ridge, Spearman)."""


def _trait_column(ctx: TaskContext, channel: str) -> np.ndarray | None:
    """One trait channel as [M], or ``None`` when this cut does not carry it."""
    if ctx.traits is None or channel not in ctx.trait_names:
        return None
    return np.asarray(ctx.traits, dtype=np.float64)[:, ctx.trait_names.index(channel)]


def _trait_absent(primary: str, what: str) -> TaskMetrics:
    """The shared note for a readout whose trait scan this cut has not been given."""
    return _absent(
        primary,
        f"the cut carries no {what}",
    )


def _trait_transfer(ctx: TaskContext, spec: TraitSpec) -> TaskMetrics:
    """The shared body of the three single-channel trait readouts."""
    column = _trait_column(ctx, spec.channel)
    if column is None:
        return _trait_absent(TRAIT_PRIMARY, f"{spec.channel} trait channel")
    rows = _labelled_rows(np.isfinite(column))
    if _too_few(rows):
        return _absent(
            TRAIT_PRIMARY,
            f"{spec.channel} is measured on {rows.size} of the {ctx.n_models} scored "
            f"models, below the {MIN_LABELLED_FIT + MIN_LABELLED_SCORE} an out-of-fold "
            f"ridge needs to fit and rank",
        )
    if np.ptp(column[rows]) < STD_FLOOR:
        return _absent(TRAIT_PRIMARY, f"{spec.channel} is constant over every covered model")
    out = _scalar_transfer(ctx, column, rows, primary=TRAIT_PRIMARY)
    out["n_unprofiled"] = int(ctx.n_models - rows.size)
    return out


def _register_trait(spec: TraitSpec) -> None:
    """Register one trait readout from its spec — a factory, for closure capture."""

    def readout(ctx: TaskContext) -> TaskMetrics:
        return _trait_transfer(ctx, spec)

    readout.__name__ = spec.task
    readout.__doc__ = _TRAIT_DOC.format(what=spec.what, channel=spec.channel)
    task(spec.task, primary=TRAIT_PRIMARY, group=STRUCTURAL)(readout)


for _trait_spec in TRAIT_SPECS:
    _register_trait(_trait_spec)


@task("model_authorship", primary="top1", group=STRUCTURAL)
def model_authorship(ctx: TaskContext) -> TaskMetrics:
    """Can a held-out model be matched to its own behavioural fingerprint?"""
    if ctx.traits is None:
        return _trait_absent("top1", "text-trait scan")
    profiles = np.asarray(ctx.traits, dtype=np.float64)
    rows = _labelled_rows(np.isfinite(profiles).all(axis=1))
    if _too_few(rows):
        return _absent(
            "top1",
            f"{rows.size} of the {ctx.n_models} scored models carry a full "
            f"{len(ctx.trait_names)}-channel profile, below the "
            f"{MIN_LABELLED_FIT + MIN_LABELLED_SCORE} a retrieval readout needs",
        )
    truth = profiles[rows]
    spread = np.maximum(truth.std(axis=0), STD_FLOOR)
    truth = (truth - truth.mean(axis=0)) / spread
    n_folds = _folds_over(rows, ctx)
    predicted = oof_ridge_predict(ctx.Z[rows], truth, n_folds=n_folds, seed=ctx.seed)
    distance = ((predicted[:, None, :] - truth[None, :, :]) ** 2).sum(axis=2)
    hit = distance.argmin(axis=1) == np.arange(rows.size)
    return {
        "top1": float(hit.mean()),
        "top1_chance": 1.0 / rows.size,
        "n_candidates": int(rows.size),
        "n_channels": len(ctx.trait_names),
        "n_folds": n_folds,
        # the candidate set is NOT resampled with the queries.
        **_mean_ci(hit.astype(np.float64), UNIT_MODELS, ctx.seed),
    }
