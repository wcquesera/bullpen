"""Readouts supervised by Hugging Face model-card properties."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

from bullpen.data.folds import UNGROUPED_PREFIX
from bullpen.data.hf_card_labels import (
    HF_CLASSIFICATIONS,
    HF_REGRESSIONS,
    ClassificationSpec,
    HFCardLabels,
    RegressionSpec,
    load_hf_card_labels,
)
from bullpen.data.hf_cards import DEFAULT_CARD_DIR
from bullpen.evaluation.battery.constants import (
    MIN_LABELLED_FIT,
    MIN_LABELLED_SCORE,
    STD_FLOOR,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _absent,
    _labelled_rows,
    _too_few,
)
from bullpen.evaluation.battery.leaderboards import (
    CLOSED_SET_PRIMARY,
    EXTERNAL_PRIMARY,
    _closed_set_transfer,
    _labelled_scalar_transfer,
    _not_a_closed_set,
    external_coverage_floor,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.groups import STRUCTURAL

# --------------------------------------------------------------------------- #
# model-card readouts
# --------------------------------------------------------------------------- #
#: Where the card scrape lands.
HF_CARD_DIR: Path = DEFAULT_CARD_DIR

#: The headline of the eight card regressions.
CARD_REGRESSION_PRIMARY: str = EXTERNAL_PRIMARY

#: The headline of the two card classifications: :data:`CLOSED_SET_PRIMARY`,
#: which every label-table closed set in the battery shares.
CARD_CLASSIFICATION_PRIMARY: str = CLOSED_SET_PRIMARY

#: The categorical card readouts.
HF_CARD_CLASSIFICATIONS: tuple[ClassificationSpec, ...] = HF_CLASSIFICATIONS


_CARD_REGRESSION_DOC = """Predict {what} from the embedding, out of fold.

    A ridge readout scored by Spearman, beside a model-permuted null and a
    competence-only (mean accuracy) baseline.
    """

_CARD_CLASSIFICATION_DOC = """Classify {what} from the embedding, out of fold.

    A logistic head scored by macro-F1 against the permuted-label macro-F1; classes
    with fewer than {min_members} members are dropped.
    """


@lru_cache(maxsize=8)
def _cached_card_labels(model_ids: tuple[str, ...], card_dir: str) -> HFCardLabels | None:
    """The card columns for one model axis, read from disk at most once per axis."""
    return load_hf_card_labels(model_ids, Path(card_dir))


def _card_labels(ctx: TaskContext) -> HFCardLabels | None:
    """This context's card columns, or ``None`` when the cut cannot be gathered."""
    if not ctx.model_ids:
        return None
    return _cached_card_labels(tuple(str(m) for m in ctx.model_ids), str(HF_CARD_DIR))


def _card_absent(primary: str) -> TaskMetrics:
    """The shared note for a cut whose cards were never scraped."""
    return _absent(
        primary,
        "the cut carries no HuggingFace model cards on this model axis (data/raw/hf_model_cards/)",
    )


def _family_group_keys(ctx: TaskContext, rows: np.ndarray) -> np.ndarray | None:
    """[len(rows)] group key per scored model, or ``None`` when the cut has none."""
    if len(ctx.family) != ctx.n_models or not any(ctx.family):
        return None
    ids = list(ctx.model_ids) if len(ctx.model_ids) == ctx.n_models else list(range(ctx.n_models))
    keys = [str(ctx.family[i]) or f"{UNGROUPED_PREFIX}{ids[i]}" for i in rows]
    return np.asarray(keys, dtype=object)


def _card_groups(
    ctx: TaskContext, rows: np.ndarray, grouped_cv: bool, primary: str, what: str
) -> tuple[np.ndarray | None, TaskMetrics | None]:
    """``(group keys, refusal)`` for one card readout — exactly one is not ``None``."""
    if not grouped_cv:
        return None, None
    groups = _family_group_keys(ctx, rows)
    if groups is None:
        return None, _absent(
            primary,
            f"{what} is a constant of the base model, so it must be folded with a "
            f"family-grouped split, and this context carries no family column; a plain "
            f"KFold here would put a sibling carrying the answer in every training fold",
        )
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        return None, _absent(
            primary,
            f"the {rows.size} models carrying {what} fall in {n_groups} family group(s); "
            f"a grouped split needs at least two, and {what} is a base-model constant "
            f"that must not be folded randomly",
        )
    logger.debug("grouped CV for {}: {} models in {} family groups", what, int(rows.size), n_groups)
    return groups, None


def _card_regression(ctx: TaskContext, spec: RegressionSpec) -> TaskMetrics:
    """The shared body of the eight continuous card readouts."""
    labels = _card_labels(ctx)
    if labels is None:
        return _card_absent(CARD_REGRESSION_PRIMARY)
    y = np.asarray(labels.regressions[spec.task], dtype=np.float64)
    rows = _labelled_rows(np.isfinite(y))
    floor = external_coverage_floor()
    if rows.size < floor:
        return _absent(
            CARD_REGRESSION_PRIMARY,
            f"the cards state {spec.what} for {rows.size} of the {ctx.n_models} scored "
            f"models, below the {floor}-model coverage floor in config/evaluate.yaml; a "
            f"ridge fitted out of fold on that many rows reports its own shrinkage, not "
            f"the bank",
        )
    if np.ptp(y[rows]) < STD_FLOOR:
        return _absent(
            CARD_REGRESSION_PRIMARY,
            f"{spec.what} is the same on every carded model in this block, so there is "
            f"no ordering to predict",
        )
    groups, refusal = _card_groups(ctx, rows, spec.grouped_cv, CARD_REGRESSION_PRIMARY, spec.what)
    if refusal is not None:
        return refusal
    out = _labelled_scalar_transfer(ctx, y, rows, groups)
    out["n_uncarded"] = int(ctx.n_models - rows.size)
    return out


def _card_class_rows(
    labels: HFCardLabels, spec: ClassificationSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """``(column, rows, truth, classes)`` for one categorical card readout."""
    column = np.asarray(labels.classifications[spec.task], dtype=object)
    classes = sorted(labels.usable_classes(spec.task, spec))
    rows = _labelled_rows(np.isin(column, classes))
    return column, rows, column[rows], classes


def _card_classification(ctx: TaskContext, spec: ClassificationSpec) -> TaskMetrics:
    """The shared body of the two categorical card readouts."""
    labels = _card_labels(ctx)
    if labels is None:
        return _card_absent(CARD_CLASSIFICATION_PRIMARY)
    column, rows, _truth, classes = _card_class_rows(labels, spec)
    if len(classes) < spec.min_classes:
        return _not_a_closed_set(spec.task, len(classes), spec.min_members)
    floor = external_coverage_floor()
    if rows.size < floor or _too_few(rows):
        return _absent(
            CARD_CLASSIFICATION_PRIMARY,
            f"{rows.size} of the {ctx.n_models} scored models carry a {spec.task} class "
            f"with enough members, below the {max(floor, MIN_LABELLED_FIT + MIN_LABELLED_SCORE)}"
            f"-model floor a head needs to both fit and score",
        )
    groups, refusal = _card_groups(
        ctx, rows, spec.grouped_cv, CARD_CLASSIFICATION_PRIMARY, spec.what
    )
    if refusal is not None:
        return refusal
    out = _closed_set_transfer(ctx, column, rows, classes, groups)
    # every model the block did not score: no card, or a class below the floor
    out["n_uncarded"] = int(ctx.n_models - rows.size)
    return out


#: Appended to the docstring of every card readout whose column is a base-model constant, so a
#: reader of the task table sees WHY its number is lower than its ungrouped twin's rather than
#: filing it as a regression.
_GROUPED_CV_DOC = """

    Folded by model family, since the column is a constant of the base model.
    """


def _register_card_regression(spec: RegressionSpec) -> None:
    """Register one continuous card readout — a factory, for closure capture."""

    def readout(ctx: TaskContext) -> TaskMetrics:
        return _card_regression(ctx, spec)

    readout.__name__ = spec.task
    readout.__doc__ = _CARD_REGRESSION_DOC.format(what=spec.what, null=spec.null) + (
        _GROUPED_CV_DOC if spec.grouped_cv else ""
    )
    task(spec.task, primary=CARD_REGRESSION_PRIMARY, group=STRUCTURAL)(readout)


def _register_card_classification(spec: ClassificationSpec) -> None:
    """Register one categorical card readout — a factory, for closure capture."""

    def readout(ctx: TaskContext) -> TaskMetrics:
        return _card_classification(ctx, spec)

    readout.__name__ = spec.task
    readout.__doc__ = _CARD_CLASSIFICATION_DOC.format(
        what=spec.what, min_members=spec.min_members
    ) + (_GROUPED_CV_DOC if spec.grouped_cv else "")
    task(spec.task, primary=CARD_CLASSIFICATION_PRIMARY, group=STRUCTURAL)(readout)


for _regression_spec in HF_REGRESSIONS:
    _register_card_regression(_regression_spec)

for _classification_spec in HF_CARD_CLASSIFICATIONS:
    _register_card_classification(_classification_spec)
