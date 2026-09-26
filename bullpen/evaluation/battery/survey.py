"""Readouts supervised by the external-eval survey targets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from bullpen.data.external_survey_labels import (
    DEFAULT_ALIAS_DIR,
    DEFAULT_SURVEY_DIR,
    SURVEY_CLASSIFICATIONS,
    SURVEY_REGRESSIONS,
    SurveyClassificationSpec,
    SurveyLabels,
    SurveyRegressionSpec,
    load_survey_labels,
)
from bullpen.evaluation.battery.constants import MIN_LABELLED_FIT, MIN_LABELLED_SCORE, STD_FLOOR
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import _absent, _labelled_rows, _too_few
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
# external-eval survey targets
# --------------------------------------------------------------------------- #
#: Four readouts off ``data/raw/external_survey/``, joined by
#: :mod:`bullpen.data.external_survey_labels`.
SURVEY_REGRESSION_PRIMARY: str = EXTERNAL_PRIMARY

#: The headline of the precision readout.
SURVEY_CLASSIFICATION_PRIMARY: str = CLOSED_SET_PRIMARY

#: Where the survey files are read from.
SURVEY_DIR: Path = DEFAULT_SURVEY_DIR
SURVEY_ALIAS_DIR: Path = DEFAULT_ALIAS_DIR


_SURVEY_REGRESSION_DOC = """Predict {what} from the embedding, out of fold.

    A ridge readout scored by Spearman, beside a model-permuted null and a
    competence-only (mean accuracy) baseline.
    """

_SURVEY_CLASSIFICATION_DOC = """Classify {what} from the embedding, out of fold.

    A logistic head scored by macro-F1 against the permuted-label macro-F1; classes
    with fewer than {min_members} members are dropped.
    """


@lru_cache(maxsize=8)
def _cached_survey_labels(
    model_ids: tuple[str, ...], survey_dir: str, alias_dir: str
) -> SurveyLabels | None:
    """The survey columns for one model axis, read from disk at most once per axis."""
    return load_survey_labels(model_ids, Path(survey_dir), Path(alias_dir))


def _survey_labels(ctx: TaskContext) -> SurveyLabels | None:
    """This context's survey columns, gathered BY NAME off ``ctx.model_ids``."""
    if not ctx.model_ids:
        return None
    return _cached_survey_labels(
        tuple(str(m) for m in ctx.model_ids), str(SURVEY_DIR), str(SURVEY_ALIAS_DIR)
    )


def _survey_absent(primary: str, what: str) -> TaskMetrics:
    """The shared note for a cut whose survey sources were never fetched."""
    return _absent(
        primary,
        f"the cut carries no external survey column for {what}",
    )


def _survey_regression(ctx: TaskContext, spec: SurveyRegressionSpec) -> TaskMetrics:
    """The shared body of the three continuous survey readouts."""
    labels = _survey_labels(ctx)
    column = None if labels is None else labels.regressions.get(spec.task)
    if column is None:
        return _survey_absent(SURVEY_REGRESSION_PRIMARY, f"{spec.source}:{spec.metric}")
    y = np.asarray(column, dtype=np.float64)
    rows = _labelled_rows(np.isfinite(y))
    floor = external_coverage_floor()
    if rows.size < floor:
        return _absent(
            SURVEY_REGRESSION_PRIMARY,
            f"{spec.source}:{spec.metric} lists {rows.size} of the {ctx.n_models} scored "
            f"models, below the {floor}-model coverage floor in config/evaluate.yaml; a "
            f"ridge fitted out of fold on that many rows reports its own shrinkage, not "
            f"the bank",
        )
    if np.ptp(y[rows]) < STD_FLOOR:
        return _absent(
            SURVEY_REGRESSION_PRIMARY,
            f"{spec.source}:{spec.metric} is the same on every listed model in this block, "
            f"so there is no ordering to predict",
        )
    out = _labelled_scalar_transfer(ctx, y, rows)
    out["n_unlisted"] = int(ctx.n_models - rows.size)
    # the survey's own ρ against mean correctness, carried into the row so the competence-only
    # number is read against how redundant the TARGET is, not only against how well the bank
    # did.
    out["rho_competence_published"] = float(spec.rho_competence)
    return out


def _survey_classification(ctx: TaskContext, spec: SurveyClassificationSpec) -> TaskMetrics:
    """The shared body of the categorical survey readouts."""
    labels = _survey_labels(ctx)
    column_list = None if labels is None else labels.classifications.get(spec.task)
    if column_list is None:
        return _survey_absent(SURVEY_CLASSIFICATION_PRIMARY, f"{spec.source}:{spec.attribute}")
    column = np.asarray(column_list, dtype=object)
    classes = list(labels.usable_classes(spec.task, spec))
    rows = _labelled_rows(np.isin(column, classes))
    if len(classes) < spec.min_classes:
        return _not_a_closed_set(spec.task, len(classes), spec.min_members)
    floor = external_coverage_floor()
    if rows.size < floor or _too_few(rows):
        return _absent(
            SURVEY_CLASSIFICATION_PRIMARY,
            f"{rows.size} of the {ctx.n_models} scored models carry a {spec.task} class "
            f"with enough members, below the {max(floor, MIN_LABELLED_FIT + MIN_LABELLED_SCORE)}"
            f"-model floor a head needs to both fit and score",
        )
    out = _closed_set_transfer(ctx, column, rows, classes)
    # every model the block did not score: unlisted, off the closed set, or in a
    # class below the member floor
    out["n_unlisted"] = int(ctx.n_models - rows.size)
    return out


def _register_survey_regression(spec: SurveyRegressionSpec) -> None:
    """Register one continuous survey readout — a factory, for closure capture."""

    def readout(ctx: TaskContext) -> TaskMetrics:
        return _survey_regression(ctx, spec)

    readout.__name__ = spec.task
    readout.__doc__ = _SURVEY_REGRESSION_DOC.format(
        what=spec.what, null=spec.null, rho=spec.rho_competence
    )
    task(spec.task, primary=SURVEY_REGRESSION_PRIMARY, group=STRUCTURAL)(readout)


def _register_survey_classification(spec: SurveyClassificationSpec) -> None:
    """Register one categorical survey readout — a factory, for closure capture."""

    def readout(ctx: TaskContext) -> TaskMetrics:
        return _survey_classification(ctx, spec)

    readout.__name__ = spec.task
    readout.__doc__ = _SURVEY_CLASSIFICATION_DOC.format(
        what=spec.what, min_members=spec.min_members
    )
    task(spec.task, primary=SURVEY_CLASSIFICATION_PRIMARY, group=STRUCTURAL)(readout)


for _survey_regression_spec in SURVEY_REGRESSIONS:
    _register_survey_regression(_survey_regression_spec)

for _survey_classification_spec in SURVEY_CLASSIFICATIONS:
    _register_survey_classification(_survey_classification_spec)
