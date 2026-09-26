"""External board columns joined onto the model axis, from the cached board JSONs.

Reads the whole-board caches under ``data/raw/external_survey*/`` (LMArena, Open LLM v2,
Hub serving and community counts, Ollama, UGI, the Japanese and Portuguese boards) and
joins each registered metric onto ``model_ids`` with
:func:`~bullpen.data.external_labels.match_rows` (exact key, then the hand-verified
aliases, never fuzzy). A join key claimed by more than one published name is refused
(:func:`unambiguous`). Values are raw per model, NaN (or :data:`UNKNOWN`) where the
board is silent; coverage floors belong to the caller.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.config import RAW_DIR as _RAW_DIR
from bullpen.data.external_labels import ALIASES_NAME, match_key, match_rows, read_aliases

#: One JSON per board.
DEFAULT_SURVEY_DIR: Path = _RAW_DIR / "external_survey"

#: Sibling directories searched after :data:`DEFAULT_SURVEY_DIR`, in order; first hit wins.
SURVEY_SIBLING_DIRS: tuple[str, ...] = ("external_survey_v2", "external_survey_v3")

#: Where the hand-verified alias map lives.
DEFAULT_ALIAS_DIR: Path = _RAW_DIR / "external_labels_src"

#: What an absent-note tells the reader to provide.
SURVEY_BUILD: str = "the board JSONs under data/raw/external_survey*/"

#: Keys inside one survey source file.
METRICS_KEY = "metrics"
ATTRIBUTES_KEY = "model_attributes"

#: Label for a model the board does not classify; such rows are dropped.
UNKNOWN: str = ""


class ExternalSurveyError(RuntimeError):
    """A board file is malformed."""


def _identity(value: float) -> float:
    """The published scale."""
    return value


def _log10(value: float) -> float:
    """``log10(x)`` for a strictly positive column, NaN elsewhere (a zero is a gap)."""
    return math.log10(value) if math.isfinite(value) and value > 0 else float("nan")


def _log10_1p(value: float) -> float:
    """``log10(1 + x)`` for a non-negative count column (zero is a value), NaN if negative."""
    return math.log10(1.0 + value) if math.isfinite(value) and value >= 0 else float("nan")


@dataclass(frozen=True)
class SurveyRegressionSpec:
    """One published metric from one board file and the task it supervises.

    ``source`` is the file stem and ``metric`` the key inside its ``metrics`` block.
    """

    task: str
    source: str
    metric: str
    what: str
    #: rank correlation with mean correctness, as measured when the task was registered
    rho_competence: float
    #: the null this column must be quoted against, beyond the permuted one
    null: str = "competence — the d=1 leaderboard baseline"
    #: subtracted from ``metric`` model by model before ``transform`` (a within-model
    #: contrast); NaN when either side is missing
    minus_metric: str | None = None
    #: the scale the target is fitted on, applied after ``minus_metric``
    transform: Callable[[float], float] = _identity


@dataclass(frozen=True)
class SurveyClassificationSpec:
    """One categorical per-model attribute from a board file.

    Values outside the closed set ``classes`` become :data:`UNKNOWN` and are dropped.
    """

    task: str
    source: str
    attribute: str
    what: str
    classes: tuple[str, ...]
    min_members: int = 3
    min_classes: int = 2


#: The registered board regressions.
SURVEY_REGRESSIONS: tuple[SurveyRegressionSpec, ...] = ()

#: Deployment and community columns, and the Japanese and Portuguese boards.
SURVEY_V2_V3_REGRESSIONS: tuple[SurveyRegressionSpec, ...] = (
    SurveyRegressionSpec(
        "serving_providers_live",
        "hf_serving",
        "n_providers_live",
        "how many inference providers currently serve this checkpoint through the Hub's router",
        rho_competence=0.21,
        null="competence AND parameter count. A provider serves what its customers "
        "ask for, so this is a popularity column before it is anything else, and "
        "the reading is the residual after both — the bank knowing which "
        "checkpoints got DEPLOYED rather than which scored well",
    ),
    SurveyRegressionSpec(
        "ollama_quant_menu",
        "ollama",
        "ollama_quant_menu",
        "how many quantisations of this checkpoint ollama.com ships, 0 to 30",
        rho_competence=0.42,
        transform=_log10_1p,
        null="competence, at +0.42 the second-strongest coupling here, and it is "
        "structural: 90 of the 118 models are a hard zero because Ollama curates "
        "a small library, so the column is close to 'is this checkpoint famous' "
        "with a menu size attached. A positive Spearman that survives the "
        "competence baseline is the only reading worth quoting",
    ),
    SurveyRegressionSpec(
        "ugi_willingness",
        "ugi",
        "W/10 👍",
        "the UGI leaderboard's willingness score — how readily a model answers a "
        "sensitive prompt, 0 to 10",
        rho_competence=0.33,
        null="competence. Two caveats belong on this row and neither is a "
        "footnote. Coverage is the thinnest among these columns — 14 of the 118 on the "
        "board as fetched — so the interval is wide by construction and "
        "`n_labelled` must be quoted with the number. And the board's OLD "
        "revision, cached beside this one as `ugi_old`, is deliberately NOT "
        "unioned in: the two revisions re-scored the same models with a changed "
        "prompt set, so pooling them would put two different measurements in one "
        "column to buy four rows",
    ),
    SurveyRegressionSpec(
        "lb_pt_average",
        "lb_pt",
        "all_grouped_average",
        "the Open PT LLM leaderboard's Portuguese grouped average",
        rho_competence=0.64,
        null="competence, at +0.64 — the highest among these columns. Portuguese "
        "capability tracks general capability closely enough that the d=1 "
        "leaderboard baseline is a serious competitor here, so a cell that does "
        "not beat the competence-only Spearman printed beside it is not a result",
    ),
)

SURVEY_REGRESSIONS = (*SURVEY_REGRESSIONS, *SURVEY_V2_V3_REGRESSIONS)

#: bfloat16 against float16 only; the other precisions are too rare on this roster.
PRECISION_CLASSES: tuple[str, ...] = ("bfloat16", "float16")

SURVEY_CLASSIFICATIONS: tuple[SurveyClassificationSpec, ...] = ()

#: Every task this module supplies a target for.
SURVEY_TASKS: tuple[str, ...] = tuple(
    spec.task for spec in (*SURVEY_REGRESSIONS, *SURVEY_CLASSIFICATIONS)
)

#: Which survey source each task reads, so a missing file names the task it cost.
SURVEY_SOURCE_BY_TASK: dict[str, str] = {
    spec.task: spec.source for spec in (*SURVEY_REGRESSIONS, *SURVEY_CLASSIFICATIONS)
}


def survey_source_path(name: str, survey_dir: Path = DEFAULT_SURVEY_DIR) -> Path | None:
    """Where ``name.json`` lives, across the survey directory and its siblings, or ``None``."""
    survey_dir = Path(survey_dir)
    candidates = [survey_dir / f"{name}.json"]
    candidates += [survey_dir.parent / sibling / f"{name}.json" for sibling in SURVEY_SIBLING_DIRS]
    return next((path for path in candidates if path.exists()), None)


def read_source(name: str, survey_dir: Path = DEFAULT_SURVEY_DIR) -> dict[str, Any] | None:
    """One cached board file, or ``None`` (with a warning) when absent; malformed raises."""
    path = survey_source_path(name, survey_dir)
    if path is None:
        logger.warning(
            f"no survey source named {name}.json under {survey_dir} or its "
            f"{len(SURVEY_SIBLING_DIRS)} sibling(s)"
        )
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt cache
        raise ExternalSurveyError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ExternalSurveyError(f"{path} is not an object")
    return dict(raw)


def _published_values(source: Mapping[str, Any], metric: str, path: str) -> dict[str, Any]:
    """``{published name: value}`` for one metric, raising if the source lacks it."""
    metrics = source.get(METRICS_KEY)
    if not isinstance(metrics, Mapping):
        raise ExternalSurveyError(f"{path} has no {METRICS_KEY!r} object")
    column = metrics.get(metric)
    if column is None:
        raise ExternalSurveyError(
            f"{path} publishes no {metric!r}; it has {len(metrics)} metric(s)"
        )
    if not isinstance(column, Mapping):
        raise ExternalSurveyError(f"{path}:{metric} is not a name-keyed object")
    return dict(column)


def _published_attribute(source: Mapping[str, Any], attribute: str, path: str) -> dict[str, Any]:
    """``{published name: attribute value}``, raising if the source has no attribute block."""
    attributes = source.get(ATTRIBUTES_KEY)
    if not isinstance(attributes, Mapping):
        raise ExternalSurveyError(f"{path} has no {ATTRIBUTES_KEY!r} object")
    return {
        name: block.get(attribute)
        for name, block in attributes.items()
        if isinstance(block, Mapping)
    }


def _finite(value: Any) -> float:
    """``float(value)`` if it is a finite number, else NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def unambiguous(published: Mapping[str, Any], source: str = "") -> dict[str, Any]:
    """``published`` with every colliding join key dropped, and a warning naming them.

    :func:`~bullpen.data.external_labels.match_key` folds away the org prefix, so a
    whole-board pull can map several uploaders (``google/gemma-2-9b-it``,
    ``someone/gemma-2-9b-it``) to one key. Such keys are refused rather than
    tie-broken by listing order; a verified entry in ``model_aliases.json`` resolves one.
    """
    by_key: dict[str, list[str]] = {}
    for name in published:
        by_key.setdefault(match_key(name), []).append(str(name))
    colliding = {key: names for key, names in by_key.items() if len(names) > 1}
    if colliding:
        logger.warning(
            f"{source or 'survey source'}: {len(colliding)} join key(s) are claimed by "
            f"more than one published name and are refused rather than tie-broken; "
            f"e.g. {sorted(colliding)[0]} <- {sorted(colliding[sorted(colliding)[0]])}"
        )
    refused = {name for names in colliding.values() for name in names}
    return {name: value for name, value in published.items() if str(name) not in refused}


def survey_column(
    published: Mapping[str, Any],
    model_ids: Sequence[str],
    aliases: Mapping[str, str],
    source: str = "",
) -> tuple[np.ndarray, dict[str, int]]:
    """[M] published value joined onto ``model_ids``, NaN where unlisted, plus match counts."""
    published = unambiguous(published, source)
    match = match_rows(model_ids, published.keys(), aliases)
    values = np.array(
        [np.nan if src is None else _finite(published[src]) for src in match.source_of],
        dtype=np.float64,
    )
    return values, match.provenance()


def survey_class_column(
    published: Mapping[str, Any],
    model_ids: Sequence[str],
    aliases: Mapping[str, str],
    classes: Sequence[str],
    source: str = "",
) -> tuple[list[str], dict[str, int]]:
    """[M] class name joined onto ``model_ids``, :data:`UNKNOWN` off the closed set."""
    admitted = set(classes)
    published = unambiguous(published, source)
    match = match_rows(model_ids, published.keys(), aliases)
    column = [
        str(published[src]) if src is not None and str(published[src]) in admitted else UNKNOWN
        for src in match.source_of
    ]
    return column, match.provenance()


@dataclass(frozen=True)
class SurveyLabels:
    """The board columns on one model axis.

    ``regressions`` is ``{task: [M] float}`` (NaN where silent); ``classifications`` is
    ``{task: [M] str}`` (:data:`UNKNOWN` where silent).
    """

    model_ids: tuple[str, ...]
    regressions: dict[str, np.ndarray] = field(default_factory=dict)
    classifications: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict[str, dict[str, int]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return len(self.model_ids)

    def n_labelled(self, task: str) -> int:
        """Models carrying a usable label for ``task`` — NaN and unknown excluded."""
        if task in self.regressions:
            return int(np.isfinite(self.regressions[task]).sum())
        return sum(1 for label in self.classifications.get(task, []) if label != UNKNOWN)

    def class_counts(self, task: str) -> dict[str, int]:
        """``{class: members}`` on this block, :data:`UNKNOWN` excluded."""
        counts: dict[str, int] = {}
        for label in self.classifications.get(task, []):
            if label != UNKNOWN:
                counts[label] = counts.get(label, 0) + 1
        return counts

    def usable_classes(self, task: str, spec: SurveyClassificationSpec) -> tuple[str, ...]:
        """The classes of ``task`` with at least ``spec.min_members`` members here."""
        return tuple(
            name for name, n in sorted(self.class_counts(task).items()) if n >= spec.min_members
        )


def load_survey_labels(
    model_ids: Sequence[str],
    survey_dir: Path = DEFAULT_SURVEY_DIR,
    alias_dir: Path = DEFAULT_ALIAS_DIR,
) -> SurveyLabels | None:
    """Every board column gathered onto ``model_ids``, or ``None`` when no board file exists.

    Tasks whose board file is missing are left out of the returned dicts.
    """
    survey_dir = Path(survey_dir)
    aliases = read_aliases(Path(alias_dir) / ALIASES_NAME)
    sources = {
        name: read_source(name, survey_dir)
        for name in sorted({spec.source for spec in (*SURVEY_REGRESSIONS, *SURVEY_CLASSIFICATIONS)})
    }
    present = {name: src for name, src in sources.items() if src is not None}
    if not present:
        logger.warning(
            f"no external survey sources under {survey_dir}; the {len(SURVEY_TASKS)} survey "
            f"readouts have no target"
        )
        return None

    regressions: dict[str, np.ndarray] = {}
    classifications: dict[str, list[str]] = {}
    provenance: dict[str, dict[str, int]] = {}
    for spec in SURVEY_REGRESSIONS:
        source = present.get(spec.source)
        if source is None:
            continue
        path = str(
            survey_source_path(spec.source, survey_dir) or survey_dir / f"{spec.source}.json"
        )
        values, counts = survey_column(
            _published_values(source, spec.metric, path),
            model_ids,
            aliases,
            f"{spec.source}:{spec.metric}",
        )
        if spec.minus_metric is not None:
            reference, ref_counts = survey_column(
                _published_values(source, spec.minus_metric, path),
                model_ids,
                aliases,
                f"{spec.source}:{spec.minus_metric}",
            )
            # NaN on either side propagates
            values = values - reference
            counts = {
                **counts,
                "n_published_reference": ref_counts["n_published"],
                "n_matched_reference": ref_counts["n_matched"],
            }
        regressions[spec.task] = np.array(
            [spec.transform(value) for value in values], dtype=np.float64
        )
        provenance[spec.task] = counts
    for spec in SURVEY_CLASSIFICATIONS:
        source = present.get(spec.source)
        if source is None:
            continue
        path = str(
            survey_source_path(spec.source, survey_dir) or survey_dir / f"{spec.source}.json"
        )
        column, counts = survey_class_column(
            _published_attribute(source, spec.attribute, path),
            model_ids,
            aliases,
            spec.classes,
            f"{spec.source}:{spec.attribute}",
        )
        classifications[spec.task] = column
        provenance[spec.task] = counts

    labels = SurveyLabels(
        model_ids=tuple(str(m) for m in model_ids),
        regressions=regressions,
        classifications=classifications,
        provenance=provenance,
        meta={
            "survey_dir": str(survey_dir),
            "alias_dir": str(alias_dir),
            "n_models": len(model_ids),
            "sources_present": sorted(present),
            "sources_absent": sorted(name for name, src in sources.items() if src is None),
        },
    )
    logger.info(
        f"external survey labels: {labels.n_models} model(s) over "
        f"{len(present)}/{len(sources)} source(s); "
        + ", ".join(
            f"{task}={labels.n_labelled(task)}"
            for task in SURVEY_TASKS
            if task in regressions or task in classifications
        )
    )
    return labels
