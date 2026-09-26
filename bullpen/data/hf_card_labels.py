"""Hugging Face model-card properties as task labels.

Reads the cached cards (:mod:`bullpen.data.hf_cards`) and defines one label column per
card property: continuous ones (downloads, context length, upload date) and categorical
ones (architecture family, licence class). Values are raw per model, gathered by model
id, NaN (or :data:`UNKNOWN`) where the card is silent; class floors and coverage gates
belong to the caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.data.hf_cards import DEFAULT_CARD_DIR, read_cards

#: Label for a model whose card does not answer a categorical question; such rows are dropped.
UNKNOWN: str = ""

#: Licences with no use restriction beyond attribution; everything else (community,
#: non-commercial, RAIL/OpenRAIL, bespoke) is restrictive.
PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "artistic-2.0",
        "mpl-2.0",
        "unlicense",
        "wtfpl",
        "isc",
        "zlib",
        "postgresql",
        "lgpl-3.0",
        "gpl-3.0",
        "agpl-3.0",
    }
)


@dataclass(frozen=True)
class RegressionSpec:
    """One continuous card property: ``of_card`` reads it, ``transform`` sets the fitted scale."""

    task: str
    what: str
    of_card: Callable[[Mapping[str, Any]], Any]
    transform: Callable[[float], float]
    #: the null this column must be quoted against, beyond the permuted one
    null: str = "competence and parameter count"
    #: True for a base-model constant, cross-validated with a family-grouped split
    grouped_cv: bool = False


@dataclass(frozen=True)
class ClassificationSpec:
    """One categorical card property; ``of_card`` returns a class name or :data:`UNKNOWN`.

    ``min_members`` is the floor a class must clear; ``min_classes`` the number of
    surviving classes a task needs.
    """

    task: str
    what: str
    of_card: Callable[[Mapping[str, Any]], str]
    min_members: int = 3
    min_classes: int = 2
    #: True for a base-model constant, cross-validated with a family-grouped split
    grouped_cv: bool = False


def _finite(value: Any) -> float:
    """``float(value)`` if it is a finite number, else NaN. Booleans are numbers here."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _log10_1p(value: float) -> float:
    """``log10(1 + x)`` for a non-negative count; zero is a value, not a gap."""
    return math.log10(1.0 + value) if math.isfinite(value) and value >= 0 else float("nan")


def _log2(value: float) -> float:
    """``log2(x)`` for a positive value, NaN elsewhere."""
    return math.log2(value) if math.isfinite(value) and value > 0 else float("nan")


def _identity(value: float) -> float:
    return value


def _popularity(field: str) -> Callable[[Mapping[str, Any]], Any]:
    return lambda card: (card.get("popularity") or {}).get(field)


def _architecture(field: str) -> Callable[[Mapping[str, Any]], Any]:
    return lambda card: (card.get("architecture") or {}).get(field)


#: Day zero for :func:`created_days`.
EPOCH: datetime = datetime(2022, 1, 1, tzinfo=UTC)


_NON_LANGUAGE_2LETTER: frozenset[str] = frozenset(
    {
        "ak",
        "bm",
        "ig",
        "ki",
        "lg",
        "ln",
        "ny",
        "rn",
        "rw",
        "sn",
        "st",
        "tn",
        "ts",
        "tw",
        "wo",
        "xh",
        "yo",
        "zu",
    }
)


def created_days(card: Mapping[str, Any]) -> float:
    """Days from :data:`EPOCH` to the repo's ``createdAt`` (the upload date), NaN if absent."""
    stamp = (card.get("timestamps") or {}).get("createdAt")
    if not isinstance(stamp, str) or not stamp:
        return float("nan")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return float("nan")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (parsed - EPOCH).total_seconds() / 86400.0


#: The continuous readouts. Base-model constants (architecture widths, vocabulary,
#: context window) are ``grouped_cv``: a random fold would let a sibling checkpoint
#: carry the answer.
HF_REGRESSIONS: tuple[RegressionSpec, ...] = (
    RegressionSpec(
        "hf_downloads",
        "the log10 monthly download count on the Hub",
        _popularity("downloads"),
        _log10_1p,
        null="competence and parameter count — this column is mostly lab and size",
    ),
    RegressionSpec(
        "hf_context_length",
        "the log2 maximum position embedding count",
        _architecture("max_position_embeddings"),
        _log2,
        grouped_cv=True,
    ),
    RegressionSpec(
        "hf_release_date",
        "days from 2022-01-01 to the repo's createdAt — when the checkpoint was uploaded",
        created_days,
        _identity,
        null="competence and parameter count — models got better over the period, "
        "so this column is correlated with competence and the residual is the "
        "reading. It is NOT a base-model constant: gemma-2-9b-it and gemma-3-1b-it "
        "are nine months apart",
    ),
)


def architecture_family(card: Mapping[str, Any]) -> str:
    """The transformers ``model_type`` (``llama``, ``mistral``, ``qwen2``, ...), else unknown."""
    value = (card.get("architecture") or {}).get("model_type")
    return str(value).lower() if value else UNKNOWN


def license_type(card: Mapping[str, Any]) -> str:
    """``permissive`` | ``restrictive`` by :data:`PERMISSIVE_LICENSES`; no licence is unknown."""
    value = (card.get("metadata") or {}).get("license")
    if not value:
        return UNKNOWN
    return "permissive" if str(value).strip().lower() in PERMISSIVE_LICENSES else "restrictive"


#: The five categorical readouts.
HF_CLASSIFICATIONS: tuple[ClassificationSpec, ...] = (
    ClassificationSpec(
        "hf_architecture_family",
        "which transformers architecture the weights load under",
        architecture_family,
        grouped_cv=True,
    ),
    ClassificationSpec(
        "hf_license_type",
        "whether the licence restricts use (permissive vs restrictive)",
        license_type,
    ),
)

#: Every task name this module defines.
HF_CARD_TASKS: tuple[str, ...] = tuple(spec.task for spec in (*HF_REGRESSIONS, *HF_CLASSIFICATIONS))

#: The card columns that fold with a family-grouped split.
HF_GROUPED_CV_TASKS: frozenset[str] = frozenset(
    spec.task for spec in (*HF_REGRESSIONS, *HF_CLASSIFICATIONS) if spec.grouped_cv
)


@dataclass(frozen=True)
class HFCardLabels:
    """The card properties on one model axis.

    ``regressions`` is ``{task: [M] float}`` (NaN where silent); ``classifications`` is
    ``{task: [M] str}`` (:data:`UNKNOWN` where silent).
    """

    model_ids: tuple[str, ...]
    regressions: dict[str, np.ndarray]
    classifications: dict[str, list[str]]
    meta: dict[str, Any]

    @property
    def n_models(self) -> int:
        return len(self.model_ids)

    def n_labelled(self, task: str) -> int:
        """Models carrying a usable label for ``task`` — NaN and unknown excluded."""
        if task in self.regressions:
            return int(np.isfinite(self.regressions[task]).sum())
        return sum(1 for label in self.classifications[task] if label != UNKNOWN)

    def class_counts(self, task: str) -> dict[str, int]:
        """``{class: members}`` for a categorical task, unknowns excluded."""
        counts: dict[str, int] = {}
        for label in self.classifications[task]:
            if label != UNKNOWN:
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def usable_classes(self, task: str, spec: ClassificationSpec) -> tuple[str, ...]:
        """The classes of ``task`` with at least ``spec.min_members`` members."""
        return tuple(name for name, n in self.class_counts(task).items() if n >= spec.min_members)


def regression_column(
    cards: Mapping[str, Mapping[str, Any]], model_ids: Sequence[str], spec: RegressionSpec
) -> np.ndarray:
    """[M] transformed target for one continuous spec, NaN where the card is silent."""
    return np.array(
        [spec.transform(_finite(spec.of_card(cards.get(str(m), {})))) for m in model_ids],
        dtype=np.float64,
    )


def classification_column(
    cards: Mapping[str, Mapping[str, Any]], model_ids: Sequence[str], spec: ClassificationSpec
) -> list[str]:
    """[M] class name for one categorical spec; a model with no card is :data:`UNKNOWN`."""
    out: list[str] = []
    for model_id in model_ids:
        card = cards.get(str(model_id))
        out.append(UNKNOWN if not card or not card.get("ok") else spec.of_card(card))
    return out


def load_hf_card_labels(
    model_ids: Sequence[str],
    card_dir: Path = DEFAULT_CARD_DIR,
) -> HFCardLabels | None:
    """Every card column gathered onto ``model_ids``, or ``None`` when there are no cards."""
    card_dir = Path(card_dir)
    cards = read_cards(model_ids, card_dir)
    if not cards:
        logger.warning(f"no HuggingFace cards under {card_dir}; the card readouts have no target")
        return None
    ok = {str(k): dict(c) for k, c in cards.items() if c.get("ok")}
    labels = HFCardLabels(
        model_ids=tuple(str(m) for m in model_ids),
        regressions={spec.task: regression_column(ok, model_ids, spec) for spec in HF_REGRESSIONS},
        classifications={
            spec.task: classification_column(ok, model_ids, spec) for spec in HF_CLASSIFICATIONS
        },
        meta={
            "card_dir": str(card_dir),
            "n_models": len(model_ids),
            "n_cards": len(cards),
            "n_ok": len(ok),
            "unscraped_models": [str(m) for m in model_ids if str(m) not in cards],
            "unfetched_models": sorted(set(cards) - set(ok)),
        },
    )
    logger.info(
        f"HF card labels: {labels.n_models} model(s), {len(ok)} card(s) fetched; "
        + ", ".join(f"{task}={labels.n_labelled(task)}" for task in HF_CARD_TASKS)
    )
    return labels
