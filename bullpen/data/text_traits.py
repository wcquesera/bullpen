"""Per-model text traits (``text_traits.npz``): how a model answers, not whether it was right.

Twelve scalars per model (log length, question-standardised token and character
length, hedge, refusal, truncation, list, boxed, code, first-person and newline rates),
precomputed from each model's answers on the eval question block. They are the labels of
the answer-statistics tasks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.data.external_labels import RAW_DIR

DEFAULT_TEXT_TRAITS = RAW_DIR / "text_traits.npz"

#: Keys of the stored npz.
TRAITS_KEY = "traits"
COVERED_KEY = "covered"
N_ANSWERS_KEY = "n_answers"
MODELS_KEY = "models"
TRAIT_NAMES_KEY = "trait_names"
META_KEY = "meta"


@dataclass(frozen=True)
class TextTraits:
    """``[M, 12]`` traits on a model axis; an uncovered model is an all-NaN row."""

    traits: np.ndarray
    covered: np.ndarray
    n_answers: np.ndarray
    model_ids: tuple[str, ...]
    trait_names: tuple[str, ...]
    meta: dict[str, Any]


def load_text_traits(
    model_ids: Sequence[str], path: Path = DEFAULT_TEXT_TRAITS
) -> TextTraits | None:
    """The trait table gathered by model id onto ``model_ids``, or ``None`` when absent."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"no text-trait table at {path}; the answer-statistics tasks have no target")
        return None
    with np.load(path, allow_pickle=False) as z:
        stored = [str(m) for m in z[MODELS_KEY]]
        names = tuple(str(t) for t in z[TRAIT_NAMES_KEY])
        table = np.asarray(z[TRAITS_KEY], dtype=np.float64)
        covered = np.asarray(z[COVERED_KEY], dtype=bool)
        n_answers = np.asarray(z[N_ANSWERS_KEY], dtype=np.int64)
        meta = json.loads(str(z[META_KEY]))
    row_of = {name: i for i, name in enumerate(stored)}
    rows = [row_of.get(str(m)) for m in model_ids]
    traits = np.full((len(rows), len(names)), np.nan)
    is_covered = np.zeros(len(rows), dtype=bool)
    answered = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        if row is None:
            continue
        traits[i] = table[row]
        is_covered[i] = covered[row]
        answered[i] = n_answers[row]
    logger.info(
        f"text traits: {int(is_covered.sum())}/{len(rows)} of the scored models carry "
        f"a {len(names)}-channel profile"
    )
    return TextTraits(
        traits=traits,
        covered=is_covered,
        n_answers=answered,
        model_ids=tuple(str(m) for m in model_ids),
        trait_names=names,
        meta=meta,
    )
