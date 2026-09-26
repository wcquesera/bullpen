"""The train/test split over models.

A strict run takes its split from a fold plan (:mod:`bullpen.data.folds`); the ``fast``
protocol uses ``fullfit``, where every model is in both arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from bullpen.data.slice import Slice

TRAIN_LABEL = "train"
TEST_LABEL = "test"

STRATEGIES: tuple[str, ...] = ("fullfit",)


@dataclass(frozen=True)
class Split:
    """Row indices of the two arms, and which rule produced them."""

    train: np.ndarray
    test: np.ndarray
    source: str

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)

    def describe(self) -> str:
        return f"{self.n_train}/{self.n_test} train/test ({self.source})"


def _finalise(train: np.ndarray, test: np.ndarray, source: str) -> Split:
    """Sort, validate and log a candidate split."""
    train = np.asarray(np.sort(train), dtype=int)
    test = np.asarray(np.sort(test), dtype=int)
    if test.size == 0:
        raise ValueError(f"{source} split assigns no model to '{TEST_LABEL}'")
    if train.size == 0:
        raise ValueError(f"{source} split assigns no model to '{TRAIN_LABEL}'")
    overlap = np.intersect1d(train, test)
    if overlap.size:
        raise ValueError(f"{source} split puts {overlap.size} model(s) in both arms")
    split = Split(train=train, test=test, source=source)
    logger.info(f"split: {split.describe()}")
    return split


def fullfit_split(sl: Slice) -> Split:
    """Every model in both arms (the ``fast`` protocol): encoders see every model, and
    the scoring heads are out of fold over models."""
    if sl.n_models < 1:
        raise ValueError("cannot fullfit a slice with no models")
    rows = np.arange(sl.n_models, dtype=int)
    split = Split(train=rows, test=rows.copy(), source="fullfit")
    logger.warning(
        f"fullfit: train and test are the same {sl.n_models} models ON PURPOSE — "
        "every test score is in-sample; monitoring only, never a held-out result"
    )
    logger.info(f"split: {split.describe()}")
    return split


def resolve_split(sl: Slice, strategy: str = "fullfit") -> Split:
    """The split for this slice under ``strategy`` (only ``fullfit``)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}; expected one of {STRATEGIES}")
    return fullfit_split(sl)
