"""Reward-model scores and answer style per (model, question) cell (``group_h_labels.npz``).

The labels of the preference and style tasks (``rm_mean``, ``rm_style``). Each ``[M, Q]``
block holds a raw measurement on the eval questions: the reward score of
``Skywork-Reward-V2-Qwen3-4B`` for the (question, answer) pair, the correctness bit, the
answer length, and hedge and refusal flags. The per-question correctness residual is
fitted inside the battery on the training models of the split (:func:`correctness_residual`).
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

DEFAULT_GROUP_H_LABELS = RAW_DIR / "group_h_labels.npz"


#: Reward-scored cells a model needs before it carries a label.
MIN_CELLS_PER_MODEL: int = 50


#: Training models a per-question correctness fit needs before it is taken.
MIN_RESIDUAL_ROWS: int = 4


#: Ridge on the 2x2 normal equations of the per-question correctness fit, as a
#: share of its trace.
RESIDUAL_RIDGE: float = 1e-6

#: Floor on a spread before it is divided out.
STD_FLOOR: float = 1e-9

#: Keys of the stored npz.
MODELS_KEY = "models"
QUESTIONS_KEY = "question_ids"
COLS_KEY = "cols"
AXES_KEY = "axes"
RM_KEY = "rm"
CORRECT_KEY = "correct"
N_TOK_MATRIX_KEY = "n_tok"
HEDGE_KEY = "hedge"
REFUSAL_KEY = "refusal"
META_KEY = "meta"


@dataclass(frozen=True)
class GroupHLabels:
    """The raw per-cell measurements: ``[M, Q]`` matrices, NaN where a cell was not measured."""

    model_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    cols: np.ndarray
    axes: tuple[str, ...]
    rm: np.ndarray
    correct: np.ndarray
    n_tok: np.ndarray
    hedge: np.ndarray
    refusal: np.ndarray
    meta: dict[str, Any]

    @property
    def n_models(self) -> int:
        return int(self.rm.shape[0])

    @property
    def n_questions(self) -> int:
        return int(self.rm.shape[1])

    @property
    def n_scored_cells(self) -> int:
        return int(np.isfinite(self.rm).sum())

    def covered(self, min_cells: int = MIN_CELLS_PER_MODEL) -> np.ndarray:
        """[M] bool — models with enough scored cells to carry a label."""
        return np.isfinite(self.rm).sum(axis=1) >= int(min_cells)


def load_group_h_labels(
    model_ids: Sequence[str], path: Path = DEFAULT_GROUP_H_LABELS
) -> GroupHLabels | None:
    """The label block gathered by model id onto ``model_ids``, or ``None`` when absent."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"no reward-model labels at {path}; the reward-model tasks have no target")
        return None
    with np.load(path, allow_pickle=False) as z:
        stored = [str(m) for m in z[MODELS_KEY]]
        question_ids = tuple(str(q) for q in z[QUESTIONS_KEY])
        cols = np.asarray(z[COLS_KEY], dtype=int)
        axes = tuple(str(a) for a in z[AXES_KEY])
        blocks = {
            key: np.asarray(z[key], dtype=np.float64)
            for key in (RM_KEY, CORRECT_KEY, N_TOK_MATRIX_KEY, HEDGE_KEY, REFUSAL_KEY)
        }
        meta = json.loads(str(z[META_KEY]))
    row_of = {name: i for i, name in enumerate(stored)}
    rows = [row_of.get(str(m)) for m in model_ids]
    gathered = {
        key: np.array(
            [block[row] if row is not None else np.full(len(question_ids), np.nan) for row in rows]
        )
        for key, block in blocks.items()
    }
    labels = GroupHLabels(
        model_ids=tuple(str(m) for m in model_ids),
        question_ids=question_ids,
        cols=cols,
        axes=axes,
        rm=gathered[RM_KEY],
        correct=gathered[CORRECT_KEY],
        n_tok=gathered[N_TOK_MATRIX_KEY],
        hedge=gathered[HEDGE_KEY],
        refusal=gathered[REFUSAL_KEY],
        meta=meta,
    )
    logger.info(
        f"reward-model labels: {labels.n_models} x {labels.n_questions}, "
        f"{labels.n_scored_cells} scored cell(s), {int(labels.covered().sum())} "
        f"model(s) above the {MIN_CELLS_PER_MODEL}-cell floor"
    )
    return labels


def correctness_residual(
    rm: np.ndarray, correct: np.ndarray, train: np.ndarray, ridge: float = RESIDUAL_RIDGE
) -> np.ndarray:
    """[M, Q] reward score with the per-question correctness bit removed.

    Per question, ``rm ~ a_q + b_q * correct`` is fitted on the training models and
    the residual taken for every model; a question with no correctness contrast
    among the training models only has its intercept removed.
    """
    rm = np.asarray(rm, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    tr = np.asarray(train, dtype=int)
    out = np.full_like(rm, np.nan)
    for j in range(rm.shape[1]):
        y, c = rm[tr, j], correct[tr, j]
        keep = np.isfinite(y) & np.isfinite(c)
        if keep.sum() < MIN_RESIDUAL_ROWS:
            continue
        yk, ck = y[keep], c[keep]
        if np.ptp(ck) < STD_FLOOR:
            out[:, j] = rm[:, j] - yk.mean()
            continue
        X = np.column_stack([np.ones(ck.size), ck])
        gram = X.T @ X
        beta = np.linalg.solve(gram + ridge * np.trace(gram) / 2.0 * np.eye(2), X.T @ yk)
        cj = np.where(np.isfinite(correct[:, j]), correct[:, j], ck.mean())
        out[:, j] = rm[:, j] - (beta[0] + beta[1] * cj)
    return out
