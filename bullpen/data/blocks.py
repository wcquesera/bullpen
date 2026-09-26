"""The three-block cut of the question axis: eval, tune, pool.

``eval`` is for scoring only (never fitted on), ``tune`` is held out of fitting for
hyperparameter selection, and ``pool`` is what every encoder is fitted on and what the
K-question interview is ranked within. The draw is
stratified by benchmark in proportion to each benchmark's size, and is a function of
``(benchmark labels, sizes, seed)`` alone; it is cached in ``question_blocks.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from bullpen.data.slice import DATA_DIR, Slice

#: Cached cut, written once and read thereafter.
DEFAULT_BLOCKS = DATA_DIR / "question_blocks.json"

#: Columns reserved for scoring.
EVAL_SIZE = 4096

#: Columns reserved for hyperparameter selection.
TUNE_SIZE = 3072

#: Seed of the draw.
BLOCK_SEED = 20260906

#: The three block names, in the order they are drawn.
BLOCK_NAMES: tuple[str, ...] = ("eval", "tune", "pool")


@dataclass(frozen=True)
class QuestionBlocks:
    """One three-block cut of the question axis, as column indices."""

    eval_idx: np.ndarray
    tune_idx: np.ndarray
    pool_idx: np.ndarray
    n_questions: int
    seed: int

    @property
    def n_eval(self) -> int:
        return int(self.eval_idx.size)

    @property
    def n_tune(self) -> int:
        return int(self.tune_idx.size)

    @property
    def n_pool(self) -> int:
        return int(self.pool_idx.size)

    def describe(self) -> str:
        return (
            f"eval {self.n_eval} / tune {self.n_tune} / pool {self.n_pool} "
            f"of {self.n_questions} questions (seed {self.seed})"
        )

    def as_dict(self) -> dict[str, object]:
        """The on-disk form."""
        return {
            "eval": self.eval_idx.tolist(),
            "tune": self.tune_idx.tolist(),
            "pool": self.pool_idx.tolist(),
            "n_questions": self.n_questions,
            "seed": self.seed,
            "eval_size": self.n_eval,
            "tune_size": self.n_tune,
        }


def axis_labels(sl: Slice) -> np.ndarray:
    """[Q] benchmark name per column (names, since ``bench`` integers are re-compacted)."""
    names = np.asarray(sl.bench_names)
    return names[sl.bench]


def stratified_sample(
    axes: np.ndarray, size: int, rng: np.random.Generator, available: np.ndarray
) -> np.ndarray:
    """``size`` of ``available``, spread over benchmarks in proportion to their width.

    Each benchmark contributes ``round(size * share)`` columns (at least one); an
    overshoot from rounding is trimmed by a second random draw.
    """
    available = np.asarray(available, dtype=int)
    if size > available.size:
        raise ValueError(f"cannot draw {size} columns from {available.size}")
    labels = np.asarray(axes)[available]
    picks: list[int] = []
    for ax in sorted(set(labels.tolist())):
        cols = available[labels == ax]
        n = round(size * cols.size / available.size)
        n = min(max(n, 1), cols.size)
        picks.extend(rng.choice(cols, size=n, replace=False).tolist())
    chosen = np.array(sorted(set(picks)), dtype=int)
    if chosen.size > size:
        chosen = np.sort(rng.choice(chosen, size=size, replace=False))
    return chosen


def build_blocks(
    axes: np.ndarray,
    eval_size: int = EVAL_SIZE,
    tune_size: int = TUNE_SIZE,
    seed: int = BLOCK_SEED,
) -> QuestionBlocks:
    """Draw the cut: eval from the whole axis, tune from the rest, pool is the remainder."""
    axes = np.asarray(axes)
    all_q = np.arange(axes.size)
    rng = np.random.default_rng(seed)
    eval_idx = stratified_sample(axes, eval_size, rng, all_q)
    rest = np.setdiff1d(all_q, eval_idx)
    tune_idx = stratified_sample(axes, tune_size, rng, rest)
    pool_idx = np.setdiff1d(rest, tune_idx)
    blocks = QuestionBlocks(
        eval_idx=eval_idx,
        tune_idx=tune_idx,
        pool_idx=pool_idx,
        n_questions=int(axes.size),
        seed=seed,
    )
    logger.info(f"built question blocks: {blocks.describe()}")
    return blocks


def _from_dict(payload: dict) -> QuestionBlocks:
    """Parse the on-disk form, checking the three blocks actually partition."""
    missing = [k for k in (*BLOCK_NAMES, "n_questions") if k not in payload]
    if missing:
        raise ValueError(f"question blocks payload is missing key(s): {missing}")
    idx = {name: np.asarray(payload[name], dtype=int) for name in BLOCK_NAMES}
    n_questions = int(payload["n_questions"])
    union = np.concatenate(list(idx.values()))
    if union.size != n_questions or np.unique(union).size != n_questions:
        raise ValueError(
            f"the three blocks cover {np.unique(union).size} distinct of "
            f"{union.size} indices and must partition all {n_questions} questions"
        )
    return QuestionBlocks(
        eval_idx=idx["eval"],
        tune_idx=idx["tune"],
        pool_idx=idx["pool"],
        n_questions=n_questions,
        seed=int(payload.get("seed", BLOCK_SEED)),
    )


def load_blocks(
    n_questions: int,
    path: Path | str = DEFAULT_BLOCKS,
    axes: np.ndarray | None = None,
    eval_size: int = EVAL_SIZE,
    tune_size: int = TUNE_SIZE,
    seed: int = BLOCK_SEED,
) -> QuestionBlocks:
    """The cached cut if it is for this many questions, else a fresh draw from ``axes``."""
    path = Path(path)
    if path.exists():
        blocks = _from_dict(json.loads(path.read_text()))
        if blocks.n_questions == n_questions:
            logger.info(f"question blocks from {path}: {blocks.describe()}")
            return blocks
        logger.warning(
            f"{path} was cut on {blocks.n_questions} questions and this cut has "
            f"{n_questions} — redrawing"
        )
    if axes is None:
        raise ValueError(
            f"no usable cached blocks at {path} and no axis labels to draw new ones from"
        )
    return build_blocks(axes, eval_size, tune_size, seed)


def blocks_for_slice(
    sl: Slice,
    path: Path | str = DEFAULT_BLOCKS,
    eval_size: int = EVAL_SIZE,
    tune_size: int = TUNE_SIZE,
    seed: int = BLOCK_SEED,
) -> QuestionBlocks:
    """The cut for one loaded slice; a view of a merged slice gets the cached cut restricted."""
    if sl.columns is None:
        return load_blocks(sl.n_questions, path, axis_labels(sl), eval_size, tune_size, seed)
    parent = _from_dict(json.loads(Path(path).read_text()))
    return restrict_blocks(parent, sl.columns)


def restrict_blocks(blocks: QuestionBlocks, columns: np.ndarray) -> QuestionBlocks:
    """``blocks`` over the parent axis, restricted to ``columns`` and re-indexed to them."""
    position = np.full(blocks.n_questions, -1)
    position[np.asarray(columns, dtype=int)] = np.arange(len(columns))

    def keep(idx: np.ndarray) -> np.ndarray:
        at = position[idx]
        return np.sort(at[at >= 0])

    return QuestionBlocks(
        eval_idx=keep(blocks.eval_idx),
        tune_idx=keep(blocks.tune_idx),
        pool_idx=keep(blocks.pool_idx),
        n_questions=len(columns),
        seed=blocks.seed,
    )


def without_columns(blocks: QuestionBlocks, drop: np.ndarray) -> QuestionBlocks:
    """``blocks`` with ``drop`` removed from the fitting blocks (pool, tune).

    Used by the benchmark-group holdout refits.
    """
    gone = np.asarray(drop, dtype=int)
    return QuestionBlocks(
        eval_idx=blocks.eval_idx,
        tune_idx=np.setdiff1d(blocks.tune_idx, gone),
        pool_idx=np.setdiff1d(blocks.pool_idx, gone),
        n_questions=blocks.n_questions,
        seed=blocks.seed,
    )


def save_blocks(blocks: QuestionBlocks, path: Path | str = DEFAULT_BLOCKS) -> None:
    """Write the cut to disk in the form :func:`load_blocks` reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blocks.as_dict()))
    logger.info(f"wrote {path}: {blocks.describe()}")
