"""Interview selectors: which K questions a -budget arm is fitted on.

A selector sees the training models' correctness matrix ``A`` [M, Q] and returns K
column indices. The paper's interview uses :func:`fisher` (2PL test information);
``random`` is the uniform baseline.

The K contract, enforced in :func:`select`: ``0 <= K <= Q`` returns K distinct
indices; ``K > Q`` is clamped to Q with a warning; ``K < 0`` raises. Ties are broken
with the caller's ``rng``, never by column order, and no selector reads global random
state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger

#: ``fn(A, K, rng) -> K column indices``
SelectorFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class Selector:
    """A registered selection strategy."""

    name: str
    fn: SelectorFn


SELECTORS: dict[str, Selector] = {}


def selector(name: str) -> Callable[[SelectorFn], SelectorFn]:
    """Register a selector under ``name``."""

    def wrap(fn: SelectorFn) -> SelectorFn:
        if name in SELECTORS:
            raise ValueError(f"selector {name!r} is already registered")
        SELECTORS[name] = Selector(name=name, fn=fn)
        return fn

    return wrap


def _normalise_k(K: int, Q: int) -> int:
    """Apply the K contract from the module docstring."""
    K = int(K)
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")
    if K > Q:
        logger.warning(f"K={K} exceeds the {Q} available questions; selecting all of them")
        return Q
    return K


def select(name: str, A: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Pick K question indices from the training matrix ``A`` [M, Q]."""
    if name not in SELECTORS:
        raise ValueError(f"unknown selector {name!r}; registered: {sorted(SELECTORS)}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be [M, Q], got shape {A.shape}")
    K = _normalise_k(K, A.shape[1])
    if K == 0:
        return np.empty(0, dtype=int)
    idx = SELECTORS[name].fn(A, K, rng)
    idx = np.asarray(idx, dtype=int)
    if idx.size != K or np.unique(idx).size != K:
        raise ValueError(
            f"selector {name!r} returned {idx.size} indices "
            f"({np.unique(idx).size} unique), expected {K} unique"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= A.shape[1]):
        raise ValueError(f"selector {name!r} returned out-of-range indices for Q={A.shape[1]}")
    return idx


# --------------------------------------------------------------------------- #
# tie-breaking
# --------------------------------------------------------------------------- #
def _top_k(scores: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """The K highest-scoring columns, ties broken by ``rng``."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((rng.permutation(scores.size), -scores))[:K]


# --------------------------------------------------------------------------- #
# bits-only selectors
# --------------------------------------------------------------------------- #
@selector("random")
def random_select(A: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample."""
    return rng.choice(A.shape[1], size=K, replace=False)


# --------------------------------------------------------------------------- #
# Fisher: 2PL test information
# --------------------------------------------------------------------------- #
def fisher_information(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """[Q] 2PL test information ``TI_q = sum_m a_q^2 P_mq (1 - P_mq)``.

    ``a`` is [Q, d] and ``theta`` [M, d]; the scalar 2PL is ``d = 1``.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    p = 1.0 / (1.0 + np.exp(-np.clip(theta @ a.T + np.asarray(b, dtype=np.float64), -60.0, 60.0)))
    return ((a**2).sum(axis=1)[None, :] * p * (1.0 - p)).sum(axis=0)


@selector("fisher")
def fisher(A: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Top-K questions by 2PL test information over the models in ``A`` (the training rows).

    Fits a unidimensional :class:`~bullpen.competitors.irt.IRTEncoder` on ``A`` with an
    initialisation seeded from ``rng`` and ranks on :func:`fisher_information`.
    """
    # torch-backed, so imported only when this selector runs
    from bullpen.competitors.irt import IRTEncoder

    A = np.asarray(A, dtype=np.float64)
    irt = IRTEncoder("fisher_2pl", dim=1, seed=int(rng.integers(1 << 31))).fit(
        A, np.arange(A.shape[1])
    )
    if irt.a is None or irt.b is None or irt.X is None:
        raise RuntimeError("the 2PL fit returned without item parameters")
    ti = fisher_information(irt.a, irt.b, irt.X)
    logger.info(
        f"fisher: test information in [{ti.min():.4f}, {ti.max():.4f}] over {ti.size} items"
    )
    return _top_k(ti, K, rng)


def interview_columns(
    A_pool: np.ndarray, pool_cols: np.ndarray, k: int, name: str, rng: np.random.Generator
) -> np.ndarray:
    """The K interview questions as sorted global column ids, selected within the pool block.

    ``A_pool`` must hold the training models only.
    """
    pool_cols = np.asarray(pool_cols, dtype=int)
    if pool_cols.size != np.asarray(A_pool).shape[1]:
        raise ValueError(
            f"pool_cols names {pool_cols.size} columns but A_pool has "
            f"{np.asarray(A_pool).shape[1]} — the ids and the block disagree"
        )
    local = select(name, A_pool, k, rng)
    return np.sort(pool_cols[local])
