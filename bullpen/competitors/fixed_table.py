"""Table-lookup arms and null baselines.

``FixedEncoder`` looks up a fixed per-model vector (``text_mean768``, the Gaussian floors
``null_random{d}``); ``LeaderboardEncoder`` is the model's mean accuracy
(``null_leaderboard``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bullpen.models.base import Encoder

#: Width of the leaderboard null: one number per model, its mean accuracy.
LEADERBOARD_DIM: int = 1


@dataclass
class FixedEncoder(Encoder):
    """A per-model vector that does not depend on the bits; fold-in is a lookup."""

    #: [M_all, d] the table this arm looks up, indexed by row in the full model table
    table: np.ndarray | None = field(default=None, repr=False)
    #: which rows of ``table`` this fold exposes as training embeddings
    rows: np.ndarray | None = field(default=None, repr=False)
    uses_probe: bool = False
    has_decoder: bool = False
    interview_requirement: str = "nothing"

    def fit(self, A: np.ndarray, cols: np.ndarray) -> FixedEncoder:
        """Record the column space and select this fold's rows. Nothing is learned."""
        self._prepare(A, cols)
        if self.table is None:
            raise ValueError(f"{self.name}: needs a table [M, d] to look embeddings up in")
        table = np.asarray(self.table, dtype=np.float64)
        self.X = table if self.rows is None else table[np.asarray(self.rows)]
        return self

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """The model's own row of the table; the interview is not read."""
        if self.table is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        return np.asarray(self.table[row], dtype=np.float64)


@dataclass
class LeaderboardEncoder(Encoder):
    """One-dimensional competence: the model's mean accuracy (the leaderboard null)."""

    dim: int = LEADERBOARD_DIM
    has_decoder: bool = False
    interview_requirement: str = "nothing"

    def fit(self, A: np.ndarray, cols: np.ndarray) -> LeaderboardEncoder:
        A = self._prepare(A, cols)
        self.X = A.mean(axis=1, keepdims=True)
        return self

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """The mean of whatever was observed, however little that is."""
        bits = np.asarray(bits, dtype=np.float64).ravel()
        if bits.size == 0:
            raise ValueError(f"{self.name}: cannot average an empty interview")
        return np.array([float(bits.mean())])


__all__ = [
    "LEADERBOARD_DIM",
    "FixedEncoder",
    "LeaderboardEncoder",
]
