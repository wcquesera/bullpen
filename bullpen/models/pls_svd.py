"""Closed-form multi-channel encoders: K-block PLS-SVD and the PCA32 channel baseline.

A model's channels are its correctness bits on the fitted questions, its question-text
block ``Qm = A_c Qe[cols] / Q`` (the correctness-weighted mean question embedding, see
:func:`question_block`) and its mean answer embedding. :class:`PLSSVDFusionEncoder` fits
per-block loadings from each block's cross-covariance with the others (inter-battery
factor analysis, Tucker 1958, at two blocks; MB-PLS block scaling, Westerhuis et al. 1998,
from three); :class:`ChannelPCAEncoder` is the PCA of the z-scored blocks at equal weight.

Every basis is fitted on the training rows. A held-out model is placed by
:meth:`fold_in`, which applies the frozen maps; a model with no text row takes the
training mean of the text block. When a fold is too thin to supply ``dim`` directions
the arm lands narrower and warns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from loguru import logger

from bullpen.models.base import Encoder, ridge_solve

#: Floor on a standard deviation before dividing by it.
_SD_FLOOR: float = 1e-9


@dataclass
class Balance:
    """z-score on the fitted rows, then rescale the block to unit total variance."""

    mu: np.ndarray
    sd: np.ndarray
    width: int

    @classmethod
    def fit(cls, X: np.ndarray) -> Balance:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        sd = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(sd > _SD_FLOOR, sd, 1.0), int(X.shape[1]))

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return ((X - self.mu) / self.sd) / np.sqrt(float(self.width))


@dataclass
class Projection:
    """A centred PCA basis: ``mu`` and the leading ``d`` right singular vectors."""

    mu: np.ndarray
    P: np.ndarray  # [d, D]

    @classmethod
    def fit(cls, X: np.ndarray, d: int) -> Projection:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        mu = X.mean(axis=0)
        Vt = np.linalg.svd(X - mu, full_matrices=False)[2]
        return cls(mu, Vt[: int(max(1, min(d, Vt.shape[0])))])

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mu) @ self.P.T

    @property
    def width(self) -> int:
        return int(self.P.shape[0])


@dataclass
class _TextChannelEncoder(Encoder):
    """An encoder that also reads a per-model text table, and how to read a row of it."""

    #: [M_all, Dt] public text block, indexed by row in the full model table
    text_table: np.ndarray | None = field(default=None, repr=False)
    #: which rows of ``text_table`` this fold sees; ``None`` means all of them
    rows: np.ndarray | None = field(default=None, repr=False)
    has_decoder: bool = False
    #: width of the text channel read, recorded in the manifest
    text_width: int = 0

    def _fold_rows(self, A: np.ndarray) -> np.ndarray:
        """This fold's slice of the text table, checked against the bits it pairs with."""
        if self.text_table is None:
            raise ValueError(
                f"{self.name}: needs a text_table [M, Dt]; build it with "
                "bullpen.data.text_banks.pooled_model_text"
            )
        T = np.asarray(self.text_table, dtype=np.float64)
        if T.shape[0] < A.shape[0]:
            raise ValueError(
                f"{self.name}: text table has {T.shape[0]} models, bits have {A.shape[0]}"
            )
        self.text_width = int(T.shape[1])
        if T.shape[0] == A.shape[0]:
            return T
        rows = np.arange(A.shape[0]) if self.rows is None else np.asarray(self.rows)
        return T[rows]

    def _text_row(self, row: int) -> np.ndarray | None:
        """One model's raw text vector, or ``None`` if the table does not reach it."""
        if self.text_table is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        if not 0 <= row < len(self.text_table):
            return None
        return np.asarray(self.text_table[row], dtype=np.float64)


def split_width(total: int, reserved: int = 0, blocks: int = 2) -> tuple[int, ...]:
    """Block widths that sum to ``total - reserved``, bits first; bits absorbs the remainder.

    ``blocks=2`` is ``(bits, text)``; ``blocks=3`` is ``(bits, question text, answer text)``.
    """
    free = int(total) - int(reserved)
    if free < blocks:
        raise ValueError(
            f"a width of {total} leaves {free} column(s) for {blocks} channels after "
            f"{reserved} reserved; each channel needs at least one"
        )
    each = free // blocks
    return (free - each * (blocks - 1),) + (each,) * (blocks - 1)


@dataclass
class _ZScoredBlocks:
    """Column z-scores of the assembled blocks, fitted on the training rows."""

    bits: Balance
    text: Balance
    #: training mean of the z-scored text block — the imputation for a model with no text
    text_mean: np.ndarray

    @classmethod
    def fit(cls, Sb: np.ndarray, St: np.ndarray) -> _ZScoredBlocks:
        # width=1 so this is a plain per-column z-score with no 1/sqrt(d) rescale:
        # the two blocks are the whole code and enter a ridge head column by column
        bits = Balance(Sb.mean(axis=0), _sd(Sb), 1)
        text = Balance(St.mean(axis=0), _sd(St), 1)
        return cls(bits, text, text.apply(St).mean(axis=0))


def _sd(X: np.ndarray) -> np.ndarray:
    sd = np.atleast_2d(X).std(axis=0)
    return np.where(sd > 1e-9, sd, 1.0)


@dataclass
class _FusionBase(_TextChannelEncoder):
    """Shared width declaration of the channel encoders."""

    def _declare_width(self, wanted: int, why: str) -> None:
        """Overwrite ``dim`` with the assembled width, warning when it fell short."""
        self.dim = int(self.X.shape[1])
        if self.dim < wanted:
            logger.warning(
                f"{self.name}: assembled {self.dim} columns, not the run width {wanted} — "
                f"{why} had fewer directions than the blocks had room for, so this arm is "
                "not width-matched to its floor"
            )


def _svd_rank_tol(X: np.ndarray, s: np.ndarray) -> int:
    """Number of singular values of ``X`` above its numerical-rank tolerance (at least 1)."""
    tol = max(X.shape) * np.finfo(np.float64).eps * float(s[0])
    return int(max((s > tol).sum(), 1))


def _frob_scaled(X: np.ndarray) -> np.ndarray:
    """``X / ||X||_F``, so a partner block enters a cross-covariance at unit total variance."""
    norm = float(np.linalg.norm(X))
    return X / norm if norm > 1e-12 else X


def question_block(A_c: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """``A_c @ Q / #cols``: each model's correctness-weighted mean question embedding.

    ``A_c`` is the centred bits on the columns ``Q`` embeds; dividing by the observed
    column count lets a fold-in on K columns estimate the same quantity.
    """
    n = A_c.shape[1]
    return (A_c @ Q) / n if n else np.zeros((A_c.shape[0], Q.shape[1]))


#: The three channels, in block order: correctness bits, question text in model space
#: (:func:`question_block`) and the model's own answer text.
CHANNEL_NAMES: tuple[str, ...] = ("bits", "qtext", "anstext")


@dataclass
class _ChannelBlocks(_FusionBase):
    """Assemble any subset of the three channels as centred model-space blocks.

    ``channels`` names the blocks in :data:`CHANNEL_NAMES` order. ``qtext`` needs
    ``q_text_emb`` (``[Q_all, Dq]``, indexed by global question id) and ``anstext``
    needs ``text_table``.
    """

    channels: tuple[str, ...] = ("bits", "anstext")
    #: [Q_total, Dq] frozen question embeddings, indexed by global question id
    q_text_emb: np.ndarray | None = field(default=None, repr=False)
    #: question-shuffled control: question ``q`` reads row ``q_row_perm[q]`` of
    #: ``q_text_emb`` (a permutation, since the table is re-attached from the slice on load)
    q_row_perm: np.ndarray | None = field(default=None, repr=False)

    _mu_b: np.ndarray | None = field(default=None, repr=False)
    #: per-block training means, in ``channels`` order
    _means: list[np.ndarray] | None = field(default=None, repr=False)

    SHARED_BANKS: ClassVar[dict[str, str]] = {"q_text_emb": "Qe"}

    def __post_init__(self) -> None:
        super().__post_init__()
        self.channels = tuple(self.channels)
        if (
            not self.channels
            or len(set(self.channels)) != len(self.channels)
            or tuple(c for c in CHANNEL_NAMES if c in self.channels) != self.channels
        ):
            raise ValueError(
                f"{self.name}: channels {self.channels} must be distinct members of "
                f"{CHANNEL_NAMES}, in that order"
            )

    def _questions(self, cols: np.ndarray) -> np.ndarray:
        """The question embeddings of global ids ``cols``, checked against the table."""
        if self.q_text_emb is None:
            raise ValueError(
                f"{self.name}: needs a [Q_all, d_text] question embedding table for its qtext block"
            )
        table = np.asarray(self.q_text_emb)
        cols = np.asarray(cols, dtype=int)
        if table.ndim != 2 or (cols.size and cols.max() >= table.shape[0]):
            raise ValueError(
                f"{self.name}: q_text_emb {table.shape} does not cover question ids up to "
                f"{int(cols.max()) if cols.size else -1}"
            )
        if self.q_row_perm is not None:
            cols = np.asarray(self.q_row_perm)[cols]
        return np.asarray(table[cols], dtype=np.float64)

    def _centred_blocks(self, A: np.ndarray, cols: np.ndarray) -> list[np.ndarray]:
        """The fit rows' blocks, each centred on its training mean (kept for fold-in)."""
        A = self._prepare(A, cols)
        self._mu_b = A.mean(axis=0)
        raw = []
        for channel in self.channels:
            if channel == "bits":
                raw.append(A)
            elif channel == "qtext":
                raw.append(question_block(A - self._mu_b, self._questions(self.cols)))
            else:
                raw.append(self._fold_rows(A))
        self._means = [X.mean(axis=0) for X in raw]
        return [X - mu for X, mu in zip(raw, self._means, strict=True)]

    def _centred_row(
        self, bits: np.ndarray, cols: np.ndarray, row: int
    ) -> tuple[np.ndarray, list[np.ndarray | None]]:
        """``(positions, blocks)`` of one model, centred by the frozen training means.

        The bits block covers the observed columns only (positions ``j``); ``Qm`` is
        computed from those bits alone. The answer-text block is ``None`` for a model
        the table does not reach.
        """
        j = self.local(cols)
        yc = np.asarray(bits, dtype=np.float64).ravel() - self._mu_b[j]
        parts: list[np.ndarray | None] = []
        for channel, mu in zip(self.channels, self._means, strict=True):
            if channel == "bits":
                parts.append(yc)
            elif channel == "qtext":
                parts.append(question_block(yc[None], self._questions(cols))[0] - mu)
            else:
                text = self._text_row(row)
                parts.append(None if text is None else text - mu)
        return j, parts


@dataclass
class PLSSVDFusionEncoder(_ChannelBlocks):
    """K-block PLS-SVD: ``[X_1 W_1 | ... | X_K W_K]``, each block's loadings steered by the rest.

    Each block's loadings are the leading left singular vectors of its
    cross-covariance with the other blocks side by side::

        W_k = svd_left( X_k^T [X~_i for i != k] )

    where ``X~`` is block ``X`` centred and scaled to unit Frobenius norm. At two
    blocks this is inter-battery factor analysis (Tucker 1958); the partner scaling
    is MB-PLS block scaling (Westerhuis et al. 1998). The code is each block's scores
    z-scored on the training rows and concatenated at widths
    ``split_width(dim, blocks=K)`` (16/16 at d=32 for two blocks, 12/10/10 for
    three). Every SVD is truncated at its numerical rank.

    Fold-in: bits by a ridge solve on the frozen loadings of the observed columns;
    ``Qm`` from the observed bits alone; answer text by lookup (the training-mean
    block for a model the table does not reach).
    """

    _W: list[np.ndarray] | None = field(default=None, repr=False)
    _z: list[Balance] | None = field(default=None, repr=False)
    #: training mean of each z-scored block, imputed for a model with no text
    _z_mean: list[np.ndarray] | None = field(default=None, repr=False)
    #: kept width per block, in ``channels`` order
    block_widths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.channels) < 2:
            raise ValueError(
                f"{self.name}: a PLS-SVD needs at least two blocks, got {self.channels}"
            )

    def fit(self, A: np.ndarray, cols: np.ndarray) -> PLSSVDFusionEncoder:
        blocks = self._centred_blocks(A, cols)
        want = split_width(self.dim, blocks=len(blocks))
        self._W, ranks, lead = pls_svd_loadings(blocks, want)
        scores = [X @ W for X, W in zip(blocks, self._W, strict=True)]
        # width=1 so this is a plain per-column z-score with no 1/sqrt(d) rescale:
        # the blocks are the whole code and enter a ridge head column by column
        self._z = [Balance(S.mean(axis=0), _sd(S), 1) for S in scores]
        Z = [z.apply(S) for z, S in zip(self._z, scores, strict=True)]
        self._z_mean = [Zk.mean(axis=0) for Zk in Z]
        self.X = np.hstack(Z)
        self.block_widths = tuple(int(W.shape[1]) for W in self._W)
        self._declare_width(sum(want), "a block cross-covariance")
        logger.info(
            f"{self.name}: "
            + " + ".join(f"{c} {w}" for c, w in zip(self.channels, self.block_widths, strict=True))
            + f" singular vectors (ranks {ranks}; blocks "
            + " x ".join(str(X.shape[1]) for X in blocks)
            + f"; leading {[f'{v:.3g}' for v in lead]})"
        )
        return self

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Frozen loadings on the observed bits; Qm from those bits alone; text by lookup."""
        if self._z is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        j, parts = self._centred_row(bits, cols, row)
        out = []
        for k, (channel, x) in enumerate(zip(self.channels, parts, strict=True)):
            if x is None:
                out.append(self._z_mean[k])
                continue
            s = ridge_solve(self._W[k][j], x)[None] if channel == "bits" else x[None] @ self._W[k]
            out.append(self._z[k].apply(s)[0])
        return np.hstack(out)


def pls_svd_loadings(
    blocks: list[np.ndarray], widths: tuple[int, ...]
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Each centred block's leading left singular vectors against the others, Frobenius-scaled.

    Returns ``(loadings, ranks, leading singular values)``, one per block; block
    ``k`` keeps ``min(widths[k], rank_k)`` directions.
    """
    scaled = [_frob_scaled(X) for X in blocks]
    loadings, ranks, lead = [], [], []
    for k, (X, w) in enumerate(zip(blocks, widths, strict=True)):
        C = X.T @ np.hstack([scaled[i] for i in range(len(blocks)) if i != k])
        try:
            U, s, _ = np.linalg.svd(C, full_matrices=False)
        except np.linalg.LinAlgError:
            # gesdd can fail to converge on ill-conditioned blocks; gesvd is slower but converges
            import scipy.linalg

            U, s, _ = scipy.linalg.svd(C, full_matrices=False, lapack_driver="gesvd")
        rank = _svd_rank_tol(C, s)
        loadings.append(U[:, : min(w, rank)])
        ranks.append(rank)
        lead.append(float(s[0]))
    return loadings, ranks, lead


@dataclass
class ChannelPCAEncoder(_ChannelBlocks):
    """The PCA32-{channel} baseline: z-score, equal block weights, hstack, leading ``dim`` PCs.

    Every column is z-scored on the training rows and every block scaled to unit
    total variance (:class:`Balance`), so blocks of very different widths enter the
    PCA with equal weight. Fold-in places observed bits in their columns with every
    unobserved column at the training mean; ``Qm`` and answer text as in
    :class:`PLSSVDFusionEncoder`.
    """

    _z: list[Balance] | None = field(default=None, repr=False)
    _z_mean: list[np.ndarray] | None = field(default=None, repr=False)
    _joint: Projection | None = field(default=None, repr=False)

    def fit(self, A: np.ndarray, cols: np.ndarray) -> ChannelPCAEncoder:
        blocks = self._centred_blocks(A, cols)
        self._z = [Balance.fit(X) for X in blocks]
        Z = [z.apply(X) for z, X in zip(self._z, blocks, strict=True)]
        self._z_mean = [Zk.mean(axis=0) for Zk in Z]
        joint = np.hstack(Z)
        self._joint = Projection.fit(joint, self.dim)
        self.X = self._joint.apply(joint)
        self._declare_width(self.dim, "the joint block")
        logger.info(
            f"{self.name}: PCA{self.dim} of {' + '.join(self.channels)} ("
            + " + ".join(str(X.shape[1]) for X in blocks)
            + " columns, each block at unit total variance)"
        )
        return self

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        if self._joint is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        j, parts = self._centred_row(bits, cols, row)
        out = []
        for k, (channel, x) in enumerate(zip(self.channels, parts, strict=True)):
            if x is None:
                out.append(self._z_mean[k])
                continue
            if channel == "bits":  # unobserved columns at the training mean
                placed = np.zeros(self.input_width)
                placed[j] = x
                out.append(self._z[k].apply(placed)[0])
            else:
                out.append(self._z[k].apply(x)[0])
        return self._joint.apply(np.hstack(out)[None])[0]
