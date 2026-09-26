"""Shared readouts. Every battery task and every decoder scores through here."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold

#: Ridge grid for the out-of-fold residualiser.
ALPHAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

#: Default number of percentile-bootstrap resamples.
N_BOOT: int = 10_000

#: Default two-sided level for bootstrap intervals.
ALPHA: float = 0.05

#: Neighbour metrics.
METRICS: tuple[str, ...] = ("cosine", "euclidean", "euclidean_z")


#: Shrinkage strengths searched by :func:`select_shrinkage_lambda`.
LAMBDA_GRID: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
)


_EPS = 1e-9


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_float(x: np.ndarray, name: str, *, ndim: int | None = None) -> np.ndarray:
    """Coerce to a finite float array of the expected rank, or explain why not."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-d, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(
            f"{name} contains non-finite entries; drop or impute the affected "
            "rows before scoring rather than letting them reach the metric"
        )
    return a


def _check_same_rows(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{name_a} has {a.shape[0]} rows, {name_b} has {b.shape[0]}; "
            "both must be indexed by the same models, in the same order"
        )


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BootstrapCI:
    """A mean with a percentile interval, and the ``n`` it was computed over."""

    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(
    values: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = 0
) -> BootstrapCI:
    """Percentile bootstrap of the mean over independent units."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("bootstrap_ci got an empty array")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise ValueError("every entry is non-finite; there is nothing to summarise")
    if finite.size < v.size:
        logger.debug("bootstrap_ci dropped {} non-finite of {} units", v.size - finite.size, v.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, finite.size, size=(n_boot, finite.size))
    means = finite[idx].mean(axis=1)
    return BootstrapCI(
        mean=float(finite.mean()),
        lo=float(np.quantile(means, alpha / 2)),
        hi=float(np.quantile(means, 1 - alpha / 2)),
        n=int(finite.size),
    )


# --------------------------------------------------------------------------- #
# regression readouts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Residualised:
    """Out-of-fold residual of a target on a feature block, and the R^2 removed."""

    residual: np.ndarray
    r2: float


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of variance explained, pooled over every cell of a 1-d or 2-d target."""
    y = _as_float(y_true, "y_true")
    p = _as_float(y_pred, "y_pred")
    if y.shape != p.shape:
        raise ValueError(f"y_true has shape {y.shape}, y_pred has shape {p.shape}")
    if y.shape[0] < 2:
        raise ValueError("R^2 needs at least 2 observations")
    y2 = y if y.ndim > 1 else y[:, None]
    p2 = p if p.ndim > 1 else p[:, None]
    total = float(((y2 - y2.mean(axis=0)) ** 2).sum())
    if total <= _EPS:
        raise ValueError(
            "y_true is constant, so R^2 is undefined; check that the target was "
            "built over more than one distinct unit"
        )
    return float(1.0 - ((y2 - p2) ** 2).sum() / total)


def model_folds(
    n: int, n_folds: int, seed: int, groups: Sequence[Any] | None = None
) -> KFold | GroupKFold:
    """The splitter an out-of-fold fit over the model axis folds with."""
    n_groups = len({str(g) for g in groups}) if groups is not None else n
    if not 2 <= n_folds <= n_groups:
        what = "group(s)" if groups is not None else "row(s)"
        raise ValueError(f"n_folds must be in [2, {n_groups}] for {n_groups} {what}, got {n_folds}")
    if groups is None:
        return KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def oof_ridge_predict(
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    alphas: tuple[float, ...] = ALPHAS,
    groups: Sequence[Any] | None = None,
) -> np.ndarray:
    """Out-of-fold ridge predictions of ``Y`` from ``X``, one prediction per row."""
    Xa = _as_float(X, "X", ndim=2)
    Ya = _as_float(Y, "Y")
    _check_same_rows(Xa, Ya, "X", "Y")
    n = Xa.shape[0]
    if groups is not None and len(groups) != n:
        raise ValueError(f"groups has {len(groups)} entries for the {n} rows of X")
    Y2 = Ya if Ya.ndim > 1 else Ya[:, None]
    out = np.zeros_like(Y2)
    for tr, te in model_folds(n, n_folds, seed, groups).split(Xa, groups=groups):
        mu = Xa[tr].mean(axis=0)
        sd = np.maximum(Xa[tr].std(axis=0), _EPS)
        model = RidgeCV(alphas=alphas).fit((Xa[tr] - mu) / sd, Y2[tr])
        out[te] = model.predict((Xa[te] - mu) / sd).reshape(len(te), -1)
    return out.reshape(Ya.shape)


def residualise(Y: np.ndarray, X: np.ndarray, n_folds: int = 5, seed: int = 0) -> Residualised:
    """Remove from ``Y`` whatever ``X`` can predict out of fold."""
    pred = oof_ridge_predict(X, Y, n_folds=n_folds, seed=seed)
    Ya = np.asarray(Y, dtype=np.float64)
    return Residualised(residual=Ya - pred, r2=r2_score(Ya, pred))


# --------------------------------------------------------------------------- #
# AUROC readouts
# --------------------------------------------------------------------------- #
def _auc_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mann–Whitney AUROC from mid-ranks, column-wise. Ties averaged."""
    n1 = labels.sum(axis=0)
    n0 = labels.shape[0] - n1
    auc = ((ranks * labels).sum(axis=0) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
    return np.where((n1 > 0) & (n0 > 0), auc, np.nan)


def row_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of one model's predicted row against its true bits."""
    yv = _as_float(y, "y", ndim=1)
    sv = _as_float(scores, "scores", ndim=1)
    if yv.shape != sv.shape:
        raise ValueError(f"y has {yv.size} entries, scores has {sv.size}")
    labels = (yv >= 0.5).astype(np.float64)
    if labels.min() == labels.max():
        raise ValueError(
            "the label vector is constant, so AUROC is undefined; this row is "
            "all-correct or all-incorrect and cannot be ordered"
        )
    return float(_auc_from_ranks(rankdata(sv)[:, None], labels[:, None])[0])


def column_auroc(P: np.ndarray, R: np.ndarray, min_models: int = 4) -> np.ndarray:
    """Per-question AUROC ACROSS models — the difficulty-controlled readout."""
    Pa = _as_float(P, "P", ndim=2)
    Ra = _as_float(R, "R", ndim=2)
    if Pa.shape != Ra.shape:
        raise ValueError(f"predictions have shape {Pa.shape}, correctness has {Ra.shape}")
    if Pa.shape[0] < min_models:
        raise ValueError(
            f"column AUROC needs at least {min_models} models, got {Pa.shape[0]}; "
            "subset the rows to the models that were actually scored"
        )
    labels = (Ra >= 0.5).astype(np.float64)
    return _auc_from_ranks(rankdata(Pa, axis=0), labels)


# --------------------------------------------------------------------------- #
# ranking readouts
# --------------------------------------------------------------------------- #
def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation between two orderings of the same units."""
    x = _as_float(a, "a", ndim=1)
    y = _as_float(b, "b", ndim=1)
    if x.shape != y.shape:
        raise ValueError(f"a has {x.size} entries, b has {y.size}")
    if x.size < 2:
        raise ValueError("rank correlation needs at least 2 units")
    ra, rb = rankdata(x), rankdata(y)
    if ra.std() < _EPS or rb.std() < _EPS:
        raise ValueError("one side is constant (all values tied), so its ranking is undefined")
    return float(np.corrcoef(ra, rb)[0, 1])


def ndcg_at_k(true: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Discounted gain of the top-``k`` units a ranker picked, over the best possible."""
    t = _as_float(true, "true", ndim=1)
    s = _as_float(scores, "scores", ndim=1)
    if t.shape != s.shape:
        raise ValueError(f"true has {t.size} entries, scores has {s.size}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if (t < 0).any():
        raise ValueError("NDCG is undefined for negative gains; shift or clip `true` first")
    kk = min(k, t.size)
    discount = 1.0 / np.log2(np.arange(2, kk + 2))
    ideal = float((np.sort(t)[::-1][:kk] * discount).sum())
    if ideal <= _EPS:
        raise ValueError("every gain is zero, so there is no ideal ranking to score against")
    gains = float((t[np.argsort(-s)[:kk]] * discount).sum())
    return gains / ideal


# --------------------------------------------------------------------------- #
# nulls
# --------------------------------------------------------------------------- #
def pool_mean_rows(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """Every model predicted by the question mean — the no-model-information null."""
    Aa = _as_float(A, "A", ndim=2)
    rng = np.random.default_rng(seed)
    base = np.broadcast_to(Aa.mean(axis=0), Aa.shape).copy()
    return base + rng.normal(scale=1e-9, size=Aa.shape)


# --------------------------------------------------------------------------- #
# neighbour prediction
# --------------------------------------------------------------------------- #
def _similarity(X: np.ndarray, metric: str) -> np.ndarray:
    """Higher is nearer: cosine similarity, or negated Euclidean distance."""
    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)
        return Xn @ Xn.T
    Z = X - X.mean(axis=0)
    if metric == "euclidean_z":
        Z = Z / np.maximum(X.std(axis=0), _EPS)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)
    return -np.sqrt(np.maximum(d2, 0.0))


def _neighbour_weights(sim: np.ndarray, metric: str) -> np.ndarray:
    """Similarity → non-negative weights, uniform if every neighbour is repelled."""
    w = np.clip(sim, 0.0, None) if metric == "cosine" else 1.0 / (1.0 - sim)
    return np.ones_like(w) if w.sum() <= _EPS else w


def neighbour_predict(
    X: np.ndarray,
    R: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Leave-one-model-out neighbour prediction of every row. Returns [M, Q]."""
    Xa = _as_float(X, "X", ndim=2)
    Ra = _as_float(R, "R", ndim=2)
    _check_same_rows(Xa, Ra, "X", "R")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    m = Xa.shape[0]
    use = np.ones(m, bool) if mask is None else np.asarray(mask, bool)
    if use.shape != (m,):
        raise ValueError(f"mask must have {m} entries, got shape {use.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if int(use.sum()) < k + 1:
        raise ValueError(
            f"k={k} needs at least {k + 1} usable models (one is always held out), "
            f"but the mask leaves {int(use.sum())}"
        )
    sim = _similarity(Xa, metric)
    np.fill_diagonal(sim, -np.inf)
    sim[:, ~use] = -np.inf
    out = np.full(Ra.shape, np.nan, dtype=np.float64)
    for i in np.flatnonzero(use):
        nb = _nearest(sim[i], k)
        out[i] = np.average(Ra[nb], axis=0, weights=_neighbour_weights(sim[i, nb], metric))
    return out


#: Relative tolerance below which two similarities are treated as tied.
NEIGHBOUR_TIE_RTOL: float = 1e-12


def _nearest(sim_row: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` nearest indices, stable under last-bit noise, ties by index."""
    finite = sim_row[np.isfinite(sim_row)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    if not scale > 0.0:
        scale = 1.0
    quantised = np.round(sim_row / (scale * NEIGHBOUR_TIE_RTOL))
    return np.lexsort((np.arange(sim_row.size), -quantised))[:k]


# --------------------------------------------------------------------------- #
# competence removal and specialisation
# --------------------------------------------------------------------------- #
def profile_target(A_half: np.ndarray, n_pc: int = 8) -> np.ndarray:
    """PC2..PC``n_pc`` of a half-matrix — "what is it good AT", competence out."""
    A = _as_float(A_half, "A_half", ndim=2)
    if n_pc < 2:
        raise ValueError(f"n_pc must be >= 2 to leave a component after dropping PC1, got {n_pc}")
    if A.shape[0] < 3:
        raise ValueError("a specialisation profile needs at least 3 models")
    if A.shape[1] < 2:
        raise ValueError("a one-question half has only a competence component to drop")
    Ac = A - A.mean(axis=0)
    U, s, _ = np.linalg.svd(Ac, full_matrices=False)
    keep = min(n_pc, s.size)
    return (U[:, :keep] * s[:keep])[:, 1:]


def pca95(Z: np.ndarray) -> int:
    """Number of components carrying 95% of a profile bank's variance."""
    Za = _as_float(Z, "Z", ndim=2)
    if Za.shape[0] < 2:
        raise ValueError("the rank of a bank with fewer than 2 models is not defined")
    Zc = Za - Za.mean(axis=0)
    s = np.linalg.svd(Zc, compute_uv=False) ** 2
    if s.sum() <= _EPS:
        raise ValueError("every model sits at the same point; this bank has collapsed entirely")
    return int(np.searchsorted(np.cumsum(s) / s.sum(), 0.95) + 1)


@dataclass(frozen=True)
class ProfileReliability:
    """Test-retest reliability of a specialisation target, and the thinnest row."""

    reliability: float
    lo: float
    hi: float
    n_models: int
    reliability_ex_thinnest: float
    thinnest_row: int
    thinnest_coverage: float


def profile_target_reliability(
    A: np.ndarray,
    coverage: np.ndarray | None = None,
    n_pc: int = 8,
    n_draws: int = 50,
    seed: int = 0,
) -> ProfileReliability:
    """How reliable is the target a :func:`specialisation_r2` is scored against?"""
    Aa = _as_float(A, "A", ndim=2)
    n_models, n_questions = Aa.shape
    if n_models < 4:
        raise ValueError(f"profile reliability needs at least 4 models, got {n_models}")
    if n_questions < 8:
        raise ValueError(
            f"splitting the question axis needs at least 8 questions, got {n_questions}"
        )

    def measure(X: np.ndarray) -> BootstrapCI:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_draws):
            perm = rng.permutation(X.shape[1])
            Y1 = profile_target(X[:, perm[: X.shape[1] // 2]], n_pc=n_pc)
            Y2 = profile_target(X[:, perm[X.shape[1] // 2 :]], n_pc=n_pc)
            keep = min(Y1.shape[1], Y2.shape[1])
            Y1, Y2 = Y1[:, :keep], Y2[:, :keep]
            # components are identified only up to sign; align before pooling or
            # half the columns cancel the other half
            Y2 = Y2 * np.sign((Y1 * Y2).sum(axis=0) + 1e-12)
            draws.append(_pooled_corr(Y1, Y2))
        return bootstrap_ci(np.array(draws), seed=seed)

    ci = measure(Aa)
    thin, thin_cov, ex_thin = -1, float("nan"), float("nan")
    if coverage is not None:
        cov = _as_float(coverage, "coverage", ndim=1)
        if cov.size != n_models:
            raise ValueError(f"coverage has {cov.size} entries, the matrix has {n_models} models")
        thin = int(np.argmin(cov))
        thin_cov = float(cov[thin])
        ex_thin = measure(np.delete(Aa, thin, axis=0)).mean
    return ProfileReliability(
        reliability=ci.mean,
        lo=ci.lo,
        hi=ci.hi,
        n_models=n_models,
        reliability_ex_thinnest=ex_thin,
        thinnest_row=thin,
        thinnest_coverage=thin_cov,
    )


def _pooled_corr(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """Pearson correlation over every cell of two aligned profile matrices."""
    a, b = Y1.ravel(), Y2.ravel()
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
