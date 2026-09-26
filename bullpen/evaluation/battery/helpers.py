"""Task-construction helpers shared by several battery readouts."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from bullpen.evaluation.battery.constants import (
    DIAG_LIST_SEP,
    DIVERGENCE_QUESTIONS,
    EXTERNAL_PERMUTATIONS,
    LABEL_PERMUTATIONS,
    LOGISTIC_C,
    LOGISTIC_MAX_ITER,
    MIN_CONTRAST_AXES,
    MIN_FAMILY_MEMBERS,
    MIN_HEADROOM,
    MIN_ITEM_WEIGHT_GB,
    MIN_LABELLED_FIT,
    MIN_LABELLED_SCORE,
    MIN_PAIR_COVERAGE,
    STD_FLOOR,
    UNIT_BENCHMARK_CLUSTERS,
    UNIT_MODELS,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.shared import (
    _cluster_ci,
    _headroom,
    _mean_or_nan,
    _no_ci,
    _or_nan,
    _resample,
    _resample_pairs,
    _ridge_fit_predict,
)
from bullpen.evaluation.metrics import (
    model_folds,
    oof_ridge_predict,
    r2_score,
    residualise,
    spearman_rho,
)


def _pair_design(Z: np.ndarray, Y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All within-block model pairs as (features, which-one-won) arrays."""
    pairs = list(itertools.combinations(rows, 2))
    if not pairs:
        return np.empty((0, 2 * Z.shape[1])), np.empty((0, Y.shape[1]))
    features = np.array([np.concatenate([np.abs(Z[i] - Z[j]), Z[i] - Z[j]]) for i, j in pairs])
    labels = np.array([(Y[i] > Y[j]).astype(float) for i, j in pairs])
    return features, labels


def _competence_r2(ctx: TaskContext, cols: np.ndarray) -> float:
    """How much of plain competence on ``cols`` the bank explains out of fold."""
    competence = ctx.A[:, cols].mean(axis=1)[:, None]
    return residualise(competence, ctx.Z, n_folds=ctx.n_folds, seed=ctx.seed).r2


def _greedy_knapsack(rows: np.ndarray, gb: np.ndarray, budget: float) -> list[int]:
    """Ratio-greedy max-coverage under a weight budget, with the best-single guard."""
    chosen: list[int] = []
    best = np.zeros(rows.shape[1])
    spent = 0.0
    while True:
        affordable = [
            i for i in range(rows.shape[0]) if i not in chosen and gb[i] + spent <= budget
        ]
        if not affordable:
            break
        gain, pick = max(
            (
                (np.maximum(best, rows[i]).mean() - best.mean()) / max(gb[i], MIN_ITEM_WEIGHT_GB),
                i,
            )
            for i in affordable
        )
        if gain <= 0:
            break
        chosen.append(pick)
        best = np.maximum(best, rows[pick])
        spent += gb[pick]
    feasible = np.flatnonzero(gb <= budget)
    if feasible.size:
        single = int(feasible[np.argmax(rows[feasible].mean(axis=1))])
        if not chosen or rows[single].mean() > best.mean():
            return [single]
    return chosen


def _random_squad(
    rng: np.random.Generator, cost: np.ndarray, budget: float, size: int
) -> list[int]:
    """An affordable squad of at most ``size`` members, drawn in a random order."""
    squad: list[int] = []
    spent = 0.0
    for candidate in rng.permutation(cost.size):
        if len(squad) >= size:
            break
        if spent + cost[candidate] <= budget:
            squad.append(int(candidate))
            spent += float(cost[candidate])
    return squad


def _portfolio_row(A_score: np.ndarray, squad: list[int]) -> np.ndarray:
    """[Q] oracle-routed accuracy of a portfolio, question by question."""
    if not squad:
        return np.zeros(A_score.shape[1])
    return np.maximum.reduce(A_score[squad])


def _top1_routing(ctx: TaskContext, predicted: np.ndarray) -> TaskMetrics:
    """Route every question to the argmax model of ``predicted``, and price the rule."""
    truth = ctx.A
    picked = predicted.argmax(axis=0)
    realised = truth[picked, np.arange(ctx.n_questions)]
    oracle = truth.max(axis=0)
    single = truth[int(truth.mean(axis=1).argmax())]
    headroom = float(oracle.mean() - single.mean())
    if headroom <= MIN_HEADROOM:
        return _absent(
            "gap_closed",
            "the per-question oracle and the best single model score the same on this "
            "block — there is no routing headroom to close a fraction of",
        )
    return {
        "gap_closed": float((realised.mean() - single.mean()) / headroom),
        "routing_acc": float(realised.mean()),
        "routing_oracle": float(oracle.mean()),
        "routing_best_single": float(single.mean()),
        "oracle_headroom": headroom,
        "n_routing_questions": ctx.n_questions,
        **_cluster_ci(
            lambda counts: _gap_over(realised, oracle, single, _resample(counts, ctx.bench)),
            float((realised.mean() - single.mean()) / headroom),
            ctx.n_benchmarks,
            UNIT_BENCHMARK_CLUSTERS,
            ctx.seed,
        ),
    }


def _gap_over(
    realised: np.ndarray, oracle: np.ndarray, floor: np.ndarray, cols: np.ndarray
) -> float:
    """``(realised - floor) / (oracle - floor)`` on one subset of the scored questions."""
    if cols.size == 0:
        return float("nan")
    return _headroom(
        float(realised[cols].mean()), float(floor[cols].mean()), float(oracle[cols].mean())
    )


def _absent(primary: str, note: str) -> TaskMetrics:
    """A task whose input the context does not carry, said in the table."""
    logger.info("task input absent: {}", note)
    return {primary: float("nan"), "note": note, **_no_ci()}


def _folds_over(rows: np.ndarray, ctx: TaskContext, groups: np.ndarray | None = None) -> int:
    """``ctx.n_folds``, lowered when a labelled subset holds fewer models."""
    ceiling = len({str(g) for g in groups}) if groups is not None else int(rows.size)
    return max(2, min(ctx.n_folds, ceiling))


def _labelled_rows(mask: np.ndarray) -> np.ndarray:
    """Row indices a label is measured on, as an int array."""
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _too_few(rows: np.ndarray) -> bool:
    """True when a labelled block is too small to both fit and score on."""
    return rows.size < MIN_LABELLED_FIT + MIN_LABELLED_SCORE


def _oof_scalar(
    Z: np.ndarray, y: np.ndarray, n_folds: int, seed: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Out-of-fold ridge from one feature block to one scalar target, as a flat [n]."""
    return oof_ridge_predict(Z, y[:, None], n_folds=n_folds, seed=seed, groups=groups).ravel()


def _scalar_transfer(
    ctx: TaskContext,
    y: np.ndarray,
    rows: np.ndarray,
    primary: str = "r2",
    groups: np.ndarray | None = None,
) -> TaskMetrics:
    """Out-of-fold ridge from the bank to one per-model scalar, on ``rows`` only."""
    if primary not in ("r2", "spearman"):
        raise ValueError(f"_scalar_transfer headlines r2 or spearman, not {primary!r}")
    readout = r2_score if primary == "r2" else spearman_rho
    n_folds = _folds_over(rows, ctx, groups)
    truth = y[rows]
    predicted = _oof_scalar(ctx.Z[rows], truth, n_folds, ctx.seed, groups)
    return {
        "r2": _or_nan(r2_score, truth, predicted),
        "spearman": _or_nan(spearman_rho, truth, predicted),
        # the anchor: the same fit on a model-permuted bank
        f"{primary}_permuted_null": _permuted_null(
            rows,
            lambda order: _or_nan(
                readout, truth, _oof_scalar(ctx.Z[order], truth, n_folds, ctx.seed, groups)
            ),
            ctx.seed,
            EXTERNAL_PERMUTATIONS,
        ),
        "n_labelled": int(rows.size),
        "n_folds": n_folds,
        "grouped_cv": float(groups is not None),
        **_cluster_ci(
            lambda counts: readout(*(a[_resample(counts)] for a in (truth, predicted))),
            _or_nan(readout, truth, predicted),
            int(rows.size),
            UNIT_MODELS,
            ctx.seed,
        ),
    }


def _oof_logistic(
    ctx: TaskContext, y: np.ndarray, rows: np.ndarray, groups: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(predicted label, decision score)`` out of fold for the models in ``rows``."""
    Z = ctx.Z[rows]
    labels = np.asarray(y)[rows]
    predicted = np.empty(rows.size, dtype=labels.dtype)
    # a fold that saw one class scores 0 (a tie) rather than NaN
    score = np.zeros(rows.size)
    splitter = model_folds(rows.size, _folds_over(rows, ctx, groups), ctx.seed, groups)
    for fit, held in splitter.split(np.arange(rows.size), groups=groups):
        if np.unique(labels[fit]).size < 2:
            logger.debug("a fold saw one class only; its held-out models get a tied score")
            predicted[held] = labels[fit][0]
            continue
        mu = Z[fit].mean(axis=0)
        sd = np.maximum(Z[fit].std(axis=0), STD_FLOOR)
        clf = LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER).fit(
            (Z[fit] - mu) / sd, labels[fit]
        )
        Zh = (Z[held] - mu) / sd
        predicted[held] = clf.predict(Zh)
        margin = clf.decision_function(Zh)
        score[held] = margin if margin.ndim == 1 else margin.max(axis=1)
    return predicted, score


def _permuted_null(
    truth: np.ndarray,
    score: Callable[[np.ndarray], float],
    seed: int,
    permutations: int = LABEL_PERMUTATIONS,
) -> float:
    """Mean of ``score`` over ``permutations`` shuffles of ``truth``."""
    rng = np.random.default_rng(seed)
    return float(np.mean([score(rng.permutation(truth)) for _ in range(permutations)]))


def _family_rows(ctx: TaskContext) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, labels)`` for the models in a family with enough members."""
    families = np.asarray(ctx.family, dtype=object)
    counts = {f: int((families == f).sum()) for f in set(ctx.family) if f}
    keep = {f for f, n in counts.items() if n >= MIN_FAMILY_MEMBERS}
    rows = np.flatnonzero(np.isin(families, sorted(keep)))
    return rows, families[rows]


def _model_pairs(rows: np.ndarray) -> np.ndarray:
    """All unordered within-block model pairs, as a [P, 2] array of row indices."""
    pairs = list(itertools.combinations(np.asarray(rows, dtype=int).tolist(), 2))
    return np.asarray(pairs, dtype=int).reshape(len(pairs), 2)


def _pair_features(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """[P, 2d+2] features for a model pair, symmetric in the two endpoints."""
    a, b = Z[pairs[:, 0]], Z[pairs[:, 1]]
    norms = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), STD_FLOOR)
    cosine = (a * b).sum(axis=1) / norms
    return np.column_stack([np.abs(a - b), a * b, cosine, np.linalg.norm(a - b, axis=1)])


def _competence_pair_features(ctx: TaskContext, pairs: np.ndarray) -> np.ndarray:
    """[P, 2] the two things a pair target can be predicted from without a bank."""
    accuracy = ctx.A.mean(axis=1)
    a, b = accuracy[pairs[:, 0]], accuracy[pairs[:, 1]]
    return np.column_stack([(a + b) / 2.0, np.abs(a - b)])


def _pair_transfer(ctx: TaskContext, target: np.ndarray, name: str) -> TaskMetrics:
    """Fit a symmetric pair target inside training folds, score on held-out pairs."""
    rhos: list[float] = []
    null_rhos: list[float] = []
    n_scored = 0
    scored_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for train, test in ctx.model_folds():
        fit_pairs, score_pairs = _model_pairs(train), _model_pairs(test)
        y_fit = target[fit_pairs[:, 0], fit_pairs[:, 1]]
        y_score = target[score_pairs[:, 0], score_pairs[:, 1]]
        ok_fit, ok_score = np.isfinite(y_fit), np.isfinite(y_score)
        if ok_fit.sum() < MIN_LABELLED_FIT or ok_score.sum() < MIN_LABELLED_SCORE:
            logger.debug(
                "{}: fold has {} fit / {} scored measured pairs, skipped",
                name,
                int(ok_fit.sum()),
                int(ok_score.sum()),
            )
            continue
        fit_pairs, score_pairs = fit_pairs[ok_fit], score_pairs[ok_score]
        y_fit, y_score = y_fit[ok_fit], y_score[ok_score]
        predicted = _ridge_fit_predict(
            _pair_features(ctx.Z, fit_pairs), y_fit, _pair_features(ctx.Z, score_pairs)
        ).ravel()
        null = _ridge_fit_predict(
            _competence_pair_features(ctx, fit_pairs),
            y_fit,
            _competence_pair_features(ctx, score_pairs),
        ).ravel()
        rhos.append(_or_nan(spearman_rho, y_score, predicted))
        null_rhos.append(_or_nan(spearman_rho, y_score, null))
        n_scored += int(y_score.size)
        scored_folds.append((score_pairs, y_score, predicted))
    if not rhos:
        return _absent("spearman_mean", f"{name}: no fold carried enough measured pairs")

    def rho_mean(counts: np.ndarray) -> float:
        """The same fold-mean Spearman, over the pairs a model resample keeps."""
        return _mean_or_nan(
            [
                _or_nan(spearman_rho, truth[kept], predicted[kept])
                for pairs, truth, predicted in scored_folds
                if (kept := _resample_pairs(counts, pairs)).size >= MIN_LABELLED_SCORE
            ]
        )

    return {
        "spearman_mean": float(np.nanmean(rhos)),
        "spearman_competence_null": float(np.nanmean(null_rhos)),
        "n_scored_pairs": n_scored,
        "n_folds_scored": len(rhos),
        **_cluster_ci(rho_mean, float(np.nanmean(rhos)), ctx.n_models, UNIT_MODELS, ctx.seed),
    }


def _divergence_matrix(ctx: TaskContext) -> np.ndarray:
    """[M, M] ``1 - mean answer cosine`` over the questions both models answered."""
    rng = np.random.default_rng([ctx.seed, DIVERGENCE_QUESTIONS])
    size = min(DIVERGENCE_QUESTIONS, ctx.n_questions)
    cols = np.sort(rng.choice(ctx.n_questions, size=size, replace=False))
    mask = np.asarray(ctx.Ae_mask, dtype=bool)[:, cols]
    total = np.zeros((ctx.n_models, ctx.n_models))
    counts = np.zeros((ctx.n_models, ctx.n_models))
    for position, column in enumerate(cols):
        # a copy: normalising a view in place would rewrite the caller's answer bank
        vectors = np.array(ctx.Ae[:, column, :], dtype=np.float64)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), STD_FLOOR)
        present = mask[:, position].astype(np.float64)
        vectors *= present[:, None]
        total += vectors @ vectors.T
        counts += np.outer(present, present)
    with np.errstate(invalid="ignore", divide="ignore"):
        divergence = np.where(
            counts >= MIN_PAIR_COVERAGE, 1.0 - total / np.maximum(counts, 1.0), np.nan
        )
    np.fill_diagonal(divergence, np.nan)
    return divergence


def _axis_contrast(ctx: TaskContext, tokens: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """``(per-model tilt, the axes on the positive side)`` for a benchmark family."""
    matched = [i for i, name in enumerate(ctx.bench_names) if any(t in name for t in tokens)]
    rest = [i for i in range(ctx.n_benchmarks) if i not in matched]
    if len(matched) < MIN_CONTRAST_AXES or not rest:
        return np.empty(0), [ctx.bench_names[i] for i in matched]
    scores = ctx.benchmark_scores
    return scores[:, matched].mean(axis=1) - scores[:, rest].mean(axis=1), [
        ctx.bench_names[i] for i in matched
    ]


def _tilt_task(ctx: TaskContext, tokens: tuple[str, ...], what: str) -> TaskMetrics:
    """The shared body of the two capability-tilt readouts."""
    if not ctx.bench_names:
        return _absent("r2", f"{what}: the context carries no benchmark names to match on")
    tilt, axes = _axis_contrast(ctx, tokens)
    if tilt.size == 0:
        return _absent(
            "r2",
            f"{what}: {len(axes)} eval axes match {list(tokens)} and a composite needs "
            f"{MIN_CONTRAST_AXES} — there is no tilt to predict on this slice",
        )
    out = _scalar_transfer(ctx, tilt, np.arange(ctx.n_models))
    out["axes"] = DIAG_LIST_SEP.join(axes)
    out["n_axes"] = len(axes)
    out["n_reference_axes"] = ctx.n_benchmarks - len(axes)
    return out
