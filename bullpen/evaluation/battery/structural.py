"""Structural tasks: what the bank knows about each model."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score

from bullpen.evaluation.battery.constants import (
    DECISION_THRESHOLD,
    MIN_BENCHMARKS_FOR_TRANSFER,
    MIN_FAMILIES,
    MIN_LABELLED_SCORE,
    NDCG_K,
    REASONING_AXIS_TOKENS,
    TRANSFER_PERMUTATIONS,
    UNIT_BENCHMARKS,
    UNIT_MODELS,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _absent,
    _divergence_matrix,
    _family_rows,
    _labelled_rows,
    _model_pairs,
    _oof_logistic,
    _pair_design,
    _pair_transfer,
    _permuted_null,
    _scalar_transfer,
    _tilt_task,
    _too_few,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.shared import (
    _cluster_ci,
    _clustered_question_ci,
    _mean_ci,
    _mean_or_nan,
    _or_nan,
    _resample,
    _resample_pairs,
    _ridge_fit_predict,
)
from bullpen.evaluation.groups import STRUCTURAL
from bullpen.evaluation.metrics import (
    column_auroc,
    ndcg_at_k,
    oof_ridge_predict,
    pool_mean_rows,
    row_auroc,
    spearman_rho,
)


# --------------------------------------------------------------------------- #
# structural tasks
# --------------------------------------------------------------------------- #
@task("score_regression", primary="r2", group=STRUCTURAL)
def score_regression(ctx: TaskContext) -> TaskMetrics:
    """Predict a model's mean accuracy on the scored block — the competence axis."""
    return _scalar_transfer(ctx, ctx.A.mean(axis=1), np.arange(ctx.n_models))


@task("pairwise", primary="auroc_mean", group=STRUCTURAL)
def pairwise(ctx: TaskContext) -> TaskMetrics:
    """Given two profiles, predict which model scores higher on each benchmark."""
    Y = ctx.benchmark_scores
    accuracies: list[float] = []
    aurocs: list[float] = []
    scored_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def fold_auroc(truth: np.ndarray, predicted: np.ndarray) -> float:
        """Mean over benchmarks of the pair-unit AUROC on one fold's scored pairs."""
        return _mean_or_nan([_or_nan(row_auroc, truth[:, b], predicted[:, b]) for b in ctx.scored])

    for train, test in ctx.model_folds():
        X_tr, Y_tr = _pair_design(ctx.Z, Y, train)
        X_te, Y_te = _pair_design(ctx.Z, Y, test)
        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            logger.debug("a fold formed no pairs; skipping it")
            continue
        P = _ridge_fit_predict(X_tr, Y_tr, X_te)
        accuracies.append(float(((P > DECISION_THRESHOLD) == (Y_te > DECISION_THRESHOLD)).mean()))
        aurocs.append(fold_auroc(Y_te, P))
        # _pair_design and _model_pairs enumerate the same combinations in the same
        # order, so row p of this block is the pair named by row p of that one
        scored_folds.append((_model_pairs(test), Y_te, P))
    if not accuracies:
        raise ValueError("no fold formed enough pairs to score; raise n_folds or add models")
    return {
        "auroc_mean": float(np.nanmean(aurocs)),
        "accuracy_mean": float(np.mean(accuracies)),
        "n_folds": len(accuracies),
        **_cluster_ci(
            lambda counts: _mean_or_nan(
                [
                    fold_auroc(truth[kept], predicted[kept])
                    for pairs, truth, predicted in scored_folds
                    if (kept := _resample_pairs(counts, pairs)).size >= MIN_LABELLED_SCORE
                ]
            ),
            float(np.nanmean(aurocs)),
            ctx.n_models,
            UNIT_MODELS,
            ctx.seed,
        ),
    }


@task("ranking", primary=f"ndcg@{NDCG_K}_mean", group=STRUCTURAL)
def ranking(ctx: TaskContext) -> TaskMetrics:
    """Rank models on a benchmark the head never trained on."""
    Y = ctx.benchmark_scores
    B = Y.shape[1]
    if B < MIN_BENCHMARKS_FOR_TRANSFER:
        raise ValueError(f"leave-one-benchmark-out needs at least {MIN_BENCHMARKS_FOR_TRANSFER}")

    def transfer(order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``([B] nDCG@k, [B] Spearman)`` of the fit reading ``ctx.Z`` in ``order``."""
        ndcgs: list[float] = []
        rhos: list[float] = []
        for b in ctx.scored:
            others = [x for x in range(B) if x != b]
            predicted = oof_ridge_predict(
                ctx.Z[order], Y[:, others].mean(axis=1), n_folds=ctx.n_folds, seed=ctx.seed
            )
            ndcgs.append(_or_nan(ndcg_at_k, Y[:, b], predicted, NDCG_K))
            rhos.append(_or_nan(spearman_rho, Y[:, b], predicted))
        return np.asarray(ndcgs), np.asarray(rhos)

    rows = np.arange(ctx.n_models)
    per_benchmark, rhos = transfer(rows)
    return {
        f"ndcg@{NDCG_K}_mean": float(np.nanmean(per_benchmark)),
        f"ndcg@{NDCG_K}_permuted_null": _permuted_null(
            rows, lambda o: float(np.nanmean(transfer(o)[0])), ctx.seed, TRANSFER_PERMUTATIONS
        ),
        "spearman_mean": float(np.nanmean(rhos)),
        "n_benchmarks": len(ctx.scored),
        **_mean_ci(per_benchmark, UNIT_BENCHMARKS, ctx.seed),
    }


@task("cross_benchmark", primary="spearman_mean", group=STRUCTURAL)
def cross_benchmark(ctx: TaskContext) -> TaskMetrics:
    """Predict accuracy on a held-out benchmark from the profile, benchmark by benchmark."""
    Y = ctx.benchmark_scores
    B = Y.shape[1]
    if B < MIN_BENCHMARKS_FOR_TRANSFER:
        raise ValueError(f"cross-benchmark transfer needs at least {MIN_BENCHMARKS_FOR_TRANSFER}")

    def transfer(order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``([B] Spearman, [B] RMSE)`` of the fit reading ``ctx.Z`` in ``order``."""
        rhos: list[float] = []
        errors: list[float] = []
        for b in ctx.scored:
            P = oof_ridge_predict(ctx.Z[order], Y[:, b], n_folds=ctx.n_folds, seed=ctx.seed)
            rhos.append(_or_nan(spearman_rho, Y[:, b], P))
            errors.append(float(np.sqrt(((P - Y[:, b]) ** 2).mean())))
        return np.asarray(rhos), np.asarray(errors)

    rows = np.arange(ctx.n_models)
    per_benchmark, errors = transfer(rows)
    return {
        "spearman_mean": float(np.nanmean(per_benchmark)),
        "spearman_permuted_null": _permuted_null(
            rows, lambda o: float(np.nanmean(transfer(o)[0])), ctx.seed, TRANSFER_PERMUTATIONS
        ),
        "rmse": float(np.mean(errors)),
        "n_benchmarks": len(ctx.scored),
        **_mean_ci(per_benchmark, UNIT_BENCHMARKS, ctx.seed),
    }


@task("error_type", primary="column_auroc", group=STRUCTURAL)
def error_type(ctx: TaskContext) -> TaskMetrics:
    """Which questions does this model get wrong — decoded from the bank alone."""
    per_question = column_auroc(ctx.decoded_cells, ctx.R)
    difficulty = column_auroc(pool_mean_rows(ctx.A, seed=ctx.seed), ctx.R)
    return {
        "column_auroc": float(np.nanmean(per_question)),
        "column_auroc_permuted_null": _mean_or_nan(
            [float(np.nanmean(column_auroc(P, ctx.R))) for P in ctx.permuted_decodes]
        ),
        "column_auroc_difficulty_baseline": float(np.nanmean(difficulty)),
        "n_questions": int(np.isfinite(per_question).sum()),
        **_clustered_question_ci(ctx, per_question),
    }


@task("answer_divergence", primary="spearman_mean", group=STRUCTURAL)
def answer_divergence(ctx: TaskContext) -> TaskMetrics:
    """Predict how differently two models WORD their answers."""
    if ctx.Ae is None:
        return _absent(
            "spearman_mean",
            "the slice carries no answer embeddings, and this task will not fall back "
            "to a bits-derived target and call it answer divergence",
        )
    return _pair_transfer(ctx, _divergence_matrix(ctx), "answer_divergence")


@task("family_classification", primary="macro_f1", group=STRUCTURAL)
def family_classification(ctx: TaskContext) -> TaskMetrics:
    """Closed-set: which publisher family is this held-out model, from the bank alone."""
    if not ctx.family:
        return _absent("macro_f1", "the context carries no family labels")
    rows, labels = _family_rows(ctx)
    classes = sorted(set(labels.tolist()))
    if len(classes) < MIN_FAMILIES or _too_few(rows):
        return _absent(
            "macro_f1",
            f"{len(classes)} families over {rows.size} labelled models is not a "
            "closed-set problem this pool can pose",
        )
    predicted, _ = _oof_logistic(ctx, np.asarray(ctx.family, dtype=object), rows)

    def macro_f1(truth: np.ndarray) -> float:
        return float(f1_score(truth, predicted, average="macro", labels=classes, zero_division=0))

    def macro_f1_over(counts: np.ndarray) -> float:
        """Macro-F1 on one resample of the labelled models, over the same class set."""
        kept = _resample(counts)
        return float(
            f1_score(
                labels[kept],
                predicted[kept],
                average="macro",
                labels=classes,
                zero_division=0,
            )
        )

    return {
        "macro_f1": macro_f1(labels),
        "macro_f1_permuted_null": _permuted_null(labels, macro_f1, ctx.seed),
        "accuracy": float((labels == predicted).mean()),
        "n_classes": len(classes),
        "n_labelled": int(rows.size),
        **_cluster_ci(macro_f1_over, macro_f1(labels), int(rows.size), UNIT_MODELS, ctx.seed),
    }


@task("base_vs_chat", primary="auroc", group=STRUCTURAL)
def base_vs_chat(ctx: TaskContext) -> TaskMetrics:
    """Binary: was this held-out model instruction-tuned, from the bank alone."""
    if ctx.is_chat is None:
        return _absent("auroc", "the context carries no instruction-tune flag")
    labels = np.asarray(ctx.is_chat, dtype=bool)
    if np.unique(labels).size < 2:
        return _absent("auroc", f"every one of {labels.size} models carries the same chat label")
    rows = np.arange(ctx.n_models)
    _, score = _oof_logistic(ctx, labels.astype(int), rows)
    y = labels.astype(float)
    return {
        "auroc": _or_nan(row_auroc, y, score),
        "n_chat": int(labels.sum()),
        "n_base": int((~labels).sum()),
        "n_models": ctx.n_models,
        **_cluster_ci(
            lambda counts: _or_nan(row_auroc, *(v[_resample(counts)] for v in (y, score))),
            _or_nan(row_auroc, y, score),
            ctx.n_models,
            UNIT_MODELS,
            ctx.seed,
        ),
    }


@task("model_size_regression", primary="r2", group=STRUCTURAL)
def model_size_regression(ctx: TaskContext) -> TaskMetrics:
    """Regress ``log10`` parameter count — the one target here that is not behavioural."""
    if ctx.params_b is None:
        return _absent("r2", "the context carries no parameter counts")
    params = np.asarray(ctx.params_b, dtype=np.float64)
    rows = _labelled_rows(np.isfinite(params) & (params > 0))
    if _too_few(rows):
        return _absent("r2", f"only {rows.size} of {ctx.n_models} models state a parameter count")
    out = _scalar_transfer(ctx, np.log10(np.where(params > 0, params, np.nan)), rows)
    out["n_unsized"] = int(ctx.n_models - rows.size)
    return out


@task("reasoning_style", primary="r2", group=STRUCTURAL)
def reasoning_style(ctx: TaskContext) -> TaskMetrics:
    """Predict reasoning TILT: the multi-step axes minus the rest, competence removed."""
    return _tilt_task(ctx, REASONING_AXIS_TOKENS, "reasoning_style")
