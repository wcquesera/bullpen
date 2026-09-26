"""Pooling battery runs into the tables the paper reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from bullpen.evaluation._utils import as_float_or_nan
from bullpen.evaluation.battery import CI_KEYS, DIAG_LIST_SEP, DIAGNOSTIC_PREFIX, TaskResult
from bullpen.evaluation.groups import GROUP_ORDER, require_group
from bullpen.evaluation.metrics import ALPHA, N_BOOT, bootstrap_ci
from bullpen.models.registry import INPUT_DIM, RUN_DIM, arm_dim

#: Tasks that are not independent evidence.
REDUNDANCY_CLUSTERS: dict[str, tuple[str, ...]] = {
    "cell_pred_structural": ("error_type", "item_information"),
    "cell_pred_decision": ("stage5a_routing", "portfolio_selection"),
    "ordering": ("ordinal_judge", "pairwise", "ranking"),
    "coldstart_forecasting": ("few_shot_coldstart", "correctness_forecasting"),
}

#: The competence leg.
COMPETENCE_TASK: str = "cross_benchmark"

#: The one-dimensional reference arm.
COMPETENCE_NULL_ARM: str = "null_leaderboard"

#: ``s(arm, task) - s(COMPETENCE_NULL_ARM, task)``, attached beside ``s`` on every per-cell
#: table.
COMPETENCE_DELTA_COLUMN: str = "competence_delta"

#: :func:`competence_residual` reference: subtract the per-task d=1 null's score.
PER_TASK_NULL: str = "per_task_null"

#: :func:`competence_residual` reference: the legacy regression on each arm's
#: :data:`COMPETENCE_TASK` score. Kept reachable, not kept as the default.
COMPETENCE_TASK_FIT: str = "competence_task_fit"

#: Anchored scores are clipped to this band.
SCORE_CLIP: float = 1.0

#: The anchored score before the clip, carried beside ``s`` on every cell.
UNCLIPPED_COLUMN: str = f"{DIAGNOSTIC_PREFIX}s_unclipped"

#: Whether the cell's raw primary metric sits further from chance than
#: ``EvaluateConfig.divergence_factor`` chance-to-ceiling spans.
DIVERGED_COLUMN: str = f"{DIAGNOSTIC_PREFIX}diverged"

#: A denominator ``best - chance`` below this is treated as no anchor at all.
MIN_ANCHOR_GAP: float = 1e-12

#: Fewest tasks a paired task-bootstrap will run on. With one task every resample
#: is that task, so the interval is a point and reads as certainty.
MIN_BOOTSTRAP_TASKS: int = 2

#: Fewest arms the competence fit needs before a residual is reported. Two points
#: define the line exactly and leave a residual of zero by construction.
MIN_FIT_ARMS: int = 4

#: Arms within this of the score floor are excluded from the competence FIT (they are still
#: scored against the resulting line).
FLOOR_MARGIN: float = 1e-3


@dataclass(frozen=True)
class RunResult:
    """One battery run: every task scored for one arm at one seed."""

    method: str
    seed: int
    results: Mapping[str, TaskResult]


@dataclass(frozen=True)
class Pooled:
    """A number pooled over runs, with what it was pooled over."""

    mean: float
    sd: float
    lo: float
    hi: float
    n: int
    n_missing: int


@dataclass(frozen=True)
class Cell:
    """One (method, task) cell: the headline number pooled over seeds, and the rest."""

    method: str
    task: str
    group: str
    primary: str
    lower_is_better: bool
    score: Pooled
    diagnostic: bool = False
    metrics: dict[str, Pooled] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    n_runs: int = 0
    n_error: int = 0
    errors: tuple[str, ...] = ()


#: The per-task interval key that is a LABEL rather than a number.
CI_UNIT_KEY: str = "resample_unit"

#: The numeric half of :data:`~bullpen.evaluation.battery.CI_KEYS`, pooled over seeds like any
#: other secondary metric.
CI_POOLED_KEYS: tuple[str, ...] = tuple(k for k in CI_KEYS if k != CI_UNIT_KEY)


def pool(values: Sequence[float], n_runs: int, seed: int = 0) -> Pooled:
    """Mean, sample sd and a bootstrap interval over runs, dropping non-finite values."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    missing = max(n_runs - n, 0)
    if n == 0:
        nan = float("nan")
        return Pooled(mean=nan, sd=nan, lo=nan, hi=nan, n=0, n_missing=missing)
    if n == 1:
        return Pooled(
            mean=float(finite[0]),
            sd=float("nan"),
            lo=float("nan"),
            hi=float("nan"),
            n=1,
            n_missing=missing,
        )
    ci = bootstrap_ci(finite, seed=seed)
    return Pooled(
        mean=ci.mean,
        sd=float(finite.std(ddof=1)),
        lo=ci.lo,
        hi=ci.hi,
        n=n,
        n_missing=missing,
    )


def _numeric(value: object) -> float:
    """A metric value as a float, or NaN if it is not a scalar number."""
    if isinstance(value, bool) or not isinstance(value, float | int):
        return float("nan")
    return float(value)


def pool_runs(runs: Iterable[RunResult], seed: int = 0) -> list[Cell]:
    """Pool a sweep into one :class:`Cell` per (method, task)."""
    by_method: dict[str, list[RunResult]] = {}
    for run in runs:
        by_method.setdefault(run.method, []).append(run)

    cells: list[Cell] = []
    for method, method_runs in by_method.items():
        seeds = [r.seed for r in method_runs]
        if len(set(seeds)) != len(seeds):
            logger.warning("method {} pools duplicate seeds {}", method, sorted(seeds))
        task_names: list[str] = []
        for run in method_runs:
            task_names.extend(t for t in run.results if t not in task_names)
        for name in task_names:
            present = [r.results[name] for r in method_runs if name in r.results]
            spec = present[0]
            scores = [r.score for r in present if r.error is None]
            metric_keys: list[str] = []
            for res in present:
                metric_keys.extend(k for k in res.metrics if k not in metric_keys)
            diag_keys = [k for k in metric_keys if k.startswith(DIAGNOSTIC_PREFIX)]
            # The unit label pools by agreement rather than by averaging, so it travels with the
            # diagnostics — but only when a run actually reported it.
            unit_keys = (CI_UNIT_KEY,) if CI_UNIT_KEY in metric_keys else ()
            metric_keys = [k for k in metric_keys if k not in diag_keys and k != CI_UNIT_KEY]
            cells.append(
                Cell(
                    method=method,
                    task=name,
                    group=spec.group,
                    primary=spec.primary,
                    lower_is_better=spec.lower_is_better,
                    diagnostic=spec.diagnostic,
                    score=pool(scores, len(present), seed=seed),
                    metrics={
                        key: pool(
                            [_numeric(r.metrics[key]) for r in present if key in r.metrics],
                            len(present),
                            seed=seed,
                        )
                        for key in metric_keys
                    },
                    diagnostics={
                        key: _one_value(
                            method, key, [r.metrics[key] for r in present if key in r.metrics]
                        )
                        for key in (*diag_keys, *unit_keys)
                    },
                    n_runs=len(present),
                    n_error=sum(r.error is not None for r in present),
                    errors=tuple(r.error for r in present if r.error is not None),
                )
            )
    return cells


_CELL_COLUMNS: list[str] = [
    "method",
    "task",
    "group",
    "primary",
    "lower_is_better",
    "diagnostic",
    "mean",
    "sd",
    "lo",
    "hi",
    "n_runs",
    "n_scored",
    "n_error",
    *CI_POOLED_KEYS,
    CI_UNIT_KEY,
]


def cell_table(cells: Iterable[Cell]) -> pd.DataFrame:
    """Cells as a long frame, one row per (method, task), in reading order."""
    rows = []
    task_rank: dict[str, int] = {}
    method_rank: dict[str, int] = {}
    diag_keys: list[str] = []
    for c in cells:
        require_group(c.group)
        task_rank.setdefault(c.task, len(task_rank))
        method_rank.setdefault(c.method, len(method_rank))
        diag_keys.extend(k for k in c.diagnostics if k not in diag_keys and k != CI_UNIT_KEY)
        rows.append(
            {
                "method": c.method,
                "task": c.task,
                "group": c.group,
                "primary": c.primary,
                "lower_is_better": c.lower_is_better,
                "diagnostic": c.diagnostic,
                "mean": c.score.mean,
                "sd": c.score.sd,
                "lo": c.score.lo,
                "hi": c.score.hi,
                "n_runs": c.n_runs,
                "n_scored": c.score.n,
                "n_error": c.n_error,
                **{
                    key: c.metrics[key].mean if key in c.metrics else float("nan")
                    for key in CI_POOLED_KEYS
                },
                CI_UNIT_KEY: c.diagnostics.get(CI_UNIT_KEY, float("nan")),
                **{k: v for k, v in c.diagnostics.items() if k != CI_UNIT_KEY},
            }
        )
    df = pd.DataFrame(rows, columns=[*_CELL_COLUMNS, *diag_keys])
    if df.empty:
        return df
    order = pd.DataFrame(
        {
            "g": df["group"].map(GROUP_ORDER.index),
            "t": df["task"].map(task_rank),
            "m": df["method"].map(method_rank),
        }
    )
    return df.loc[order.sort_values(["g", "t", "m"]).index].reset_index(drop=True)


# :data:`~bullpen.evaluation.battery.DIAGNOSTIC_PREFIX` is imported rather than defined here.

#: The interview budget K the run drew its interview at.
K_COLUMN: str = f"{DIAGNOSTIC_PREFIX}interview_k"

#: Rank of the fitted bank at 95% of its variance.
PCA95_COLUMN: str = f"{DIAGNOSTIC_PREFIX}pca95"

#: The width the run asked every ``run_dim`` arm for, read
#: once from the manifest and carried onto every row.
RUN_DIM_COLUMN: str = f"{DIAGNOSTIC_PREFIX}run_dim"

#: The width the arm's fitted bank CAME OUT AT — ``bank_shape[1]``, which is what every
#: downstream width-matched contrast is actually between.
EMB_DIM_COLUMN: str = f"{DIAGNOSTIC_PREFIX}emb_dim"

#: Joins an arm's input blocks into one cell.
BLOCK_SEP: str = DIAG_LIST_SEP


@dataclass(frozen=True)
class ArmDiagnostics:
    """How one fitted (arm, seed) was built, and the rank its bank came out at."""

    method: str
    seed: int
    #: interview budget K of the run the arm was fitted in
    interview_k: float
    #: columns the arm actually read, which is K only for an interview-block arm
    n_cols: float
    #: question blocks the arm consumed, fitting block first
    blocks: tuple[str, ...]
    #: components carrying 95% of the bank's variance
    pca95: float
    #: width the run asked for
    run_dim: float = float("nan")
    #: width the fitted bank CAME OUT AT, from ``bank_shape[1]``
    emb_dim: float = float("nan")


#: Columns of the unpooled per-(method, task, seed) frame, in reading order.
_UNPOOLED_COLUMNS: list[str] = [
    "method",
    "seed",
    "task",
    "group",
    "primary",
    "lower_is_better",
    "diagnostic",
    "score",
    "error",
    *CI_KEYS,
]


def unpooled_cell_table(runs: Iterable[RunResult]) -> pd.DataFrame:
    """Every run's cells, one row per (method, task, seed), nothing pooled."""
    rows = []
    task_rank: dict[str, int] = {}
    method_rank: dict[str, int] = {}
    for run in runs:
        method_rank.setdefault(run.method, len(method_rank))
        for name, res in run.results.items():
            require_group(res.group)
            task_rank.setdefault(name, len(task_rank))
            rows.append(
                {
                    "method": run.method,
                    "seed": run.seed,
                    "task": name,
                    "group": res.group,
                    "primary": res.primary,
                    "lower_is_better": res.lower_is_better,
                    "diagnostic": res.diagnostic,
                    "score": res.score,
                    "error": res.error,
                    **{k: res.metrics.get(k, float("nan")) for k in CI_KEYS},
                }
            )
    df = pd.DataFrame(rows, columns=_UNPOOLED_COLUMNS)
    if df.empty:
        return df
    order = pd.DataFrame(
        {
            "g": df["group"].map(GROUP_ORDER.index),
            "t": df["task"].map(task_rank),
            "m": df["method"].map(method_rank),
            "s": df["seed"],
        }
    )
    return df.loc[order.sort_values(["g", "t", "m", "s"]).index].reset_index(drop=True)


def realised_width(arm: Mapping[str, Any]) -> float:
    """The width one manifest arm row's fitted bank came out at, or ``NaN``."""
    shape = arm.get("bank_shape")
    if isinstance(shape, Sequence) and not isinstance(shape, str) and len(shape) >= 2:
        return as_float_or_nan(shape[1])
    return as_float_or_nan(arm.get("dim"))


def arm_diagnostics(manifest: Mapping[str, Any]) -> list[ArmDiagnostics]:
    """Per-(arm, seed) build diagnostics from a ``train.py`` manifest."""
    questions = manifest.get("questions") or {}
    interview_k = as_float_or_nan(questions.get("interview_k"))
    run_dim = as_float_or_nan(manifest.get("dim"))
    diags = [
        ArmDiagnostics(
            method=str(arm["arm"]),
            seed=int(arm["seed"]),
            interview_k=interview_k,
            n_cols=as_float_or_nan(arm.get("n_cols")),
            blocks=tuple(str(b) for b in arm.get("blocks") or ()),
            pca95=as_float_or_nan(arm.get("pca95")),
            run_dim=run_dim,
            emb_dim=realised_width(arm),
        )
        for arm in manifest.get("arms") or []
    ]
    if diags and not np.isfinite([d.pca95 for d in diags]).any():
        logger.warning(
            "no arm in this manifest recorded pca95; the diagnostic column will be empty"
        )
    return diags


_DIAGNOSTIC_COLUMNS: list[str] = [
    "method",
    "seed",
    K_COLUMN,
    RUN_DIM_COLUMN,
    EMB_DIM_COLUMN,
    f"{DIAGNOSTIC_PREFIX}n_cols",
    f"{DIAGNOSTIC_PREFIX}blocks",
    PCA95_COLUMN,
]


def diagnostic_table(diags: Iterable[ArmDiagnostics]) -> pd.DataFrame:
    """Diagnostics as a long frame, one row per (arm, seed), unpooled."""
    rows = [
        {
            "method": d.method,
            "seed": d.seed,
            K_COLUMN: d.interview_k,
            RUN_DIM_COLUMN: d.run_dim,
            EMB_DIM_COLUMN: d.emb_dim,
            f"{DIAGNOSTIC_PREFIX}n_cols": d.n_cols,
            f"{DIAGNOSTIC_PREFIX}blocks": BLOCK_SEP.join(d.blocks),
            PCA95_COLUMN: d.pca95,
        }
        for d in diags
    ]
    return pd.DataFrame(rows, columns=_DIAGNOSTIC_COLUMNS)


_POOLED_DIAGNOSTIC_COLUMNS: list[str] = [
    "method",
    K_COLUMN,
    RUN_DIM_COLUMN,
    EMB_DIM_COLUMN,
    f"{DIAGNOSTIC_PREFIX}n_cols",
    f"{DIAGNOSTIC_PREFIX}blocks",
    PCA95_COLUMN,
    f"{PCA95_COLUMN}_sd",
    f"{DIAGNOSTIC_PREFIX}n_seeds",
]


def pooled_diagnostics(diags: Iterable[ArmDiagnostics]) -> pd.DataFrame:
    """One diagnostic row per method, pooled over that method's seeds."""
    by_method: dict[str, list[ArmDiagnostics]] = {}
    for d in diags:
        by_method.setdefault(d.method, []).append(d)

    rows = []
    for method, group in by_method.items():
        ranks = np.asarray([d.pca95 for d in group], dtype=float)
        finite = ranks[np.isfinite(ranks)]
        rows.append(
            {
                "method": method,
                K_COLUMN: _one_value(method, "interview_k", [d.interview_k for d in group]),
                RUN_DIM_COLUMN: _one_value(method, "run_dim", [d.run_dim for d in group]),
                # can vary by seed for arms that land below their width; pooled as NaN then
                EMB_DIM_COLUMN: _one_value(method, "emb_dim", [d.emb_dim for d in group]),
                f"{DIAGNOSTIC_PREFIX}n_cols": _one_value(
                    method, "n_cols", [d.n_cols for d in group]
                ),
                f"{DIAGNOSTIC_PREFIX}blocks": _one_value(
                    method, "blocks", [BLOCK_SEP.join(d.blocks) for d in group]
                ),
                PCA95_COLUMN: float(finite.mean()) if finite.size else float("nan"),
                f"{PCA95_COLUMN}_sd": (
                    float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
                ),
                f"{DIAGNOSTIC_PREFIX}n_seeds": len({d.seed for d in group}),
            }
        )
    return pd.DataFrame(rows, columns=_POOLED_DIAGNOSTIC_COLUMNS)


#: Columns of ``width_fence.csv``, in reading order.
WIDTH_FENCE_COLUMNS: list[str] = [
    "method",
    "seed",
    "run_dim",
    "emb_dim",
    "delta",
    "declared_width",
    "declared_fixed",
    "reason",
]

#: ``reason`` for an arm the registry declares at a fixed width.
FENCE_DECLARED: str = "declared_fixed_width"

#: ``reason`` for an arm whose width is its input's (``text_mean768``).
FENCE_INPUT: str = "declared_input_width"

#: ``reason`` for an arm that asked for the run's width and landed BELOW it.
FENCE_SATURATED: str = "saturated_below_request"

#: ``reason`` for an arm that landed ABOVE the run's width. Not impossible and
#: not benign -- it would mean the width axis is not the axis it is labelled.
FENCE_WIDER: str = "wider_than_request"

#: ``reason`` for an arm whose manifest row recorded no bank, so the realised width is unknown.
FENCE_UNRECORDED: str = "width_not_recorded"


def _fence_reason(method: str, run_dim: float, emb_dim: float) -> str | None:
    """Why this (arm, seed) is on the fence, or ``None`` if it is not."""
    if not np.isfinite(run_dim):
        return None
    if not np.isfinite(emb_dim):
        return FENCE_UNRECORDED
    if emb_dim == run_dim:
        return None
    declared = arm_dim(method)
    if declared == INPUT_DIM:
        return FENCE_INPUT
    if declared != RUN_DIM:
        return FENCE_DECLARED
    return FENCE_SATURATED if emb_dim < run_dim else FENCE_WIDER


def width_fence_table(diags: Iterable[ArmDiagnostics]) -> pd.DataFrame:
    """Every (arm, seed) whose realised bank width missed the width its cell asked for."""
    rows = []
    for d in diags:
        reason = _fence_reason(d.method, d.run_dim, d.emb_dim)
        if reason is None:
            continue
        declared = arm_dim(d.method)
        rows.append(
            {
                "method": d.method,
                "seed": d.seed,
                "run_dim": d.run_dim,
                "emb_dim": d.emb_dim,
                "delta": d.emb_dim - d.run_dim,
                "declared_width": str(declared),
                "declared_fixed": declared != RUN_DIM,
                "reason": reason,
            }
        )
    frame = pd.DataFrame(rows, columns=WIDTH_FENCE_COLUMNS)
    for row in rows:
        logger.warning(
            "width fence: {} s{} asked for d={:g} and realised d={:g} ({}) — its "
            "numbers are not width-matched to the rest of this run",
            row["method"],
            row["seed"],
            row["run_dim"],
            row["emb_dim"],
            row["reason"],
        )
    return frame


def _one_value(method: str, field_name: str, values: Sequence[Any]) -> Any:
    """The single value ``values`` agree on, or ``NaN`` after saying they disagree."""
    kept = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    distinct = set(kept)
    if len(distinct) == 1:
        return distinct.pop()
    if len(distinct) > 1:
        logger.warning(
            "method {} reports {} seed-varying {} values {}; pooled as NaN",
            method,
            len(distinct),
            field_name,
            sorted(distinct),
        )
    return float("nan")


def attach_diagnostics(table: pd.DataFrame, diags: Iterable[ArmDiagnostics]) -> pd.DataFrame:
    """Append the pooled ``diag_*`` columns to a per-arm table, joined on ``method``."""
    pooled = pooled_diagnostics(diags)
    if "method" not in table.columns:
        raise ValueError("a diagnostic annotation needs a 'method' column to join on")
    merged = table.merge(pooled, on="method", how="left", sort=False)
    merged.index = table.index
    readouts = readout_columns(merged)
    return merged[[*readouts, *(c for c in merged.columns if c not in set(readouts))]]


def readout_columns(table: pd.DataFrame) -> list[str]:
    """Every column of ``table`` that is not a diagnostic."""
    return [c for c in table.columns if not str(c).startswith(DIAGNOSTIC_PREFIX)]


def scored_only(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` without its diagnostic rows — the tasks that are reported but do not vote."""
    if "diagnostic" not in df.columns:
        return df
    return df[~df["diagnostic"].fillna(False).astype(bool)]


def task_weights(tasks: pd.Series) -> pd.Series:
    """``1 / cluster_size`` for clustered tasks, ``1.0`` for everything else."""
    w: dict[str, float] = {}
    for members in REDUNDANCY_CLUSTERS.values():
        for t in members:
            w[t] = 1.0 / len(members)
    return tasks.map(lambda t: w.get(t, 1.0)).astype(float)


def _oriented_span(chance: pd.Series, lower_is_better: pd.Series) -> pd.Series:
    """``best - chance``, signed so a usable anchor is POSITIVE whatever the direction."""
    lib = lower_is_better.to_numpy(dtype=bool)
    best = pd.Series(np.where(lib, 0.0, 1.0), index=chance.index)
    return (best - chance) * np.where(lib, -1.0, 1.0)


def unclipped_score(mean: pd.Series, chance: pd.Series, lower_is_better: pd.Series) -> pd.Series:
    """``(metric - chance) / (best - chance)``, oriented so higher is better, unclipped."""
    best = np.where(lower_is_better.to_numpy(dtype=bool), 0.0, 1.0)
    denom = pd.Series(best, index=mean.index) - chance
    usable = _oriented_span(chance, lower_is_better) > MIN_ANCHOR_GAP
    s = (mean - chance) / denom.where(usable)
    return s.replace([np.inf, -np.inf], np.nan)


def competence_delta(
    df: pd.DataFrame,
    keys: Sequence[str] = ("task",),
    arm: str = COMPETENCE_NULL_ARM,
) -> pd.Series:
    """``s`` minus the one-dimensional null's ``s`` on the same task ."""
    nan = pd.Series(float("nan"), index=df.index, dtype=float)
    if df.empty or not {"method", "s"} <= set(df.columns):
        return nan
    ref = df[df["method"] == arm]
    if ref.empty:
        logger.warning(
            "competence_delta: reference arm {!r} is absent from this run, so every "
            "{} is NaN — the raw scores are NOT margins over the d=1 baseline",
            arm,
            COMPETENCE_DELTA_COLUMN,
        )
        return nan
    key_list = list(keys)
    baseline = ref.groupby(key_list, sort=False)["s"].mean()
    index = (
        pd.MultiIndex.from_arrays([df[k] for k in key_list])
        if len(key_list) > 1
        else pd.Index(df[key_list[0]])
    )
    return df["s"] - baseline.reindex(index).to_numpy()


#: A chance level keyed by task name (one anchor for every arm, the analytic levels) or by
#: ``(method, task)`` (an arm's OWN measured null).
ChanceLevels = Mapping[str, float] | Mapping[tuple[str, str], float]

#: The per-seed anchored score with the largest magnitude before the clip, signed.
WORST_SEED_COLUMN: str = f"{DIAGNOSTIC_PREFIX}s_unclipped_worst_seed"

#: Seeds whose own anchored score was finite and entered the cell's ``s``.
N_SEEDS_SCORED_COLUMN: str = f"{DIAGNOSTIC_PREFIX}n_seeds_anchored"


def _chance_for(df: pd.DataFrame, chance: Mapping[Any, float]) -> pd.Series:
    """The chance level of every row of ``df``: per-arm key first, then task key."""
    return pd.Series(
        [
            chance.get((m, t), chance.get(t, float("nan")))
            for m, t in zip(df["method"], df["task"], strict=True)
        ],
        index=df.index,
        dtype=float,
    )


def warn_no_headroom(df: pd.DataFrame) -> None:
    """Name every (method, task) whose chance level sits at or past the ceiling."""
    if df.empty or "chance" not in df.columns:
        return
    span = _oriented_span(df["chance"], df["lower_is_better"])
    hit = df[df["chance"].notna() & (span <= MIN_ANCHOR_GAP)]
    if hit.empty:
        return
    logger.warning(
        "{} cell(s) have a chance level at or past the metric's ceiling and are scored "
        "NaN, not sign-flipped: {}",
        len(hit),
        ", ".join(
            f"{r['method']}/{r['task']} (chance {r['chance']:.4g})"
            for _, r in hit.drop_duplicates(["method", "task"]).iterrows()
        ),
    )


def score_seeds(
    unpooled: pd.DataFrame, chance: ChanceLevels, factor: float | None = None
) -> pd.DataFrame:
    """Anchor and clip every (method, task, seed) row of :func:`unpooled_cell_table`."""
    df = unpooled.copy()
    df["chance"] = _chance_for(df, chance)
    unclipped = unclipped_score(df["score"], df["chance"], df["lower_is_better"])
    df[UNCLIPPED_COLUMN] = unclipped
    df["s"] = unclipped.clip(-SCORE_CLIP, SCORE_CLIP)
    df[COMPETENCE_DELTA_COLUMN] = competence_delta(df, keys=("task", "seed"))
    if factor is not None:
        df[DIVERGED_COLUMN] = (unclipped.abs() > factor).astype("boolean").mask(unclipped.isna())
    return df


def score_frame(
    table: pd.DataFrame, chance: ChanceLevels, seeds: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Attach ``chance``, ``s`` (anchored score), ``w`` (weight) and ``above_chance``."""
    df = table.copy()
    df["chance"] = _chance_for(df, chance)
    if seeds is None:
        unclipped = unclipped_score(df["mean"], df["chance"], df["lower_is_better"])
        df["s"] = unclipped.clip(-SCORE_CLIP, SCORE_CLIP)
    else:
        per_seed = score_seeds(seeds, chance)
        finite = per_seed[per_seed["s"].notna()]
        by_cell = finite.groupby(["method", "task"], sort=False)
        worst = finite.loc[
            finite[UNCLIPPED_COLUMN].abs().groupby([finite["method"], finite["task"]]).idxmax()
        ].set_index(["method", "task"])[UNCLIPPED_COLUMN]
        key = pd.MultiIndex.from_arrays([df["method"], df["task"]])
        df["s"] = by_cell["s"].mean().reindex(key).to_numpy()
        unclipped = pd.Series(
            by_cell[UNCLIPPED_COLUMN].mean().reindex(key).to_numpy(), index=df.index
        )
        df[WORST_SEED_COLUMN] = worst.reindex(key).to_numpy()
        df[N_SEEDS_SCORED_COLUMN] = (
            by_cell["s"].size().reindex(key).fillna(0).astype(int).to_numpy()
        )
    df["w"] = task_weights(df["task"])
    df["above_chance"] = (df["s"] > 0.0).astype("boolean").mask(df["s"].isna())
    df[COMPETENCE_DELTA_COLUMN] = competence_delta(df)
    df[UNCLIPPED_COLUMN] = unclipped
    warn_no_headroom(df)
    return df


def flag_divergence(table: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Add :data:`DIVERGED_COLUMN`: was this cell clipped from a divergent magnitude?"""
    df = table.copy()
    if UNCLIPPED_COLUMN not in df.columns:
        raise ValueError(
            f"a divergence flag needs {UNCLIPPED_COLUMN!r}; pass a frame from score_frame"
        )
    # the worst single seed when the frame was scored per seed: a mean over seeds
    # can sit inside the factor while one of them is five orders past it
    source = WORST_SEED_COLUMN if WORST_SEED_COLUMN in df.columns else UNCLIPPED_COLUMN
    unclipped = df[source].abs()
    df[DIVERGED_COLUMN] = (unclipped > factor).astype("boolean").mask(unclipped.isna())
    return df


_DIVERGED_COLUMNS: list[str] = ["method", "task", "primary", "mean", "sd", UNCLIPPED_COLUMN, "s"]


def diverged_cells(table: pd.DataFrame) -> pd.DataFrame:
    """The flagged cells of a :func:`flag_divergence` frame, worst magnitude first."""
    if DIVERGED_COLUMN not in table.columns:
        raise ValueError(f"{DIVERGED_COLUMN!r} is absent; call flag_divergence first")
    hit = table[table[DIVERGED_COLUMN].fillna(False).astype(bool)]
    cols = [c for c in _DIVERGED_COLUMNS if c in hit.columns]
    return hit[cols].reindex(hit[UNCLIPPED_COLUMN].abs().sort_values(ascending=False).index)


#: Label of the provisional group score over the beyond-capability tasks.
GROUP_H: str = "group_h"
GROUP_H_SCORE_TASKS: tuple[str, ...] = (
    "hedge_style",
    "refusal_rate",
    "length_profile",
    "rm_style",
    "rm_mean",
    "reasoning_style",
    "answer_divergence",
)

_GROUP_COLUMNS: list[str] = [
    "method",
    "group",
    "provisional",
    "score",
    "n_tasks",
    "eff_tasks",
    "score_common",
    "n_tasks_common",
    "n_above",
    "any_above",
    "n_dropped",
    "tasks_scored",
    "tasks_dropped",
    # Self-description, because two files named group_scores.csv ship with the same `score`
    # column and DIFFERENT weighting.
    "weighting",
]


def _drop_reason(row: Mapping[str, Any]) -> str:
    """Why a cell carries no anchored score, in the reader's terms."""
    chance = row.get("chance", 0.0)
    if bool(pd.isna(row.get("mean", 0.0))):
        return "primary metric never measured"
    if bool(pd.isna(chance)):
        return "no chance level, unanchored"
    lib = bool(row.get("lower_is_better", False))
    span = float(chance) if lib else 1.0 - float(chance)
    if span <= MIN_ANCHOR_GAP:
        return f"chance {chance:g} sits at or past the metric's ceiling, no headroom"
    return "anchored score is NaN"


def _group_rows(d: pd.DataFrame, group_of: pd.Series, provisional: bool) -> list[dict[str, Any]]:
    """One group-score row per (method, group) of ``d``, grouping by ``group_of``."""
    d = d.assign(_group=group_of.to_numpy())
    rows = []
    for g, in_group in d.groupby("_group", sort=True):
        # the tasks EVERY arm in this group scored: the only set on which two arms'
        # group numbers are the same quantity
        scored_by = in_group[in_group["s"].notna()].groupby("method")["task"].agg(set)
        common = set.intersection(*scored_by) if len(scored_by) else set()
        for m, grp in in_group.groupby("method", sort=True):
            kept = grp[grp["s"].notna()]
            dropped = grp[grp["s"].isna()]
            if kept.empty:
                continue
            if not dropped.empty:
                logger.warning(
                    "{} / {}: averaging {} of {} tasks — dropped {}",
                    m,
                    g,
                    len(kept),
                    len(grp),
                    ", ".join(f"{r['task']} ({_drop_reason(r)})" for _, r in dropped.iterrows()),
                )
            w = kept["w"].to_numpy()
            shared = kept[kept["task"].isin(common)]
            rows.append(
                {
                    "method": m,
                    "group": g,
                    "provisional": provisional,
                    "score": float(np.average(kept["s"], weights=w)),
                    "n_tasks": len(kept),
                    "eff_tasks": float(w.sum()),
                    "score_common": (
                        float(np.average(shared["s"], weights=shared["w"]))
                        if not shared.empty
                        else float("nan")
                    ),
                    "n_tasks_common": len(shared),
                    "n_above": int(kept["above_chance"].sum()),
                    "any_above": bool(kept["above_chance"].any()),
                    "n_dropped": len(dropped),
                    "tasks_scored": BLOCK_SEP.join(kept["task"]),
                    "tasks_dropped": BLOCK_SEP.join(dropped["task"]),
                }
            )
    return rows


def group_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean anchored score per (method, group), with the honesty columns."""
    d = scored_only(df)
    rows = _group_rows(d, d["group"], provisional=False)
    h = d[d["task"].isin(GROUP_H_SCORE_TASKS)]
    if not h.empty:
        rows.extend(_group_rows(h, pd.Series(GROUP_H, index=h.index), provisional=True))
    out = pd.DataFrame(rows, columns=_GROUP_COLUMNS)
    out["weighting"] = "redundancy"
    out = out.copy()
    out["shares_bank_with"] = ""  # no reported arm shares another arm's fitted bank
    return out


def _paired_panel(df: pd.DataFrame, group: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Tasks × methods panel of anchored scores, complete cases only, with weights."""
    d = scored_only(df).dropna(subset=["s"])
    if group == GROUP_H:
        d = d[d["task"].isin(GROUP_H_SCORE_TASKS)]
    elif group is not None:
        d = d[d["group"] == group]
    piv = d.pivot_table(index="task", columns="method", values="s")
    wts = d.drop_duplicates("task").set_index("task")["w"]
    piv = piv.dropna(axis=0, how="any")
    return piv, wts.reindex(piv.index).to_numpy(dtype=float)


def _weighted_draws(X: np.ndarray, w: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """``n_boot`` weighted column means of ``X`` under a shared row resample."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    draws = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = np.average(X[idx], axis=0, weights=w[idx])
    return draws


def _p_top1(draws: np.ndarray) -> np.ndarray:
    """Share of draws each column led, splitting a tie equally among the leaders."""
    leaders = draws == draws.max(axis=1, keepdims=True)
    return (leaders / leaders.sum(axis=1, keepdims=True)).mean(axis=0)


_BOOT_COLUMNS: list[str] = ["method", "score", "lcb", "ucb", "p_top1", "n_tasks"]


def paired_bootstrap(
    df: pd.DataFrame,
    group: str | None = None,
    methods: Sequence[str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired task-bootstrap over the anchored scores, per method."""
    if methods is None:
        cohorts = task_set_cohorts(df, group)
        if len(cohorts) > 1:
            logger.warning(
                "paired_bootstrap({}): {} arms on {} different task sets are paired over "
                "their common tasks only; pass methods= from task_set_cohorts",
                group,
                sum(len(c) for c in cohorts),
                len(cohorts),
            )
    d = df if methods is None else df[df["method"].isin(list(methods))]
    piv, w = _paired_panel(d, group)
    if piv.shape[0] < MIN_BOOTSTRAP_TASKS:
        logger.debug("paired_bootstrap: {} complete tasks, not enough to pair", piv.shape[0])
        return pd.DataFrame(columns=_BOOT_COLUMNS)

    X = piv.to_numpy()
    draws = _weighted_draws(X, w, n_boot, seed)
    return (
        pd.DataFrame(
            {
                "method": piv.columns,
                "score": np.average(X, axis=0, weights=w),
                "lcb": np.percentile(draws, 100 * ALPHA / 2, axis=0),
                "ucb": np.percentile(draws, 100 * (1 - ALPHA / 2), axis=0),
                "p_top1": _p_top1(draws),
                "n_tasks": X.shape[0],
            }
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def task_set_cohorts(df: pd.DataFrame, group: str | None = None) -> list[list[str]]:
    """Arms partitioned by the exact set of tasks they carry a finite ``s`` on."""
    d = scored_only(df).dropna(subset=["s"])
    if group == GROUP_H:
        d = d[d["task"].isin(GROUP_H_SCORE_TASKS)]
    elif group is not None:
        d = d[d["group"] == group]
    by_set: dict[frozenset[str], list[str]] = {}
    for method, grp in d.groupby("method", sort=True):
        by_set.setdefault(frozenset(grp["task"]), []).append(str(method))
    return sorted(by_set.values(), key=lambda arms: (-len(arms), arms))


def maximal_set(boot: pd.DataFrame) -> list[str]:
    """Arms not excluded by the leader's lower bound — the resolvable object."""
    if boot.empty:
        return []
    bar = boot["lcb"].max()
    return boot.loc[boot["ucb"] >= bar, "method"].tolist()


_RESIDUAL_COLUMNS: list[str] = ["method", "task", "group", "w", "s", "above_chance"]


def _residual_against_per_task_null(df: pd.DataFrame) -> pd.DataFrame:
    """:data:`PER_TASK_NULL`: every task's own d=1 reference subtracted, no fit."""
    d = scored_only(df).dropna(subset=["s"])
    if d.empty:
        return pd.DataFrame(columns=_RESIDUAL_COLUMNS)
    residual = competence_delta(d)
    return pd.DataFrame(
        {
            "method": d["method"],
            "task": d["task"],
            "group": d["group"],
            "w": d["w"],
            "s": residual,
            "above_chance": (residual > 0.0).astype("boolean").mask(residual.isna()),
        }
    ).reset_index(drop=True)


def competence_residual(df: pd.DataFrame, reference: str = PER_TASK_NULL) -> pd.DataFrame:
    """Per-task anchored scores with the competence axis taken out, across arms."""
    if reference == PER_TASK_NULL:
        return _residual_against_per_task_null(df)
    if reference != COMPETENCE_TASK_FIT:
        raise ValueError(
            f"unknown competence reference {reference!r}; "
            f"expected {PER_TASK_NULL!r} or {COMPETENCE_TASK_FIT!r}"
        )
    d = scored_only(df).dropna(subset=["s"])
    comp = d[d["task"] == COMPETENCE_TASK].set_index("method")["s"]
    out = []
    for task_name, grp in d.groupby("task", sort=False):
        if task_name == COMPETENCE_TASK:
            continue
        g = grp.set_index("method")
        x = comp.reindex(g.index)
        ok = x.notna() & g["s"].notna() & (x > -SCORE_CLIP + FLOOR_MARGIN)
        if ok.sum() < MIN_FIT_ARMS or x[ok].std() < MIN_ANCHOR_GAP:
            logger.debug("competence_residual: task {} has too flat a fit, skipped", task_name)
            continue
        slope, intercept = np.polyfit(x[ok], g.loc[ok, "s"], 1)
        residual = g["s"] - (intercept + slope * x)
        out.append(
            pd.DataFrame(
                {
                    "method": g.index,
                    "task": task_name,
                    "group": g["group"].to_numpy(),
                    "w": g["w"].to_numpy(),
                    "s": residual.to_numpy(),
                    "above_chance": (residual > 0.0).astype("boolean").mask(residual.isna()),
                }
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=_RESIDUAL_COLUMNS)
