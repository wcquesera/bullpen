"""Score fitted arms through the decoding battery (library for ``eval.py``)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from bullpen.config import (
    EvaluateConfig,
    load_battery,
    load_benchmark_groups,
    load_evaluate,
    load_metrics,
)
from bullpen.data.enrichment import (
    chat_flags_for_models,
    dates_for_benchmarks,
    families_for_models,
    load_benchmark_dates,
    load_footprints,
    params_b_for_models,
    select_footprints,
)
from bullpen.data.group_h_labels import GroupHLabels
from bullpen.data.labels import load_labels
from bullpen.data.slice import DEFAULT_ANSWERS, Slice, load_slice
from bullpen.data.text_traits import TextTraits
from bullpen.evaluation import battery
from bullpen.evaluation.aggregate import (
    GROUP_H,
    Cell,
    RunResult,
    arm_diagnostics,
    attach_diagnostics,
    cell_table,
    competence_residual,
    diagnostic_table,
    diverged_cells,
    flag_divergence,
    group_scores,
    maximal_set,
    paired_bootstrap,
    pool_runs,
    score_frame,
    score_seeds,
    task_set_cohorts,
    unpooled_cell_table,
    width_fence_table,
)
from bullpen.evaluation.battery import (
    BATTERY,
    EXTERNAL_TABLE_BY_TASK,
    GROUP_H_TASKS,
    MIN_BENCHMARKS_FOR_TRANSFER,
    TRAIT_TASKS,
    TaskContext,
    run_battery,
)
from bullpen.evaluation.battery.constants import BENCHMARK_HOLDOUT_TASKS
from bullpen.evaluation.groups import GROUP_ORDER, select_tasks
from bullpen.evaluation.metrics import LAMBDA_GRID
from bullpen.models.base import Encoder
from bullpen.training import (
    MANIFEST_NAME,
    Artifact,
    config_digest,
    file_digest,
    git_sha,
    load_artifact,
)

#: The tables written on every run.
CELLS_CSV = "cells.csv"
#: The per-seed record ``cells.csv`` is pooled from.
UNPOOLED_CSV = "cells_unpooled.csv"
GROUPS_CSV = "group_scores.csv"
BOOTSTRAP_CSV = "bootstrap.csv"
RESIDUAL_CSV = "competence_residual.csv"
RUNS_JSON = "runs.json"

#: Per-(arm, seed) build diagnostics.
DIAGNOSTICS_CSV = "diagnostics.csv"

#: The width fence: one row per (arm, seed) whose fitted bank missed the width its cell
#: asked for -- and written on EVERY run, empty header included, because "checked and nothing
#: fired" and "never checked" are different facts and a reader must be able to tell them apart
#: from the directory alone.
WIDTH_FENCE_CSV = "width_fence.csv"

#: ``group`` value for the bootstrap row computed over every task at once.
ALL_GROUPS = "all"

#: Columns of ``bootstrap.csv``. ``cohort`` numbers the task-set cohort a row was
#: paired within; ``provisional`` is True on the provisional-group rows.
BOOTSTRAP_COLUMNS: list[str] = [
    "method",
    "score",
    "lcb",
    "ucb",
    "p_top1",
    "n_tasks",
    "group",
    "cohort",
    "provisional",
]


@dataclass
class HeldOutArm:
    """A fitted arm re-expressed over the models it was never shown."""

    inner: Encoder
    #: global slice row of each local row, in local order
    rows: np.ndarray
    #: [n_rows, d] bank over the held-out models, one refit each
    X: np.ndarray

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def cols(self) -> np.ndarray | None:
        return self.inner.cols

    @property
    def uses_probe(self) -> bool:
        return self.inner.uses_probe

    @property
    def has_decoder(self) -> bool:
        return self.inner.has_decoder

    @property
    def input_width(self) -> int:
        return self.inner.input_width

    def metric_weights(self) -> np.ndarray | None:
        return self.inner.metric_weights()

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Place a held-out model from probe cells, under its GLOBAL row index."""
        return self.inner.fold_in(bits, cols, int(self.rows[row]))

    def predict(self, theta: np.ndarray, cols: np.ndarray) -> np.ndarray | None:
        return self.inner.predict(theta, cols)


@dataclass
class Skip:
    """Something that was not scored, and why."""

    what: str
    reason: str


@dataclass
class Evaluation:
    """One evaluation run: the scored runs, and everything it refused to score."""

    runs: list[RunResult] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)


def read_manifest(artifact_dir: Path) -> dict[str, Any]:
    """Read the training run's manifest, or explain that the directory is not one."""
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {artifact_dir}. Point --in at a directory written by train.py."
        )
    return json.loads(path.read_text())


def eval_columns(manifest: dict[str, Any]) -> np.ndarray:
    """The scoring block, read from the training manifest."""
    try:
        cols = np.asarray(manifest["questions"]["eval"], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"{MANIFEST_NAME} carries no eval question block ({exc}). Point --in at a "
            f"directory written by this version of train.py."
        ) from None
    if cols.size == 0:
        raise ValueError(f"{MANIFEST_NAME} records an empty eval block; nothing to score on")
    return cols


def input_block(sl: Slice, artifact: Artifact, rows: np.ndarray) -> np.ndarray:
    """[n_rows, W] the held-out models' correctness on the arm's OWN columns."""
    return sl.A[np.ix_(np.asarray(rows, dtype=int), np.asarray(artifact.cols, dtype=int))]


def hold_out(artifact: Artifact, A_in: np.ndarray, rows: np.ndarray) -> HeldOutArm:
    """Place every held-out model with one fitted arm, returning the adapter."""
    cols = artifact.cols
    n_rows = len(rows)
    bank = np.vstack([artifact.encoder.refit(A_in[i], cols, int(rows[i])) for i in range(n_rows)])
    if bank.shape[0] != n_rows:
        raise RuntimeError(f"{artifact.arm}: placed {bank.shape[0]} of {n_rows} held-out models")
    return HeldOutArm(inner=artifact.encoder, rows=np.asarray(rows, dtype=int), X=bank)


def _analysis_seeds() -> dict[str, object]:
    """The seeds and constants that move a published interval but not a point."""
    cfg = load_evaluate()
    return {
        "aggregate_seed": cfg.aggregate_seed,
        "divergence_factor": cfg.divergence_factor,
    }


def _check_placement_identity(arm: HeldOutArm, sl: Slice, sub: Slice) -> None:
    """``arm.X[i]`` is the model ``sub.model_ids[i]`` names, by NAME and not length."""
    placed = [sl.model_ids[int(r)] for r in arm.rows]
    if placed != list(sub.model_ids):
        first = next(
            (i for i, (a, b) in enumerate(zip(placed, sub.model_ids, strict=False)) if a != b),
            min(len(placed), len(sub.model_ids)),
        )
        raise RuntimeError(
            f"placement identity broken: arm row {first} holds "
            f"{placed[first] if first < len(placed) else '<missing>'!r} but the scored "
            f"slice calls it "
            f"{sub.model_ids[first] if first < len(sub.model_ids) else '<missing>'!r}. "
            f"Every per-model target would be attributed to the wrong model."
        )


def _coverage_skips(
    tasks: Iterable[str],
    covered: int,
    n_models: int,
    floor: int,
    what: str,
    build: str,
) -> dict[str, str]:
    """``{task: reason}`` for a label table too thin to score any arm on."""
    if covered >= floor:
        return {}
    return dict.fromkeys(
        tasks,
        f"{what} cover {covered} of the {n_models} held-out models, below the "
        f"{floor}-model floor in config/evaluate.yaml ({build})",
    )


def runnable_tasks(
    sub: Slice,
    selected: list[str],
    external: Mapping[str, np.ndarray],
    min_external_coverage: int,
    traits: TextTraits | None = None,
    group_h: GroupHLabels | None = None,
) -> tuple[list[str], list[Skip]]:
    """Split a task selection into the tasks this slice supports and the rest."""
    has_vram = sub.vram_gb is not None and bool(np.isfinite(sub.vram_gb).any())
    reasons: dict[str, str] = {}
    for name, table in EXTERNAL_TABLE_BY_TASK.items():
        column = external.get(table)
        covered = 0 if column is None else int(np.isfinite(column).sum())
        if covered < min_external_coverage:
            reasons[name] = (
                f"{table} scores {covered} of the {sub.n_models} held-out models, below "
                f"the {min_external_coverage}-model floor in config/evaluate.yaml; a "
                f"leave-one-out ridge on that many rows measures its own shrinkage"
            )
    reasons.update(
        _coverage_skips(
            TRAIT_TASKS,
            0 if traits is None else int(traits.covered.sum()),
            sub.n_models,
            min_external_coverage,
            what="text_traits.npz profiles",
            build="text_traits.npz",
        )
    )
    reasons.update(
        _coverage_skips(
            GROUP_H_TASKS,
            0 if group_h is None else int(group_h.covered().sum()),
            sub.n_models,
            min_external_coverage,
            what="group_h_labels.npz reward rows",
            build="group_h_labels.npz",
        )
    )
    if not has_vram:
        reasons["knapsack"] = (
            "vram_gb carries no finite value in this slice, and the portfolio "
            "task will not invent model weights to solve its own knapsack"
        )
    if sub.n_benchmarks < MIN_BENCHMARKS_FOR_TRANSFER:
        for name in ("ranking", "cross_benchmark"):
            reasons[name] = (
                f"leave-one-benchmark-out needs {MIN_BENCHMARKS_FOR_TRANSFER} "
                f"benchmarks, the held-out block spans {sub.n_benchmarks}"
            )
    keep = [t for t in selected if t not in reasons]
    skipped = [Skip(what=t, reason=reasons[t]) for t in selected if t in reasons]
    for skip in skipped:
        logger.warning(f"skipping task {skip.what}: {skip.reason}")
    if not keep:
        raise ValueError(f"every selected task was skipped: {[s.what for s in skipped]}")
    return keep, skipped


@dataclass(frozen=True)
class ModelMetadata:
    """The per-model labels the metadata readouts need, on the slice's model axis."""

    family: list[str]
    params_b: np.ndarray
    is_chat: np.ndarray


def model_metadata(model_ids: list[str]) -> ModelMetadata:
    """Look up family, size and the lexical chat flag for one model axis."""
    footprints = select_footprints(model_ids, load_footprints().values())
    logger.info(f"footprints resolved for {len(footprints)}/{len(model_ids)} models")
    return ModelMetadata(
        family=families_for_models(model_ids, footprints),
        params_b=params_b_for_models(model_ids, footprints),
        is_chat=chat_flags_for_models(model_ids),
    )


def score_arms(
    artifacts_or_paths: list[Path] | list[Artifact],
    sl: Slice,
    sub: Slice,
    cols: np.ndarray,
    rows: np.ndarray,
    tasks: list[str],
    n_folds: int,
    external: Mapping[str, np.ndarray],
    traits: TextTraits | None,
    group_h: GroupHLabels | None,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    scored_benchmarks: np.ndarray | None = None,
) -> Evaluation:
    """Run the battery over the held-out block, once per (arm, seed) artifact."""
    vram = sub.vram_gb if sub.vram_gb is not None and np.isfinite(sub.vram_gb).any() else None
    meta = model_metadata(sub.model_ids)
    bench_dates = dates_for_benchmarks(sub.bench_names, load_benchmark_dates())
    n_items = len(artifacts_or_paths)
    out = Evaluation(tasks=tasks)
    for i, item in enumerate(artifacts_or_paths):
        if isinstance(item, Artifact):
            artifact = item
            owns_artifact = False
        else:
            artifact = load_artifact(item, sl)
            owns_artifact = True
        logger.info(f"loaded artifact {i + 1}/{n_items}: {artifact.arm} s{artifact.seed}")
        leaked = np.intersect1d(np.asarray(artifact.cols, dtype=int), cols)
        if leaked.size:
            raise ValueError(
                f"{artifact.arm} s{artifact.seed} was fitted on {leaked.size} of the "
                f"{cols.size} evaluation columns — that arm cannot be scored on this block"
            )
        try:
            A_in = input_block(sl, artifact, rows)
            arm = hold_out(artifact, A_in, rows)
            _check_placement_identity(arm, sl, sub)
        except Exception as exc:  # noqa: BLE001  (one arm must not sink the sweep)
            logger.exception(f"{artifact.arm} s{artifact.seed}: could not place held-out models")
            out.skipped.append(
                Skip(
                    what=f"{artifact.arm} s{artifact.seed}",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            if owns_artifact:
                del artifact
            continue
        ctx = TaskContext(
            encoder=arm,
            A=sub.A,
            R=sub.R,
            bench=sub.bench,
            bench_names=sub.bench_names,
            vram_gb=vram,
            seed=artifact.seed,
            n_folds=n_folds,
            lambdas=lambdas,
            cols=np.asarray(cols, dtype=int),
            A_in=A_in,
            in_cols=np.asarray(artifact.cols, dtype=int),
            model_ids=sub.model_ids,
            family=meta.family,
            params_b=meta.params_b,
            is_chat=meta.is_chat,
            release_date=sub.release_date,
            bench_dates=bench_dates,
            Ae=sub.Ae,
            Ae_mask=sub.Ae_mask,
            # the question bridge rides in answers.npz on the same question axis the
            # slice was cut on, so the item-side covariate needs no second loader
            Qe=sub.Qe,
            external=external,
            traits=None if traits is None else traits.traits,
            trait_names=() if traits is None else traits.trait_names,
            group_h=group_h,
            scored_benchmarks=scored_benchmarks,
        )
        logger.info(
            f"scoring {artifact.arm} s{artifact.seed} on {len(tasks)} task(s): "
            f"{A_in.shape[1]}-column {'+'.join(artifact.blocks)} input, "
            f"{sub.n_questions}-column eval block"
        )
        results = run_battery(ctx, names=tasks)
        out.runs.append(RunResult(method=artifact.arm, seed=artifact.seed, results=results))
        if owns_artifact:
            del artifact
        del arm, ctx, A_in
    return out


def chance_levels(cells: list[Cell], cfg: EvaluateConfig) -> dict[Any, float]:
    """The chance level of every scored cell, from the config and each arm's OWN null."""
    levels: dict[Any, float] = dict(cfg.chance)
    for task, key in cfg.chance_from_metric.items():
        task_cells = [c for c in cells if c.task == task]
        missing: list[str] = []
        measured: list[float] = []
        for c in task_cells:
            value = c.metrics[key].mean if key in c.metrics else float("nan")
            if np.isfinite(value):
                levels[(c.method, task)] = float(value)
                measured.append(float(value))
            else:
                missing.append(c.method)
        if not task_cells:
            continue
        if missing:
            logger.warning(
                f"{task}: config/evaluate.yaml anchors it on each arm's measured {key!r}, "
                f"which {len(missing)} arm(s) did not report — left unanchored for "
                f"{sorted(missing)}"
            )
        if measured:
            logger.info(
                f"{task}: chance = each arm's own measured {key} "
                f"(range across arms {min(measured):.4f}–{max(measured):.4f})"
            )
    scored = {c.task for c in cells}
    undeclared = sorted(scored - cfg.declared())
    if undeclared:
        logger.warning(
            f"task(s) {undeclared} have no chance level in config/evaluate.yaml and "
            f"are reported unanchored; declare them or list them under `unanchored`"
        )
    return {k: v for k, v in levels.items() if (k[1] if isinstance(k, tuple) else k) in scored}


def bootstrap_table(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """The paired task-bootstrap over every task, then within each group."""
    frames = []
    for group in (None, *GROUP_ORDER, GROUP_H):
        cohorts = task_set_cohorts(df, group)
        if len(cohorts) > 1:
            logger.info(
                f"bootstrap {group or ALL_GROUPS}: {len(cohorts)} task-set cohorts "
                f"{[len(c) for c in cohorts]} arms — each is paired only within itself"
            )
        for cohort_id, methods in enumerate(cohorts):
            boot = paired_bootstrap(df, group=group, methods=methods, n_boot=n_boot, seed=seed)
            if boot.empty:
                logger.debug(f"no paired bootstrap for {group or ALL_GROUPS} cohort {cohort_id}")
                continue
            frames.append(
                boot.assign(
                    group=group or ALL_GROUPS, cohort=cohort_id, provisional=group == GROUP_H
                )
            )
    if not frames:
        logger.warning("no group had enough complete tasks to pair; bootstrap.csv is empty")
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def runs_payload(evaluation: Evaluation) -> dict[str, Any]:
    """Every metric every run reported, keyed by ``arm__s<seed>``."""
    return {
        f"{run.method}__s{run.seed}": {
            task: {
                "group": res.group,
                "primary": res.primary,
                "score": res.score,
                "error": res.error,
                "metrics": res.metrics,
            }
            for task, res in run.results.items()
        }
        for run in evaluation.runs
    }


def write_tables(
    evaluation: Evaluation,
    out: Path,
    cfg: EvaluateConfig,
    n_boot: int,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Pool the runs, write every table, and return the scored cell frame."""
    out.mkdir(parents=True, exist_ok=True)
    cells = pool_runs(evaluation.runs, seed=cfg.aggregate_seed)
    chance = chance_levels(cells, cfg)
    unpooled = unpooled_cell_table(evaluation.runs)
    # anchored and clipped per seed, then averaged (score_frame's docstring)
    scored = flag_divergence(
        score_frame(cell_table(cells), chance, seeds=unpooled), cfg.divergence_factor
    )
    diags = arm_diagnostics(manifest)

    attach_diagnostics(scored, diags).to_csv(out / CELLS_CSV, index=False)
    score_seeds(unpooled, chance, cfg.divergence_factor).to_csv(out / UNPOOLED_CSV, index=False)
    diagnostic_table(diags).to_csv(out / DIAGNOSTICS_CSV, index=False)
    fence = width_fence_table(diags)
    fence.to_csv(out / WIDTH_FENCE_CSV, index=False)
    if fence.empty:
        logger.info(f"width fence: every arm reached its cell's width -> empty {WIDTH_FENCE_CSV}")
    else:
        logger.warning(
            f"width fence: {len(fence)} (arm, seed) row(s) missed their cell's width; "
            f"see {WIDTH_FENCE_CSV}. Those arms are not width-matched to their floor"
        )
    group_scores(scored).to_csv(out / GROUPS_CSV, index=False)
    if n_boot > 0:
        boot = bootstrap_table(scored, n_boot, cfg.aggregate_seed)
    else:
        # header only, so a reader sees the table was skipped rather than lost
        logger.warning(f"bootstrap skipped (n_boot={n_boot}); {BOOTSTRAP_CSV} is header-only")
        boot = pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    boot.to_csv(out / BOOTSTRAP_CSV, index=False)
    competence_residual(scored).to_csv(out / RESIDUAL_CSV, index=False)
    (out / RUNS_JSON).write_text(json.dumps(runs_payload(evaluation), indent=2, default=str) + "\n")
    logger.info(
        f"wrote {CELLS_CSV}, {UNPOOLED_CSV}, {DIAGNOSTICS_CSV}, {WIDTH_FENCE_CSV}, "
        f"{GROUPS_CSV}, {BOOTSTRAP_CSV}, {RESIDUAL_CSV} -> {out}"
    )
    return scored


def write_manifest(
    evaluation: Evaluation,
    out: Path,
    manifest: dict[str, Any],
    artifact_dir: Path,
    slice_path: Path,
    total_seconds: float,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    speed: dict[str, Any] | None = None,
) -> Path:
    """Record what was scored, against which fits, with which ridge grid, and what was refused."""
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "config_sha256": config_digest(),
        "slice_path": str(slice_path),
        "slice_sha256": file_digest(slice_path),
        "artifact_dir": str(artifact_dir),
        "train_manifest": {
            key: manifest.get(key)
            for key in ("created_at", "git_sha", "config_sha256", "slice_sha256", "dim", "seeds")
        },
        "split_source": manifest.get("split", {}).get("source"),
        "n_test": manifest.get("split", {}).get("n_test"),
        "eval_block": {
            key: manifest.get("questions", {}).get(key)
            for key in ("block_seed", "n_eval", "n_tune", "n_pool", "n_interview")
        },
        "arm_widths": {
            f"{a['arm']}__s{a['seed']}": {"blocks": a.get("blocks"), "n_cols": a.get("n_cols")}
            for a in manifest.get("arms", [])
        },
        "lambda_grid": list(lambdas),
        "speed": speed or {},
        # The analysis seeds, recorded because they MOVE PUBLISHED NUMBERS and were previously
        # recoverable only by digesting the whole config.
        "analysis_seeds": _analysis_seeds(),
        # The BLAS thread count, because it is not inert.
        "threads": {
            var: os.environ.get(var)
            for var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "tasks": evaluation.tasks,
        "runs": [f"{r.method}__s{r.seed}" for r in evaluation.runs],
        "skipped": [{"what": s.what, "reason": s.reason} for s in evaluation.skipped],
        "total_seconds": round(total_seconds, 3),
    }
    path = out / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"wrote {path}")
    return path


def log_divergence(scored: pd.DataFrame) -> None:
    """Warn, loudly and by name, about every cell the clip saturated from a divergence."""
    hit = diverged_cells(scored)
    if hit.empty:
        logger.info("no cell's raw primary metric diverged past the configured factor")
        return
    logger.warning(
        f"{len(hit)} cell(s) were clipped from a DIVERGENT raw metric — their group "
        f"scores understate the failure by orders of magnitude:\n" + hit.to_string(index=False)
    )
    logger.warning(
        "diverged (method, task): "
        + ", ".join(f"({r['method']}, {r['task']})" for _, r in hit.iterrows())
    )


def log_summary(scored: pd.DataFrame, out: Path) -> None:
    """Log the headline table and the resolvable answer to "which arm wins"."""
    pivot = scored.pivot_table(index="method", columns="task", values="mean", sort=False)
    logger.info("scored cells (task primary metric, pooled over seeds):\n" + pivot.to_string())
    log_divergence(scored)
    groups = pd.read_csv(out / GROUPS_CSV)
    if groups.empty:
        logger.warning("no anchored task survived; there is no group score to report")
        return
    logger.info("group scores (anchored, redundancy-weighted):\n" + groups.to_string(index=False))
    boot = pd.read_csv(out / BOOTSTRAP_CSV)
    overall = boot[boot["group"] == ALL_GROUPS] if not boot.empty else boot
    if "cohort" in overall.columns:
        # a maximal set is only defined among arms paired on the same tasks
        overall = overall[overall["cohort"] == 0]
    if not overall.empty:
        logger.info(
            "maximal set over every task, within the largest task-set cohort — the arms "
            "this battery cannot separate: " + ", ".join(maximal_set(overall))
        )


class SliceMismatch(ValueError):
    """The slice being scored is not the slice the arms were fitted on."""


def check_slice_digest(slice_path: Path, manifest: Mapping[str, Any]) -> None:
    """Refuse to score arms against a slice whose hash differs from the training manifest's."""
    digest = file_digest(slice_path)
    expected = manifest.get("slice_sha256")
    if digest != expected:
        raise SliceMismatch(
            f"{slice_path.name} hashes to {digest[:12]} but the arms were fitted "
            f"against {str(expected)[:12]} — these are different cuts; refusing to score"
        )


def set_knapsack_draws(n: int) -> None:
    """Point the portfolio task at ``n`` knapsack draws."""
    battery.KNAPSACK_DRAWS = battery.downstream.KNAPSACK_DRAWS = n


def evaluate_run(
    fits: Path,
    out: Path,
    tasks: list[str] | None = None,
    battery_config: Path | None = None,
    benchmark_groups: Path | None = None,
    bootstrap: bool = True,
) -> pd.DataFrame | None:
    """Score one training directory into ``out``; ``None`` when nothing could be scored."""
    cfg = load_evaluate()
    battery_cfg = load_battery(battery_config)
    set_knapsack_draws(battery_cfg.knapsack_draws)
    manifest = read_manifest(fits)

    slice_path = Path(manifest["slice_path"])
    answers = Path(manifest.get("answers_path") or DEFAULT_ANSWERS)
    sl = load_slice(slice_path, answers, view=manifest.get("view"))
    check_slice_digest(slice_path, manifest)

    artifact_paths = [Path(fits) / m["file"] for m in manifest["arms"]]
    if not artifact_paths:
        logger.error(f"{fits} holds no fitted arm; run train.py first")
        return None
    cols = eval_columns(manifest)
    group = manifest.get("hold_out_group")
    scored_benchmarks = None
    if group is not None:
        # leave-one-group-out: the arm never saw a question of the group, so all of
        # them are scored beside the eval block, and only the transfer tasks read them
        cols = np.union1d(cols, np.asarray(manifest["held_out_columns"], dtype=int))
        if tasks is None:
            tasks = list(BENCHMARK_HOLDOUT_TASKS)

    rows = np.array(manifest["split"]["test"], dtype=int)
    sub = sl.subset(models=rows, questions=cols)
    if group is not None:
        members = load_benchmark_groups(sl.bench_names, benchmark_groups)[group]
        scored_benchmarks = np.array([sub.bench_names.index(b) for b in members])
        logger.info(f"hold-out group {group}: scoring {list(members)} only")
    logger.info(
        f"held-out block: {sub.summary()} on the {cols.size}-column eval block "
        f"(split {manifest['split']['source']})"
    )

    external, traits, group_h = load_labels(sub.model_ids, sl.labels_root)
    selected = select_tasks(BATTERY.values(), names=tasks, groups=None)
    tasks, task_skips = runnable_tasks(
        sub, selected, external, cfg.min_external_coverage, traits, group_h
    )
    logger.info(f"scoring {len(artifact_paths)} fitted arm-run(s) from {fits}")

    started = time.perf_counter()
    evaluation = score_arms(
        artifact_paths,
        sl,
        sub,
        cols,
        rows,
        tasks,
        battery_cfg.n_folds,
        external,
        traits,
        group_h,
        scored_benchmarks=scored_benchmarks,
    )
    evaluation.skipped.extend(task_skips)
    if not evaluation.runs:
        logger.error("no arm could be scored; no table was written")
        return None

    n_boot = load_metrics().n_boot if bootstrap else 0
    scored = write_tables(evaluation, out, cfg, n_boot=n_boot, manifest=manifest)
    elapsed = time.perf_counter() - started
    speed = {
        "battery_config": str(battery_config or "config/battery.yaml"),
        "n_folds": battery_cfg.n_folds,
        "knapsack_draws": battery.KNAPSACK_DRAWS,
        "transfer_permutations": battery.TRANSFER_PERMUTATIONS,
        "n_boot": n_boot,
    }
    write_manifest(evaluation, out, manifest, Path(fits), slice_path, elapsed, speed=speed)
    log_summary(scored, out)
    logger.info(f"evaluation complete in {elapsed:.1f}s -> {out}")
    return scored
