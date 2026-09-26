"""Run configs (``config/runs/*.yaml``): what ``train.py`` fits and ``eval.py`` scores.

One *run* is one model split: ``fullfit`` or ``fold{i}_s{seed}``. Output layout::

    <out>/[k<K>/]<run>/fits/                     every arm, every seed (train.py)
    <out>/[k<K>/]<run>/floor/fits/               the null_random{d} floor draws
    <out>/[k<K>/]<run>/holdout/<group>/fits/     refits without one benchmark group
    <out>/[k<K>/]<run>/.../eval/                 the battery tables (eval.py)
    <out>/[k<K>/]scores.csv, summary.csv         skill over the width-matched floor

``k<K>`` appears only when ``train.interview_k`` is a list (a K sweep).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bullpen.config import REPO_ROOT

#: The arm and task registry every run config draws from.
PAPER_CONFIG = REPO_ROOT / "config" / "paper.yaml"
RUNS_DIR = REPO_ROOT / "config" / "runs"


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def load_paper(path: Path = PAPER_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass(frozen=True)
class Run:
    """One model split of a run config."""

    name: str
    #: ``fullfit`` for the fast protocol, else the fold scheme (``grouped_all``, ...)
    scheme: str
    fold: int | None
    #: the fold draw (strict) or the split seed stand-in (fullfit)
    fold_seed: int | None
    #: encoder seeds fitted in this run
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class RunConfig:
    name: str
    slice: Path
    answers: Path | None
    view: str | None
    split: str | None
    fold_scheme: str | None
    k_folds: int | None
    seeds: tuple[int, ...]
    dim: int | None
    interview_k: tuple[int | None, ...]
    interview_selector: str | None
    arms: tuple[str, ...]
    floor_arms: tuple[str, ...]
    floor_draws: int
    battery: Path
    #: config/benchmark_groups.yaml unless the cut has its own benchmark axis
    benchmark_groups: Path | None
    tasks: tuple[str, ...] | None
    transfer: bool
    holdout_skip: tuple[str, ...]
    out: Path
    #: where the fold plans live (default: ``folds/`` beside the slice)
    fold_dir: Path
    source: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def sweep(self) -> bool:
        return len(self.interview_k) > 1

    def runs(self) -> list[Run]:
        if self.fold_scheme is None:
            return [Run("fullfit", "fullfit", None, None, self.seeds)]
        return [
            Run(f"fold{i}_s{seed}", self.fold_scheme, i, seed, (seed,))
            for seed in self.seeds
            for i in range(int(self.k_folds))
        ]

    def roots(self) -> Iterator[tuple[int | None, Path]]:
        """``(interview_k, output root)`` per budget: ``out`` itself unless a K sweep."""
        for k in self.interview_k:
            yield k, (self.out / f"k{k}" if self.sweep else self.out)

    def floor_seeds(self, run: Run) -> tuple[int, ...]:
        """Encoder seeds of the floor draws: the multi-draw Gaussian floor keeps the run's
        folds and varies only the draw."""
        return tuple(range(self.floor_draws)) if self.floor_draws else ()


def load_run_config(path: Path | str) -> RunConfig:
    path = _path(path)
    raw = yaml.safe_load(path.read_text())
    paper = load_paper()
    data, protocol = raw["data"], raw["protocol"]
    train = raw.get("train") or {}
    floor = raw.get("floor") or {}
    ev = raw.get("eval") or {}
    arms = tuple(train.get("arms") or paper["arms"])
    unknown = sorted(set(arms) - set(paper["arms"]))
    if unknown:
        raise ValueError(f"{path.name}: arms not registered in config/paper.yaml: {unknown}")
    tasks = ev.get("tasks")
    if tasks is not None:
        bad = sorted(set(tasks) - set(paper["tasks"]))
        if bad:
            raise ValueError(f"{path.name}: tasks not in config/paper.yaml: {bad}")
    k = train.get("interview_k")
    ks = tuple(k) if isinstance(k, list) else (k,)
    scheme = protocol.get("fold_scheme")
    if scheme is None and protocol.get("split", "fullfit") != "fullfit":
        raise ValueError(f"{path.name}: protocol needs fold_scheme or split: fullfit")
    if scheme is not None and int(protocol.get("k_folds", 0)) < 2:
        raise ValueError(f"{path.name}: a fold scheme needs k_folds >= 2")
    slice_path = _path(data["slice"])
    return RunConfig(
        name=raw.get("name", path.stem),
        slice=slice_path,
        answers=_path(data["answers"]) if data.get("answers") else None,
        view=data.get("view"),
        split=None if scheme else "fullfit",
        fold_scheme=scheme,
        k_folds=int(protocol["k_folds"]) if scheme else None,
        seeds=tuple(int(s) for s in protocol.get("seeds", [0])),
        dim=train.get("dim"),
        interview_k=ks,
        interview_selector=train.get("interview_selector"),
        arms=arms,
        floor_arms=tuple(floor.get("arms") or ()),
        floor_draws=int(floor.get("draws", 0)),
        battery=_path(ev.get("battery", "config/battery.yaml")),
        benchmark_groups=_path(ev["benchmark_groups"]) if ev.get("benchmark_groups") else None,
        tasks=None if tasks is None else tuple(tasks),
        transfer=bool(ev.get("transfer", False)),
        holdout_skip=tuple(paper.get("holdout_skip") or ()),
        out=_path(raw.get("out", f"results/{path.stem}")),
        fold_dir=_path(data["fold_dir"]) if data.get("fold_dir") else slice_path.parent / "folds",
        source=path,
    )


def task_lists(cfg: RunConfig) -> tuple[list[str], list[str]]:
    """``(main tasks, benchmark-transfer tasks)`` of a run config, in paper.yaml order."""
    tasks = load_paper()["tasks"]
    chosen = [t for t in tasks if cfg.tasks is None or t in cfg.tasks]
    transfer = [t for t in chosen if tasks[t].get("benchmark_holdout")]
    return [t for t in chosen if t not in transfer], transfer
