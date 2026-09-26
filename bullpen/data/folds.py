"""K-fold plans over models, stored on disk and joined on model id.

A plan assigns every model to a fold once and is written to ``folds/`` beside the
slice, so every run of a (scheme, k, seed) scores the same held-out models. The
``grouped_all`` scheme keeps each publisher family (the footprint file's ``family``
field) inside one fold, so a model's same-publisher siblings are never on the other
side of the split; a model with no family is its own group.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.data.enrichment import families_for_models, load_footprints
from bullpen.data.slice import DATA_DIR, Slice
from bullpen.data.splits import Split, _finalise

#: Default fold seed.
DEFAULT_FOLD_SEED: int = 0

#: Where plans live: one file per (scheme, k, seed), see :func:`plan_path`.
DEFAULT_FOLD_DIR = DATA_DIR / "folds"

#: Fold id of a model that trains in every fold and is never held out.
ALWAYS_TRAIN: int = -1

#: Fold id of a model the plan uses for nothing (neither test nor training row).
EXCLUDED: int = -2

#: Publisher families kept whole, every model in some fold.
GROUPED_ALL = "grouped_all"

#: Group key prefix of a model with no recorded family (each is its own group).
UNGROUPED_PREFIX = "model:"

#: ``(slice, k, seed) -> [M] int fold ids``, values in ``{ALWAYS_TRAIN, 0..k-1}``.
FoldFn = Callable[[Slice, int, int], np.ndarray]

FOLD_SCHEMES: dict[str, FoldFn] = {}

#: JSON keys of a saved plan; ``fold_of`` maps model id to fold id.
PLAN_KEYS: tuple[str, ...] = ("scheme", "k", "seed", "fold_of")


def fold_scheme(name: str) -> Callable[[FoldFn], FoldFn]:
    """Register a fold scheme under ``name``."""

    def wrap(fn: FoldFn) -> FoldFn:
        if name in FOLD_SCHEMES:
            raise ValueError(f"fold scheme {name!r} is already registered")
        FOLD_SCHEMES[name] = fn
        return fn

    return wrap


def _check_k(k: int, n: int, what: str) -> int:
    """``k`` folds over ``n`` rows, or a ValueError naming both."""
    k = int(k)
    if k < 2:
        raise ValueError(f"k must be at least 2 to cross-validate, got {k}")
    if n < k:
        raise ValueError(f"cannot cut {k} folds from {n} {what}")
    return k


def family_groups(
    model_ids: Sequence[str], footprints: Mapping[str, Mapping[str, Any]] | None = None
) -> list[str]:
    """[M] group key per model: its publisher family, or its own id if it has none."""
    if footprints is None:
        footprints = load_footprints()
    families = families_for_models(model_ids, footprints)
    return [f if f else f"{UNGROUPED_PREFIX}{m}" for m, f in zip(model_ids, families, strict=True)]


def _greedy_group_folds(groups: Sequence[str], k: int, seed: int) -> dict[str, int]:
    """``{group key: fold}`` — largest group first into the least-loaded fold.

    ``seed`` orders groups that tie on size; ties between equally loaded folds go to
    the lowest fold index.
    """
    sizes: dict[str, int] = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    keys = sorted(sizes)
    shuffled = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    order = sorted(shuffled, key=lambda g: -sizes[g])  # stable: size first, draw second
    load = np.zeros(k, dtype=int)
    fold_of_group: dict[str, int] = {}
    for g in order:
        fold = int(np.argmin(load))
        fold_of_group[g] = fold
        load[fold] += sizes[g]
    return fold_of_group


@fold_scheme(GROUPED_ALL)
def grouped_all_folds(
    sl: Slice,
    k: int,
    seed: int,
    footprints: Mapping[str, Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Publisher families kept whole inside one fold, folds balanced by row count."""
    rows = list(range(sl.n_models))
    groups = family_groups([sl.model_ids[i] for i in rows], footprints)
    k = _check_k(k, len(set(groups)), "family group(s)")

    fold_of_group = _greedy_group_folds(groups, k, int(seed))
    assignments = np.full(sl.n_models, EXCLUDED, dtype=int)
    folds_of_group: dict[str, set[int]] = {}
    for row, group in zip(rows, groups, strict=True):
        assignments[row] = fold_of_group[group]
        folds_of_group.setdefault(group, set()).add(int(assignments[row]))

    straddling = {g for g, folds in folds_of_group.items() if len(folds) > 1}
    if straddling:
        raise AssertionError(
            f"grouped_all folds split {len(straddling)} family/families: {straddling}"
        )
    if (assignments < 0).any():
        raise AssertionError(
            f"{int((assignments < 0).sum())} model(s) got no fold under grouped_all"
        )
    logger.info(
        f"grouped_all folds: {len(set(groups))} group(s) over {len(rows)} model(s), "
        f"largest {max(groups.count(g) for g in set(groups))}"
    )
    return assignments


@dataclass(frozen=True)
class FoldPlan:
    """One (scheme, k, seed) assignment of every model to a fold."""

    scheme: str
    k: int
    seed: int
    model_ids: tuple[str, ...]
    assignments: np.ndarray  # [M] int, values in {EXCLUDED, ALWAYS_TRAIN, 0..k-1}
    source: str  # "built" | "file:<name>"

    @property
    def n_always_train(self) -> int:
        return int(np.count_nonzero(self.assignments == ALWAYS_TRAIN))

    @property
    def n_excluded(self) -> int:
        """Rows in neither bank (:data:`EXCLUDED`)."""
        return int(np.count_nonzero(self.assignments == EXCLUDED))

    def excluded(self) -> np.ndarray:
        """Global rows the plan uses for nothing, in any fold."""
        return np.flatnonzero(self.assignments == EXCLUDED)

    def members(self, fold: int) -> np.ndarray:
        """Global rows assigned to ``fold``."""
        if not 0 <= int(fold) < self.k:
            raise ValueError(f"fold must be in [0, {self.k}), got {fold}")
        return np.flatnonzero(self.assignments == int(fold))

    def split(self, fold: int) -> Split:
        """``fold`` is the holdout; every other row but the :data:`EXCLUDED` ones trains."""
        test = self.members(fold)
        train = np.setdiff1d(np.arange(self.assignments.size), np.union1d(test, self.excluded()))
        return _finalise(
            train, test, f"kfold:{self.scheme}:k={self.k}:fold={fold}:seed={self.seed}"
        )

    def as_dict(self) -> dict:
        """The stored form: fold per model id, not per row."""
        return {
            "scheme": self.scheme,
            "k": self.k,
            "seed": self.seed,
            "fold_of": {m: int(f) for m, f in zip(self.model_ids, self.assignments, strict=True)},
        }

    def describe(self) -> str:
        sizes = [int(self.members(f).size) for f in range(self.k)]
        # Arithmetic rather than self.split(f), which logs a line per fold.
        banks = [self.assignments.size - self.n_excluded - n for n in sizes]
        excluded = f", {self.n_excluded} excluded" if self.n_excluded else ""
        return (
            f"{self.scheme} k={self.k} seed={self.seed}: sizes {sizes}, "
            f"banks {banks}, {self.n_always_train} always-train{excluded}"
        )


def plan_path(scheme: str, k: int, seed: int, directory: Path = DEFAULT_FOLD_DIR) -> Path:
    """``<directory>/<scheme>_k<k>_s<seed>.json``."""
    return Path(directory) / f"{scheme}_k{int(k)}_s{int(seed)}.json"


def build_plan(sl: Slice, scheme: str, k: int, seed: int) -> FoldPlan:
    """Draw a plan for ``sl`` under a registered scheme, and check it is usable."""
    if scheme not in FOLD_SCHEMES:
        raise ValueError(f"unknown fold scheme {scheme!r}; registered: {sorted(FOLD_SCHEMES)}")
    k = int(k)
    assignments = np.asarray(FOLD_SCHEMES[scheme](sl, k, int(seed)), dtype=int)
    plan = FoldPlan(
        scheme=scheme,
        k=k,
        seed=int(seed),
        model_ids=tuple(sl.model_ids),
        assignments=assignments,
        source="built",
    )
    for fold in range(k):
        n_test = int(plan.members(fold).size)
        if n_test == 0:
            raise ValueError(f"{scheme} k={k} leaves fold {fold} empty on {sl.n_models} model(s)")
        # a one-row training bank cannot be fitted
        n_train = sl.n_models - plan.n_excluded - n_test
        if n_train < 2:
            raise ValueError(
                f"{scheme} k={k} leaves fold {fold} with {n_train} training row(s); "
                f"a bank of fewer than 2 rows has no rank to report"
            )
    logger.info(f"built fold plan: {plan.describe()}")
    return plan


def save_plan(plan: FoldPlan, path: Path) -> Path:
    """Write ``plan`` as JSON. Model ids, so a re-ordered slice still joins."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=False) + "\n")
    logger.info(f"wrote fold plan {path} ({plan.describe()})")
    return path


def load_plan(path: Path, sl: Slice) -> FoldPlan:
    """Read a plan and join it onto ``sl`` by model id; a model-set mismatch raises."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no fold plan at {path}")
    payload = json.loads(path.read_text())
    missing_keys = [k for k in PLAN_KEYS if payload.get(k) is None]
    if missing_keys:
        raise ValueError(f"{path.name} is missing key(s) {missing_keys}")
    fold_of = payload["fold_of"]
    absent = [m for m in fold_of if m not in set(sl.model_ids)]
    if absent:
        raise ValueError(
            f"{path.name} assigns {len(absent)} model(s) absent from the slice, e.g. "
            f"{absent[:3]} — the plan and the slice are different cuts"
        )
    unassigned = [m for m in sl.model_ids if m not in fold_of]
    if unassigned:
        raise ValueError(
            f"{path.name} assigns no fold to {len(unassigned)} slice model(s), e.g. "
            f"{unassigned[:3]} — the plan and the slice are different cuts"
        )
    plan = FoldPlan(
        scheme=str(payload["scheme"]),
        k=int(payload["k"]),
        seed=int(payload["seed"]),
        model_ids=tuple(sl.model_ids),
        assignments=np.array([int(fold_of[m]) for m in sl.model_ids], dtype=int),
        source=f"file:{path.name}",
    )
    logger.info(f"loaded fold plan {path} ({plan.describe()})")
    return plan


def resolve_plan(
    sl: Slice,
    scheme: str,
    k: int,
    seed: int,
    directory: Path = DEFAULT_FOLD_DIR,
) -> FoldPlan:
    """The plan on disk for this (scheme, k, seed), drawing and writing one if absent."""
    path = plan_path(scheme, k, seed, directory)
    if path.exists():
        logger.info(f"fold plan: reading {path}")
        return load_plan(path, sl)
    logger.info(f"fold plan: no {path} — drawing one")
    plan = build_plan(sl, scheme, k, seed)
    save_plan(plan, path)
    return plan
