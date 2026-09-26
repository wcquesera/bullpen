"""Skill over the width-matched Gaussian floor: the unit every reported number is in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["mean", "ci_lo", "ci_hi", "n_units", "chance", "s", "primary", "diag_emb_dim"]
EMPTY_FLOOR = ["method", "task", "group", "s", "draws", "sd"]


def read_cells(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.is_file()]
    if not frames:
        return pd.DataFrame()
    cells = pd.concat(frames, ignore_index=True)
    return cells[["method", "task", *[c for c in COLUMNS if c in cells.columns]]]


def run_rows(run_dir: Path) -> pd.DataFrame:
    """The main evaluation plus every held-out benchmark group, and the mean over groups."""
    main = read_cells([run_dir / "eval" / "cells.csv"]).assign(group="")
    groups = [
        read_cells([cells]).assign(group=cells.parents[1].name)
        for cells in sorted(run_dir.glob("holdout/*/eval/cells.csv"))
    ]
    if not groups:
        return main
    held = pd.concat(groups, ignore_index=True)
    mean = (
        held.groupby(["method", "task"], as_index=False)
        .agg(
            mean=("mean", "mean"),
            ci_lo=("mean", "min"),
            ci_hi=("mean", "max"),
            n_units=("group", "nunique"),
            chance=("chance", "mean"),
            s=("s", "mean"),
            primary=("primary", "first"),
            diag_emb_dim=("diag_emb_dim", "first"),
        )
        .assign(group="mean")
    )
    return pd.concat([main, held, mean], ignore_index=True)


def floor_rows(run_dir: Path) -> pd.DataFrame:
    """The multi-draw nulls of one run: pooled ``s`` per (arm, task, group), the number of
    draws and their spread. Empty when no floor was fitted for this run."""
    base = run_dir / "floor"
    rows = pd.DataFrame()
    if base.is_dir():
        main = read_cells([base / "eval" / "cells.csv"]).assign(group="")
        held = [
            read_cells([p]).assign(group=p.parents[2].name)
            for p in sorted(run_dir.glob("holdout/*/floor/eval/cells.csv"))
        ]
        rows = pd.concat([main, *held], ignore_index=True)
    if rows.empty:
        return pd.DataFrame(columns=EMPTY_FLOOR)
    rows = rows[rows.method.str.startswith("null_random")]
    per_seed = [
        pd.read_csv(p).assign(group="")
        for p in [base / "eval" / "cells_unpooled.csv"]
        if p.is_file()
    ] + [
        pd.read_csv(p).assign(group=p.parents[2].name)
        for p in sorted(run_dir.glob("holdout/*/floor/eval/cells_unpooled.csv"))
    ]
    spread = (
        pd.concat(per_seed, ignore_index=True)
        .groupby(["method", "task", "group"])
        .s.agg(draws="count", sd="std")
        .reset_index()
        if per_seed
        else pd.DataFrame(columns=["method", "task", "group", "draws", "sd"])
    )
    out = rows[["method", "task", "group", "s"]].merge(
        spread, on=["method", "task", "group"], how="left"
    )
    held = out[out.group != ""]
    if len(held):
        # the transfer mean over held-out groups: draws per group, spread not defined
        mean = held.groupby(["method", "task"], as_index=False).s.mean().assign(group="mean")
        mean["draws"] = int(spread.draws.max()) if len(spread) else np.nan
        out = pd.concat([out, mean], ignore_index=True)
    return out


def attach_floor(rows: pd.DataFrame, floors: pd.DataFrame | None = None) -> pd.DataFrame:
    """The nearest-width ``null_random{d}`` on the same task and group, and the gap to it."""
    nulls = rows[rows.method.str.startswith("null_random")].copy()
    if floors is not None and len(floors):
        nulls = pd.concat([nulls, floors.assign(n_units=np.nan)], ignore_index=True)
    nulls["width"] = nulls.method.str.removeprefix("null_random").astype(int)
    widths = sorted(nulls.width.unique())
    out = rows.copy()
    if not widths:
        out["floor_arm"], out["floor_s"], out["s_over_floor"] = "", np.nan, np.nan
        out["floor_draws"], out["floor_sd"] = np.nan, np.nan
        return out
    dim = pd.to_numeric(out.get("diag_emb_dim"), errors="coerce").fillna(widths[0])
    out["floor_arm"] = [f"null_random{min(widths, key=lambda w: abs(w - d))}" for d in dim]
    single = rows[rows.method.str.startswith("null_random")].set_index(["method", "task", "group"])[
        "s"
    ]
    multi = (
        floors.set_index(["method", "task", "group"])
        if floors is not None and len(floors)
        else pd.DataFrame(columns=["s", "draws", "sd"])
    )
    s_, draws, sd = [], [], []
    for f, t, g in zip(out.floor_arm, out.task, out.group, strict=True):
        if (f, t, g) in multi.index and pd.notna(multi.at[(f, t, g), "s"]):
            s_.append(multi.at[(f, t, g), "s"])
            draws.append(multi.at[(f, t, g), "draws"])
            sd.append(multi.at[(f, t, g), "sd"])
        else:
            s_.append(single.get((f, t, g), np.nan))
            draws.append(1 if (f, t, g) in single.index else np.nan)
            sd.append(np.nan)
    out["floor_s"], out["floor_draws"], out["floor_sd"] = s_, draws, sd
    out["s_over_floor"] = out.s - out.floor_s
    return out


def score_table(root: Path, runs: list[str]) -> pd.DataFrame:
    """One row per (run, group, arm, task): ``s``, its floor and ``s_over_floor``."""
    parts = []
    for run in runs:
        run_dir = root / run
        rows = run_rows(run_dir)
        if rows.empty:
            continue
        parts.append(attach_floor(rows, floor_rows(run_dir)).assign(run=run))
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True).rename(columns={"method": "arm", "n_units": "n"})
    first = ["run", "group", "arm", "task"]
    return table[first + [c for c in table.columns if c not in first]]


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, task, group): the mean over runs of ``s``, ``floor_s`` and ``s_over_floor``."""
    if scores.empty:
        return scores
    return (
        scores.groupby(["arm", "task", "group"], as_index=False, dropna=False)
        .agg(
            s=("s", "mean"),
            floor_s=("floor_s", "mean"),
            s_over_floor=("s_over_floor", "mean"),
            s_over_floor_sd=("s_over_floor", "std"),
            n_runs=("run", "nunique"),
            floor_arm=("floor_arm", "first"),
        )
        .sort_values(["group", "task", "arm"])
        .reset_index(drop=True)
    )
