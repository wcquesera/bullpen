"""Step 3 of 4: fit the encoder arms of a run config, one model split at a time.

    uv run python train.py --config config/runs/strict.yaml              # every run
    uv run python train.py --config config/runs/strict.yaml --run fold0_s0
    uv run python train.py --config config/runs/sample.yaml --arms pca_bits32 null_random32

For every run (``fullfit``, or ``fold{i}_s{seed}`` of a K-fold over models) this fits

1. every arm of the config, at the run's seed(s), on the run's training models and
   the question POOL (or the K-question interview for a ``__interview`` / ``__itext``
   twin) into ``<out>/<run>/fits/``;
2. the Gaussian floor: each ``floor.arms`` null (``null_random{d}``) at ``floor.draws``
   encoder seeds on the same split, into ``<out>/<run>/floor/fits/``;
3. with ``eval.transfer``, every arm again without each benchmark group
   (``config/benchmark_groups.yaml``) into ``<out>/<run>/holdout/<group>/fits/``.

Score the result with ``eval.py --config`` on the same file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="a run config (config/runs/*.yaml)")
    ap.add_argument("--run", nargs="*", default=None, help="only these runs (e.g. fold0_s0)")
    ap.add_argument("--arms", nargs="*", default=None, help="only these arms of the config")
    ap.add_argument("--no-floor", action="store_true", help="skip the floor draws")
    ap.add_argument("--no-transfer", action="store_true", help="skip the benchmark-group refits")
    ap.add_argument("--out", type=Path, default=None, help="override the config's out")
    return ap


def set_raw_dir(config: Path) -> None:
    """Point :data:`bullpen.config.RAW_DIR` at the config's ``data.raw_dir`` before import."""
    raw = (yaml.safe_load(Path(config).read_text()).get("data") or {}).get("raw_dir")
    if raw:
        os.environ["BULLPEN_RAW_DIR"] = str(Path(raw) if Path(raw).is_absolute() else REPO / raw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_raw_dir(args.config)

    from dataclasses import replace

    import numpy as np
    from loguru import logger

    from bullpen.config import load_benchmark_groups, load_splits, load_train
    from bullpen.data.blocks import blocks_for_slice, without_columns
    from bullpen.data.folds import resolve_plan
    from bullpen.data.slice import load_slice
    from bullpen.data.splits import resolve_split
    from bullpen.logging import configure_console
    from bullpen.runs import load_run_config
    from bullpen.training import (
        resolve_dim,
        resolve_interview_k,
        resolve_interview_selector,
        train,
        write_manifest,
    )

    configure_console()
    rc = load_run_config(args.config)
    if args.out is not None:
        rc = replace(rc, out=args.out)
    arms = list(args.arms or rc.arms)
    if bad := sorted(set(arms) - set(rc.arms)):
        raise SystemExit(f"arms not in {rc.source.name}: {bad}")
    runs = [r for r in rc.runs() if not args.run or r.name in args.run]
    if not runs:
        raise SystemExit(f"no run matches {args.run}; runs: {[r.name for r in rc.runs()]}")

    splits_cfg = load_splits()
    sl = load_slice(rc.slice, rc.answers, view=rc.view)
    base_blocks = blocks_for_slice(sl, path=rc.slice.parent / "question_blocks.json")
    groups = (
        load_benchmark_groups(sl.bench_names, rc.benchmark_groups)
        if rc.transfer and not args.no_transfer
        else {}
    )

    def fit(out, split, fold, blocks, cfg, names, seeds, held_out=None) -> None:
        started = time.perf_counter()
        outcome = train(
            sl,
            split,
            out,
            cfg,
            blocks,
            splits_cfg.split_seed,
            arms=names,
            seeds=seeds,
            llmmap_bank_path=rc.slice.parent / "llmmap_probe_bank.npz",
        )
        write_manifest(
            outcome,
            out,
            sl,
            split,
            rc.slice,
            cfg,
            seeds,
            blocks,
            time.perf_counter() - started,
            arms_requested=list(names),
            fold=fold,
            fold_dir=rc.fold_dir,
            splits_cfg=splits_cfg,
            answers_path=rc.answers,
            run_config=str(rc.source),
            view=rc.view,
            held_out=held_out,
        )
        if outcome.skipped:
            logger.warning(
                f"{out}: {len(outcome.skipped)} arm-run(s) not fitted: "
                + "; ".join(f"{s.arm} ({s.reason})" for s in outcome.skipped)
            )

    for k, root in rc.roots():
        cfg = resolve_interview_selector(
            resolve_interview_k(load_train(), k), rc.interview_selector
        )
        cfg = resolve_dim(cfg, rc.dim)
        for run in runs:
            if run.fold is None:
                split, fold = resolve_split(sl, strategy="fullfit"), None
            else:
                plan = resolve_plan(sl, run.scheme, rc.k_folds, run.fold_seed, rc.fold_dir)
                split, fold = plan.split(run.fold), (plan, run.fold)
            base = root / run.name
            logger.info(f"[{rc.name} K={cfg.interview_k}] {run.name}: {len(arms)} arms")
            fit(base / "fits", split, fold, base_blocks, cfg, arms, run.seeds)
            floor = [] if args.no_floor else list(rc.floor_arms)
            if floor:
                fit(
                    base / "floor" / "fits",
                    split,
                    fold,
                    base_blocks,
                    cfg,
                    floor,
                    rc.floor_seeds(run),
                )
            refit = [a for a in arms if a not in rc.holdout_skip]
            for g, members in groups.items():
                idx = [sl.bench_names.index(b) for b in members]
                cols = np.flatnonzero(np.isin(sl.bench, idx))
                blocks = without_columns(base_blocks, cols)
                gdir = base / "holdout" / g
                fit(gdir / "fits", split, fold, blocks, cfg, refit, run.seeds, (g, cols))
                if floor:
                    fit(
                        gdir / "floor" / "fits",
                        split,
                        fold,
                        blocks,
                        cfg,
                        floor,
                        rc.floor_seeds(run),
                        (g, cols),
                    )
    logger.info(f"done -> {rc.out}; score with: uv run python eval.py --config {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
