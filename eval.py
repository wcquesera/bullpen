"""Step 4 of 4: score a run config's fitted arms and read them against the Gaussian floor.

    uv run python eval.py --config config/runs/strict.yaml
    uv run python eval.py --config config/runs/strict.yaml --run fold0_s0 --no-bootstrap
    uv run python eval.py --config config/runs/strict.yaml --table-only   # rebuild the tables

For every run ``train.py`` fitted, this scores ``<run>/fits`` on the config's tasks
(the eval question block, the run's held-out models), the floor draws in
``<run>/floor/fits``, and each benchmark-group refit on the transfer tasks. It then
writes, under the config's ``out`` (``out/k<K>`` for a K sweep):

    scores.csv    one row per (run, group, arm, task): the chance-anchored skill ``s``,
                  the width-matched floor ``floor_s`` (mean over the floor draws of the
                  nearest-width null_random{d}) and ``s_over_floor``
    summary.csv   per (arm, task, group): those three averaged over runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train import set_raw_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="a run config (config/runs/*.yaml)")
    ap.add_argument("--run", nargs="*", default=None, help="only these runs (e.g. fold0_s0)")
    ap.add_argument(
        "--no-bootstrap", action="store_true", help="skip the paired task bootstrap (faster)"
    )
    ap.add_argument(
        "--table-only", action="store_true", help="only rebuild scores.csv / summary.csv"
    )
    ap.add_argument("--out", type=Path, default=None, help="override the config's out")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_raw_dir(args.config)

    from dataclasses import replace

    from loguru import logger

    from bullpen.floor import score_table, summary_table
    from bullpen.logging import configure_console
    from bullpen.runs import load_run_config, task_lists
    from bullpen.scoring import evaluate_run

    configure_console()
    rc = load_run_config(args.config)
    if args.out is not None:
        rc = replace(rc, out=args.out)
    main_tasks, transfer_tasks = task_lists(rc)
    runs = [r.name for r in rc.runs() if not args.run or r.name in args.run]

    def score(fits: Path, tasks: list[str]) -> None:
        if not tasks:
            return
        if not (fits / "manifest.json").is_file():
            logger.warning(f"{fits}: not fitted, skipped")
            return
        evaluate_run(
            fits,
            fits.parent / "eval",
            tasks=tasks,
            battery_config=rc.battery,
            benchmark_groups=rc.benchmark_groups,
            bootstrap=not args.no_bootstrap,
        )

    status = 0
    for _, root in rc.roots():
        if not args.table_only:
            for run in runs:
                base = root / run
                score(base / "fits", main_tasks)
                score(base / "floor" / "fits", main_tasks)
                if transfer_tasks:
                    for gdir in sorted((base / "holdout").glob("*")):
                        score(gdir / "fits", transfer_tasks)
                        score(gdir / "floor" / "fits", transfer_tasks)
        scores = score_table(root, [r.name for r in rc.runs()])
        if scores.empty:
            logger.error(f"{root}: nothing scored yet")
            status = 1
            continue
        summary = summary_table(scores)
        scores.to_csv(root / "scores.csv", index=False)
        summary.to_csv(root / "summary.csv", index=False)
        overall = (
            summary[summary.group == ""]
            .groupby("arm")[["s", "floor_s", "s_over_floor"]]
            .mean()
            .sort_values("s_over_floor", ascending=False)
        )
        logger.info(f"{root}: mean over tasks (main evaluation)\n" + overall.round(3).to_string())
        logger.info(f"wrote {root / 'scores.csv'} and {root / 'summary.csv'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
