"""Step 1 of 4: ask every model every probe question and grade the answers.

    uv run python collect.py --probe data/raw/probe_set.jsonl --models Qwen/Qwen2.5-7B-Instruct
    uv run python collect.py --probe data/core/probe.jsonl --models-file data/core/models.txt \\
        --recorded data/core/responses --raw results/sample/raw          # offline
    uv run python collect.py --probe ... --models ... --dry-run          # what would run

Writes one graded ``<raw>/responses/<model>/results.jsonl`` per model (see
:mod:`bullpen.collection`). Generation uses ``transformers`` (``uv sync --extra
collect``) under the budget in ``config/collect.yaml``; ``--recorded DIR`` grades
pre-recorded responses (``DIR/<model>.jsonl`` or ``.jsonl.xz``) instead and needs no
GPU or network.
Next: ``process.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", type=Path, required=True, help="probe set JSONL")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--models", nargs="+", help="model ids (Hub repo ids or local paths)")
    who.add_argument("--models-file", type=Path, help="one model id per line")
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw", help="output raw tree")
    ap.add_argument("--recorded", type=Path, default=None, help="grade DIR/<model>.jsonl offline")
    ap.add_argument("--dry-run", action="store_true", help="report the plan; load no model")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from loguru import logger

    from bullpen.collection import CollectError, collect_model, load_probe
    from bullpen.config import load_collect, load_grading
    from bullpen.logging import configure_console

    configure_console()
    models = args.models or [
        m.strip() for m in args.models_file.read_text().splitlines() if m.strip()
    ]
    try:
        probe = load_probe(args.probe)
    except CollectError as exc:
        logger.error(str(exc))
        return 1
    gen = load_collect().generation
    grading = load_grading()
    if args.dry_run:
        for m in models:
            k, t = gen.budget(m)
            logger.info(f"{m}: {len(probe)} questions at max_new_tokens={k} temperature={t}")
        return 0
    for m in models:
        collect_model(
            m,
            probe,
            args.raw,
            gen,
            recorded=args.recorded,
            sandbox=grading.sandbox(),
            markers=grading.reasoning_markers,
        )
    logger.info(f"done -> {args.raw / 'responses'}; next: process.py build --raw {args.raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
