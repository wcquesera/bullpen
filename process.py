"""Step 2 of 4: graded answers in, the slice and its embedding banks out.

    uv run python process.py build --probe P.jsonl --raw data/raw --out data/mycut
    uv run python process.py embed --probe P.jsonl --raw data/raw --out data/mycut      # BGE
    uv run python process.py embed ... --encoder hashing:32                             # offline

``build`` writes ``<out>/slice.npz`` (correctness bits ``A``/``R``, benchmark index,
model and question ids, VRAM and release dates where the metadata has them),
``<out>/question_blocks.json`` (the disjoint eval / tune / pool cut of the question
axis) and ``<out>/qc_report.json`` (who the gates dropped and why).
``embed`` writes ``<out>/answers.npz``: ``Ae`` [models, questions, D] answer
embeddings, ``Ae_mask`` (which cells carry an answer) and ``Qe`` [questions, D]
question embeddings. ``pack`` rewrites a bank as int8 parts under a size cap
(``answers.part00.npz``, ...), the form the shipped core subset uses; every loader
reads either form. See :mod:`bullpen.processing`. Next: ``train.py``.

    uv run python process.py pack --bank data/mycut/answers.npz            # -> answers.partNN.npz

The paper's cut (``data/paper_ckpt/``) was built this way over 260 models x
31,286 questions with ``--eval-size 4096 --tune-size 3072`` and is read through its
1,181-question core view. The block sizes default to those fractions of the axis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

#: The paper's eval and tune blocks as fractions of its 31,286-question axis
#: (bullpen.data.blocks.EVAL_SIZE / TUNE_SIZE); used when no size is given.
EVAL_FRAC = 4096 / 31286
TUNE_FRAC = 3072 / 31286


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (
        ("build", "answers -> slice + question blocks"),
        ("embed", "answers -> answer and question embeddings"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--probe", type=Path, required=True, help="probe set JSONL")
        p.add_argument("--raw", type=Path, default=REPO / "data" / "raw", help="collect.py's tree")
        p.add_argument("--out", type=Path, required=True, help="the cut directory to write")
    pk = sub.add_parser("pack", help="answer bank -> int8 parts of at most --part-mb MB")
    pk.add_argument("--bank", type=Path, required=True, help="an answers.npz to pack")
    pk.add_argument("--part-mb", type=float, default=None, help="size cap per part (default 90)")
    b = sub.choices["build"]
    b.add_argument("--models", nargs="*", default=None, help="restrict to these models")
    b.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="directory with model_footprints.json / model_release_dates.json (default: --raw)",
    )
    b.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="label projections to copy into <out>/labels/ (external_labels.json, ...)",
    )
    b.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help="assembly gate (default config/collect.yaml)",
    )
    b.add_argument(
        "--coverage-floor",
        type=float,
        default=None,
        help="slice gate (default config/process.yaml)",
    )
    b.add_argument("--drop-degenerate", action="store_true", help="drop flagged degenerate models")
    b.add_argument("--eval-size", type=int, default=None, help="scoring block width")
    b.add_argument("--tune-size", type=int, default=None, help="tuning block width")
    b.add_argument("--block-seed", type=int, default=None, help="question-block draw seed")
    e = sub.choices["embed"]
    e.add_argument(
        "--encoder",
        default="hf",
        help="'hf' (config/text_banks.yaml, the paper) or 'hashing:<dim>' (offline)",
    )
    e.add_argument("--device", default=None, help="torch device for 'hf'")
    return ap


def cmd_build(args: argparse.Namespace) -> int:
    import dataclasses
    import shutil

    from loguru import logger

    from bullpen.collection import load_probe
    from bullpen.config import load_collect, load_process
    from bullpen.data.assemble import assemble
    from bullpen.data.blocks import BLOCK_SEED, axis_labels, build_blocks, save_blocks
    from bullpen.data.slice import save_slice
    from bullpen.processing import gate_and_build, write_json

    probe = load_probe(args.probe)
    gates = load_collect().assembly
    if args.min_coverage is not None:
        gates = dataclasses.replace(gates, min_coverage=args.min_coverage)
    sub = assemble(args.raw, [i.id for i in probe], [i.axis for i in probe], gates, args.models)
    if not sub.models:
        logger.error(f"no collected model in {args.raw / 'responses'} passes the gates")
        return 1
    pcfg = load_process()
    floor = pcfg.coverage_floor if args.coverage_floor is None else args.coverage_floor
    sl, report = gate_and_build(sub, floor, pcfg.degenerate_accuracy, args.drop_degenerate)
    args.out.mkdir(parents=True, exist_ok=True)
    save_slice(sl, slice_path=args.out / "slice.npz", answers_path=None)
    n = sl.n_questions
    eval_size = args.eval_size or max(1, round(n * EVAL_FRAC))
    tune_size = args.tune_size or max(1, round(n * TUNE_FRAC))
    seed = BLOCK_SEED if args.block_seed is None else args.block_seed
    save_blocks(
        build_blocks(axis_labels(sl), eval_size, tune_size, seed), args.out / "question_blocks.json"
    )
    write_json(args.out / "qc_report.json", report)
    if args.labels is not None:
        shutil.copytree(args.labels, args.out / "labels", dirs_exist_ok=True)
    logger.info(sl.summary())
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    import numpy as np

    from bullpen.collection import load_probe
    from bullpen.processing import answer_banks, make_encoder, save_banks

    with np.load(args.out / "slice.npz", allow_pickle=False) as z:
        models = [str(m) for m in z["model_ids"]]
        questions = [str(q) for q in z["question_ids"]]
    text = {i.id: i.question for i in load_probe(args.probe)}
    Ae, mask, Qe = answer_banks(
        args.raw, models, questions, text, make_encoder(args.encoder, args.device)
    )
    save_banks(args.out / "answers.npz", Ae, mask, Qe)
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    import numpy as np

    from bullpen.data.slice import PACKED_PART_MB, save_packed_answers

    with np.load(args.bank, allow_pickle=False) as z:
        save_packed_answers(
            z["Ae"],
            z["Ae_mask"],
            z["Qe"] if "Qe" in z.files else None,
            args.bank,
            args.part_mb or PACKED_PART_MB,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "build":
        # model metadata is read from the raw tree (bullpen.config.RAW_DIR) at import
        os.environ["BULLPEN_RAW_DIR"] = str((args.meta or args.raw).resolve())
    from bullpen.logging import configure_console

    configure_console()
    return {"build": cmd_build, "embed": cmd_embed, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
