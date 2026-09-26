"""Fit encoder arms on one model split and save the artifacts (library for ``train.py``).

:func:`train` fits every selected arm at every seed on the training rows of one
split, on the question block the registry declares for it (the pool, or the K
interview questions for a -budget twin), and writes one pickle per (arm, seed).
:func:`write_manifest` records the split, the eval/tune/pool question blocks and
each arm's width, which :mod:`bullpen.scoring` reads back. An arm whose input is
absent or whose fit raises is recorded as skipped.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.competitors.embedllm.bridge import load_if_present as load_mpnet_bridge
from bullpen.config import (
    SplitsConfig,
    TrainConfig,
    config_dir,
)
from bullpen.data.blocks import QuestionBlocks
from bullpen.data.folds import (
    DEFAULT_FOLD_DIR,
    FoldPlan,
    plan_path,
)
from bullpen.data.llmmap_probes import aligned_probe_tables
from bullpen.data.slice import Slice
from bullpen.data.splits import Split
from bullpen.data.text_banks import pooled_model_text
from bullpen.evaluation.metrics import pca95
from bullpen.models.base import CPU_THREADS, Encoder, park_fitted_modules
from bullpen.models.question_selection import SELECTORS, interview_columns
from bullpen.models.registry import (
    ENCODER_ORDER,
    INTERVIEW_TEXT_ARMS,
    IRT_FAMILY_ARMS,
    arm_blocks,
    arm_fit_block,
    build_encoders,
)

ARTIFACT_SUFFIX = ".pkl"
MANIFEST_NAME = "manifest.json"
_HASH_CHUNK = 1 << 20
#: the rows every arm's ``pca95`` diagnostic is computed over
PCA95_ROWS = "train"


@dataclass
class Artifact:
    """One fitted arm: the whole encoder (to place held-out models later) and its metadata."""

    encoder: Encoder
    arm: str
    seed: int
    dim: int
    input_width: int
    #: slice rows the arm was fitted on, as indices into the FULL model axis
    train_rows: np.ndarray
    #: global question ids the arm spans
    cols: np.ndarray
    #: question blocks this arm consumed, fitting block first
    blocks: tuple[str, ...]
    fit_seconds: float

    def meta(self) -> dict[str, Any]:
        """The manifest row for this artifact: everything but the encoder itself."""
        bank = self.encoder.X
        text_width = int(getattr(self.encoder, "text_width", 0) or 0)
        return {
            "arm": self.arm,
            "seed": self.seed,
            "dim": self.dim,
            "input_width": self.input_width,
            # correctness columns and text dimensions, separately and summed
            "text_width": text_width,
            "total_input_width": self.input_width + text_width,
            "bank_shape": None if bank is None else list(bank.shape),
            "blocks": list(self.blocks),
            "n_cols": int(np.asarray(self.cols).size),
            "pca95": bank_pca95(bank),
            "n_train": int(self.train_rows.size),
            "fit_seconds": round(self.fit_seconds, 3),
            "file": artifact_name(self.arm, self.seed),
        }


@dataclass
class Skip:
    """An arm that produced no artifact, and why."""

    arm: str
    reason: str
    #: ``None`` when the arm could not be built at all, so no seed was reached
    seed: int | None = None


@dataclass
class Outcome:
    """What one training run produced."""

    artifacts: list[Artifact] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)
    #: the K interview columns, or ``None`` when no arm in the run asked for them
    interview: np.ndarray | None = None


def bank_pca95(bank: np.ndarray | None) -> int | None:
    """Components carrying 95% of a fitted bank's variance, over the training rows.

    A diagnostic written to the manifest; ``None`` when undefined (one row, or a
    constant bank).
    """
    if bank is None:
        return None
    try:
        return pca95(np.asarray(bank, dtype=float))
    except ValueError as exc:
        logger.debug(f"pca95 is undefined for this bank: {exc}")
        return None


def artifact_name(arm: str, seed: int) -> str:
    """File name of one fitted arm: ``<arm>__s<seed>.pkl``."""
    return f"{arm}__s{seed}{ARTIFACT_SUFFIX}"


def file_digest(path: Path) -> str:
    """sha256 of a file, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def config_digest(directory: Path | None = None) -> str:
    """sha256 over every ``config/`` file, in name order."""
    directory = config_dir() if directory is None else Path(directory)
    h = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def git_sha() -> str:
    """Recorded in manifests; a release checkout has no revision to record."""
    return "unknown"


def resolve_interview_k(cfg: TrainConfig, override: int | None) -> TrainConfig:
    """``cfg`` with ``interview_k`` replaced when a run config sets one."""
    if override is None:
        return cfg
    if override < 1:
        raise ValueError(f"interview_k must be at least 1, got {override}")
    return replace(cfg, interview_k=override)


def resolve_interview_selector(cfg: TrainConfig, override: str | None) -> TrainConfig:
    """``cfg`` with ``interview_selector`` replaced when a run config sets one."""
    if override is None:
        return cfg
    if override not in SELECTORS:
        raise ValueError(f"unknown selector {override!r}; registered: {sorted(SELECTORS)}")
    return replace(cfg, interview_selector=override)


def resolve_dim(cfg: TrainConfig, override: int | None) -> TrainConfig:
    """``cfg`` with the embedding width replaced when a run config sets one."""
    return cfg if override is None else replace(cfg, dim=override)


def resolve_arms(arms: Sequence[str] | None = None) -> list[str] | None:
    """The requested arms in registry order, or ``None`` for every arm."""
    if arms is None:
        return None
    unknown = [a for a in arms if a not in ENCODER_ORDER]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known arms: {ENCODER_ORDER}")
    return [a for a in ENCODER_ORDER if a in arms]


def select_interview(
    sl: Slice, split: Split, blocks: QuestionBlocks, cfg: TrainConfig, seed: int
) -> np.ndarray:
    """The run's K interview columns, ranked inside the pool block on the training rows.

    Drawn once per run from the split seed, so every arm and seed of a run reads the
    same interview.
    """
    rng = np.random.default_rng(seed)
    cols = interview_columns(
        sl.A[split.train][:, blocks.pool_idx],
        blocks.pool_idx,
        cfg.interview_k,
        cfg.interview_selector,
        rng,
    )
    logger.info(
        f"interview: {cols.size} question(s) by {cfg.interview_selector!r} out of the "
        f"{blocks.n_pool}-column pool, on {split.n_train} training models"
    )
    return cols


def fit_one(
    arm: str,
    encoder: Encoder,
    sl: Slice,
    split: Split,
    cols: np.ndarray,
    blocks: tuple[str, ...],
    seed: int,
) -> Artifact:
    """Fit one arm on the training rows over ``cols`` and time it."""
    started = time.perf_counter()
    encoder.fit(sl.A[split.train][:, cols], cols)
    elapsed = time.perf_counter() - started
    bank = encoder.X
    if bank is None:
        raise RuntimeError(f"{arm}: fit() returned without filling a bank")
    if bank.shape[0] != split.n_train:
        raise RuntimeError(
            f"{arm}: fitted a bank of {bank.shape[0]} models on {split.n_train} "
            f"training rows — the arm and the split disagree about the cut"
        )
    logger.info(
        f"{arm} s{seed}: bank {bank.shape} width={encoder.input_width} "
        f"on {'+'.join(blocks)} in {elapsed:.1f}s"
    )
    return Artifact(
        encoder=encoder,
        arm=arm,
        seed=seed,
        dim=int(encoder.dim),
        input_width=int(encoder.input_width),
        train_rows=split.train,
        cols=cols,
        blocks=blocks,
        fit_seconds=elapsed,
    )


def save_artifact(artifact: Artifact, out: Path) -> Path:
    """Pickle one fitted arm into ``out``, without its shared banks.

    The encoder's ``SHARED_BANKS`` (answer and question embeddings cut from the slice)
    are detached from a shallow copy; :func:`load_artifact` re-attaches them from the
    slice. The write is atomic (temporary file, then rename).
    """
    path = out / artifact_name(artifact.arm, artifact.seed)
    encoder = artifact.encoder
    banks: dict[str, dict[str, Any]] = {}
    if type(encoder).SHARED_BANKS:
        encoder = copy.copy(encoder)
        for name, slice_attr in type(encoder).SHARED_BANKS.items():
            bank = getattr(encoder, name, None)
            if bank is None:
                continue
            banks[name] = {
                "slice_attr": slice_attr,
                "shape": tuple(int(n) for n in bank.shape),
                "dtype": str(bank.dtype),
            }
            setattr(encoder, name, None)
    payload = {
        "encoder": encoder,
        "arm": artifact.arm,
        "seed": artifact.seed,
        "dim": artifact.dim,
        "input_width": artifact.input_width,
        "train_rows": artifact.train_rows,
        "cols": artifact.cols,
        "blocks": artifact.blocks,
        "fit_seconds": artifact.fit_seconds,
        "banks": banks,
    }
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)
    return path


def load_artifact(path: Path, sl: Slice | None = None) -> Artifact:
    """Read back one artifact, re-attaching its shared banks from ``sl``."""
    payload = pickle.loads(Path(path).read_bytes())
    banks = payload.pop("banks", {})
    encoder = payload["encoder"]
    for name, spec in banks.items():
        if sl is None:
            raise ValueError(
                f"{Path(path).name} was written without its {name} and no slice was "
                "passed to re-attach it from. Pass the slice the run was fitted on — "
                f"the manifest's slice_sha256 names it. Reading it without {name} would "
                f"score {payload['arm']} as an arm that cannot read text."
            )
        bank = getattr(sl, spec["slice_attr"], None)
        if bank is None:
            raise ValueError(
                f"{Path(path).name} needs {name}, but the slice carries no "
                f"{spec['slice_attr']} — that is a different cut, or one built without "
                "an answer bank"
            )
        if tuple(int(n) for n in bank.shape) != tuple(spec["shape"]):
            raise ValueError(
                f"{Path(path).name} was fitted against {spec['slice_attr']} of shape "
                f"{tuple(spec['shape'])} and this slice's is {tuple(bank.shape)} — "
                "different cuts. Every row of that bank would be paired with the wrong "
                "model."
            )
        if str(bank.dtype) != spec["dtype"]:
            # not fatal: the same bank can arrive as float16 or float32
            logger.warning(
                f"{Path(path).name}: {name} was fitted at {spec['dtype']} and the slice "
                f"supplies {bank.dtype}"
            )
        setattr(encoder, name, bank)
    return Artifact(**payload)


def interview_text_block(
    sl: Slice, interview: np.ndarray, train_rows: np.ndarray
) -> np.ndarray | None:
    """The ``[M_all, d]`` mean answer embedding over the K interview questions only.

    Models with no answer text on the interview get the mean of the training rows.
    """
    if sl.Ae is None or sl.Ae_mask is None:
        return None
    table, has_text = pooled_model_text(sl.Ae, sl.Ae_mask, cols=interview)
    donors = np.asarray(train_rows)[has_text[np.asarray(train_rows)]]
    if donors.size == 0:
        logger.warning("no training model has text on the interview — interview text block absent")
        return None
    table[~has_text] = table[donors].mean(axis=0)
    logger.warning(
        f"interview text block: {int((~has_text).sum())} of {has_text.size} models have no "
        f"answer text on the {np.asarray(interview).size} interview questions "
        f"({int((~has_text[np.asarray(train_rows)]).sum())} of them training rows); "
        "filled with the training-row mean"
    )
    return table


def model_text_block(sl: Slice, blocks: QuestionBlocks) -> np.ndarray | None:
    """The ``[M_all, d]`` mean answer embedding over the pool block (``text_mean768``'s table).

    ``None`` when the slice has no answer bank or no coverage mask.
    """
    if sl.Ae is None:
        return None
    if sl.Ae_mask is None:
        logger.warning(
            "answer bank carries no coverage mask, so the public text block cannot be "
            "pooled — skipping it"
        )
        return None
    table, has_text = pooled_model_text(sl.Ae, sl.Ae_mask, cols=blocks.pool_idx)
    logger.info(
        f"public text block: {int(has_text.sum())} of {has_text.size} models pooled from "
        f"the answer bank over {blocks.pool_idx.size} pool columns"
    )
    return table


def train(
    sl: Slice,
    split: Split,
    out: Path,
    cfg: TrainConfig,
    blocks: QuestionBlocks,
    split_seed: int,
    arms: Sequence[str] | None = None,
    seeds: tuple[int, ...] | None = None,
    llmmap_bank_path: Path | None = None,
) -> Outcome:
    """Fit every selected arm at every seed, writing one artifact each.

    ``llmmap_bank_path`` is comp_llmmap_orig's probe bank (beside the slice); ``None``
    or an absent file leaves that arm unbuilt, with a warning.
    """
    wanted = resolve_arms(arms)
    seeds = cfg.seeds if seeds is None else seeds
    text_table = model_text_block(sl, blocks)
    mpnet_bridge = (
        load_mpnet_bridge(sl.question_ids)
        if wanted is None or {"comp_embedllm", "comp_locus", *IRT_FAMILY_ARMS} & set(wanted)
        else None
    )
    out.mkdir(parents=True, exist_ok=True)
    outcome = Outcome()

    def columns_for(arm: str) -> np.ndarray:
        """The block ``arm`` is fitted on, drawing the interview if it is the first to ask."""
        if arm_fit_block(arm) != "interview":
            return blocks.pool_idx
        if outcome.interview is None:
            outcome.interview = select_interview(sl, split, blocks, cfg, split_seed)
        return outcome.interview

    llmmap_probe_bank = (
        aligned_probe_tables(sl.model_ids, path=llmmap_bank_path)
        if llmmap_bank_path is not None and (wanted is None or "comp_llmmap_orig" in wanted)
        else None
    )
    interview_text_table = None
    if wanted is not None and set(INTERVIEW_TEXT_ARMS) & set(wanted):
        if outcome.interview is None:
            outcome.interview = select_interview(sl, split, blocks, cfg, split_seed)
        interview_text_table = interview_text_block(sl, outcome.interview, split.train)

    for i, seed in enumerate(seeds):
        # n_models is the FULL model axis and rows are the training ones: the
        # table-lookup arms index their table by global row.
        encoders = build_encoders(
            n_models=sl.n_models,
            text_table=text_table,
            rows=split.train,
            seed=seed,
            dim=cfg.dim,
            subset=wanted,
            q_text_table=sl.Qe,
            q_text_table_mpnet=mpnet_bridge,
            answer_table=sl.Ae,
            answer_mask=sl.Ae_mask,
            interview_text_table=interview_text_table,
            llmmap_probe_bank=llmmap_probe_bank,
        )
        requested = list(ENCODER_ORDER) if wanted is None else wanted
        if i == 0:
            outcome.skipped.extend(
                Skip(arm=arm, reason="input block absent — not buildable from this slice")
                for arm in requested
                if arm not in encoders
            )
        for arm in (a for a in requested if a in encoders):
            try:
                artifact = fit_one(
                    arm, encoders[arm], sl, split, columns_for(arm), arm_blocks(arm), seed
                )
            except Exception as exc:  # noqa: BLE001  (one arm must not sink the run)
                logger.exception(f"{arm} s{seed} failed to fit: {exc}")
                outcome.skipped.append(
                    Skip(arm=arm, seed=seed, reason=f"{type(exc).__name__}: {exc}")
                )
                continue
            save_artifact(artifact, out)
            park_fitted_modules(artifact.encoder)
            outcome.artifacts.append(artifact)
    return outcome


def config_snapshot(
    cfg: TrainConfig, splits_cfg: SplitsConfig | None, run_config: str | None
) -> dict[str, Any]:
    """What the run was asked for: the training and splits configs and the run config."""
    return {
        "train": asdict(cfg),
        "splits": None if splits_cfg is None else asdict(splits_cfg),
        "run_config": run_config,
    }


def write_manifest(
    outcome: Outcome,
    out: Path,
    sl: Slice,
    split: Split,
    slice_path: Path,
    cfg: TrainConfig,
    seeds: tuple[int, ...],
    blocks: QuestionBlocks,
    total_seconds: float,
    arms_requested: list[str] | None = None,
    fold: tuple[FoldPlan, int] | None = None,
    fold_dir: Path = DEFAULT_FOLD_DIR,
    splits_cfg: SplitsConfig | None = None,
    answers_path: Path | None = None,
    run_config: str | None = None,
    view: str | None = None,
    held_out: tuple[str, np.ndarray] | None = None,
) -> Path:
    """Describe the fitted directory: provenance, the question blocks, the split, the arms.

    ``eval.py`` reads the eval block, the held-out rows and the arm list from here
    rather than re-deriving them.
    """
    interview = outcome.interview
    if interview is None:
        adopted = [a for a in outcome.artifacts if "interview" in a.blocks]
        interview = None if not adopted else np.asarray(adopted[0].cols, dtype=int)
    plan, fold_index = (None, None) if fold is None else fold
    plan_file = None if plan is None else plan_path(plan.scheme, plan.k, plan.seed, fold_dir)
    fold_block = (
        None
        if plan is None
        else {
            "scheme": plan.scheme,
            "k": plan.k,
            "index": int(fold_index),
            "seed": plan.seed,
            "plan_path": str(plan_file),
            "plan_sha256": file_digest(plan_file) if plan_file.is_file() else None,
            "n_always_train": plan.n_always_train,
            "n_excluded": plan.n_excluded,
        }
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "config_sha256": config_digest(),
        "slice_path": str(slice_path),
        "slice_sha256": file_digest(slice_path),
        # the slice view (e.g. core); every column index below is on it
        "view": view,
        # the answer bank the text arms were fitted on; eval.py reads the same one
        "answers_path": None if answers_path is None else str(answers_path),
        # the benchmark group removed from pool and tune (transfer refits)
        "hold_out_group": None if held_out is None else held_out[0],
        "held_out_columns": None if held_out is None else [int(c) for c in held_out[1]],
        "cpu_threads": CPU_THREADS,
        "arms_requested": arms_requested,
        "seeds": list(seeds),
        "dim": cfg.dim,
        "n_models": sl.n_models,
        "n_questions": sl.n_questions,
        "pca95_rows": PCA95_ROWS,
        "questions": {
            "block_seed": blocks.seed,
            "n_eval": blocks.n_eval,
            "n_tune": blocks.n_tune,
            "n_pool": blocks.n_pool,
            "n_interview": None if interview is None else int(interview.size),
            "interview_k": cfg.interview_k,
            "interview_selector": cfg.interview_selector,
            "eval": [int(c) for c in blocks.eval_idx],
            "tune": [int(c) for c in blocks.tune_idx],
            "pool": [int(c) for c in blocks.pool_idx],
            "interview": None if interview is None else [int(c) for c in interview],
        },
        "split": {
            "source": split.source,
            "n_train": split.n_train,
            "n_test": split.n_test,
            "train": [int(i) for i in split.train],
            "test": [int(i) for i in split.test],
            "train_model_ids": [sl.model_ids[i] for i in split.train],
            "test_model_ids": [sl.model_ids[i] for i in split.test],
            "fold": fold_block,
        },
        "fold_of": None if plan is None else plan.as_dict()["fold_of"],
        "arms": [a.meta() for a in outcome.artifacts],
        "skipped": [asdict(s) for s in outcome.skipped],
        "total_seconds": round(total_seconds, 3),
        "config_snapshot": config_snapshot(cfg, splits_cfg, run_config),
    }
    path = out / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    logger.info(f"wrote {path}")
    return path
