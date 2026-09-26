"""Graded answers in, the slice and its embedding banks out (library for ``process.py``).

``build``: fold the per-model ``results.jsonl`` files into one model x question cut
(:func:`bullpen.data.assemble.assemble`: bad-slug patterns and the assembly coverage
gate of ``config/collect.yaml``), apply the two quality gates of
``config/process.yaml`` (the coverage floor drops a model that answered too little of
the question axis; a model with zero valid outputs is dropped; a survivor at or below
``degenerate_accuracy`` is flagged), write every uncollected cell of a survivor as
incorrect so the slice is finite, and cut the question axis into the disjoint
eval / tune / pool blocks (:mod:`bullpen.data.blocks`).

``embed``: one L2-normalised embedding per (model, question) answer (``Ae`` with its
coverage mask ``Ae_mask``) and per question (``Qe``), under the shared text policy of
``config/text_banks.yaml`` (the paper: BAAI/bge-base-en-v1.5, attention-mask mean
pooling, left truncation to 512 tokens). :class:`HashingEncoder` is an offline
stand-in for tests and the sample data only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.data.assemble import Substrate, load_results, responses_path
from bullpen.data.enrichment import (
    DEFAULT_FOOTPRINTS,
    DEFAULT_RELEASE_DATES,
    dates_for_models,
    load_footprints,
    load_release_dates,
    vram_for_models,
)
from bullpen.data.slice import DEGENERATE_ACCURACY, Slice


def row_accuracy(R: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """[M] accuracy over the cells a model answered (0.0 for a row with none)."""
    answered = covered.sum(axis=1)
    return np.divide(
        np.where(covered, R, 0).sum(axis=1),
        answered,
        out=np.zeros(R.shape[0], dtype=np.float64),
        where=answered > 0,
    )


def quality_report(
    sub: dict[str, np.ndarray],
    coverage_floor: float,
    degenerate_accuracy: float = DEGENERATE_ACCURACY,
) -> dict[str, Any]:
    """Apply the coverage floor and the zero-output gate, then flag suspicious survivors.

    A model below ``coverage_floor`` or with zero correct answers is dropped; a survivor
    at or below ``degenerate_accuracy`` (or with mostly empty responses) is flagged.
    Returns ``kept`` row indices, ``dropped`` and ``flagged`` with reasons.
    """
    models = [str(m) for m in sub["models"]]
    covered = sub["covered"]
    coverage = covered.mean(axis=1)
    accuracy = row_accuracy(sub["R"], covered)
    n_rows = sub.get("n_rows")
    n_empty = sub.get("n_empty")

    kept: list[int] = []
    dropped: list[dict[str, Any]] = []
    for i, name in enumerate(models):
        if coverage[i] >= coverage_floor:
            kept.append(i)
            continue
        reason = (
            "no responses collected"
            if coverage[i] == 0.0
            else f"coverage {coverage[i]:.2%} below the {coverage_floor:.0%} floor"
        )
        dropped.append(
            {
                "model": name,
                "coverage": float(coverage[i]),
                "accuracy": float(accuracy[i]),
                "reason": reason,
            }
        )

    # zero-output gate: attempted every question, never produced a correct answer
    zero_output: list[int] = []
    for i in kept:
        if accuracy[i] == 0.0:
            dropped.append(
                {
                    "model": models[i],
                    "coverage": float(coverage[i]),
                    "accuracy": 0.0,
                    "reason": "zero valid outputs across all covered cells",
                }
            )
            zero_output.append(i)
    if zero_output:
        kept = [i for i in kept if i not in set(zero_output)]

    flagged: list[dict[str, Any]] = []
    for i in kept:
        reasons: list[str] = []
        if n_rows is not None and int(n_rows[i]) == 0:
            reasons.append("no rows in the response logs")
        if n_rows is not None and n_empty is not None and int(n_rows[i]) > 0:
            empty = int(n_empty[i]) / int(n_rows[i])
            if empty == 1.0:
                reasons.append("every response was empty")
            elif empty > 0.5:
                reasons.append(f"{empty:.1%} of responses were empty")
        if accuracy[i] == 0.0:
            reasons.append("zero accuracy over every answered cell")
        elif accuracy[i] <= degenerate_accuracy:
            reasons.append(f"accuracy {accuracy[i]:.2%} at or below the degenerate-accuracy gate")
        if reasons:
            flagged.append(
                {
                    "model": models[i],
                    "coverage": float(coverage[i]),
                    "accuracy": float(accuracy[i]),
                    "reasons": reasons,
                }
            )

    return {
        "coverage_floor": coverage_floor,
        "degenerate_accuracy": degenerate_accuracy,
        "n_models_in": len(models),
        "n_models_kept": len(kept),
        "overall_coverage": float(covered.mean()),
        "min_kept_coverage": float(coverage[kept].min()) if kept else 0.0,
        "uncovered_cells_written_as_incorrect": int((~covered[kept]).sum()) if kept else 0,
        "kept": kept,
        "dropped": dropped,
        "flagged": flagged,
    }


def log_quality_report(report: dict[str, Any]) -> None:
    """The QC report as lines in the run log, not only as a file."""
    logger.info(
        f"coverage floor {report['coverage_floor']:.0%}: kept "
        f"{report['n_models_kept']} of {report['n_models_in']} model(s), "
        f"{report['uncovered_cells_written_as_incorrect']} uncovered cell(s) "
        f"written as incorrect"
    )
    for row in report["dropped"]:
        logger.warning(
            f"dropped {row['model']}: coverage {row['coverage']:.2%}, "
            f"accuracy {row['accuracy']:.2%} — {row['reason']}"
        )
    for row in report["flagged"]:
        logger.warning(
            f"flagged {row['model']} (kept): coverage {row['coverage']:.2%}, "
            f"accuracy {row['accuracy']:.2%} — {'; '.join(row['reasons'])}"
        )
    if not report["dropped"]:
        logger.info("no model fell below the coverage floor")
    if not report["flagged"]:
        logger.info("no surviving model looks like a collection failure")


def subset_source(sub: dict[str, np.ndarray], kept: list[int]) -> dict[str, np.ndarray]:
    """Take the kept model rows, leaving the question axis alone."""
    rows = np.asarray(kept, dtype=int)
    out = dict(sub)
    for key in (
        "R",
        "covered",
        "models",
        "n_rows",
        "n_empty",
        "text_sha1",
        "rule",
        "latest_row_disagrees",
    ):
        if key in sub:
            out[key] = sub[key][rows]
    return out


def degenerate_models(
    R: np.ndarray, covered: np.ndarray, threshold: float = DEGENERATE_ACCURACY
) -> np.ndarray:
    """Row indices whose accuracy over answered cells is at or below ``threshold``."""
    return np.flatnonzero(row_accuracy(R, covered) <= threshold)


def build_slice(
    sub: dict[str, np.ndarray],
    manifest: dict,
    footprints_path: Path | None = DEFAULT_FOOTPRINTS,
    dates_path: Path | None = DEFAULT_RELEASE_DATES,
) -> Slice:
    """Source arrays to a :class:`Slice`, with benchmarks indexed by name.

    Unanswered cells are written as incorrect (``covered`` records which they are);
    ``vram_gb`` and ``release_date`` come from the metadata files, NaN / "" without them.
    """
    bench_names = sorted(set(sub["axes"].tolist()))
    index = {name: i for i, name in enumerate(bench_names)}
    bench = np.array([index[a] for a in sub["axes"]], dtype=np.int32)

    R = np.where(sub["covered"], sub["R"], 0).astype(np.uint8)

    model_ids = sub["models"].tolist()
    n_models = R.shape[0]
    if footprints_path is not None and footprints_path.exists():
        vram = vram_for_models(model_ids, load_footprints(footprints_path))
    else:
        logger.warning(f"no footprint file at {footprints_path} — vram_gb stays all-NaN")
        vram = np.full(n_models, np.nan, dtype=np.float64)
    if dates_path is not None and dates_path.exists():
        dates = dates_for_models(model_ids, load_release_dates(dates_path))
    else:
        logger.warning(f"no date file at {dates_path} — release_date stays empty")
        dates = [""] * n_models

    return Slice(
        A=R.astype(np.float32),
        R=R,
        bench=bench,
        bench_names=bench_names,
        model_ids=model_ids,
        question_ids=sub["questions"].tolist(),
        vram_gb=vram,
        release_date=dates,
        split=[""] * n_models,
        # R is 0 both for a wrong and for a missing answer; this mask tells them apart
        covered=np.asarray(sub["covered"], dtype=bool),
        manifest=manifest,
    )


def source_arrays(sub: Substrate) -> dict[str, np.ndarray]:
    """The assembled cut in the layout the gates read: ``R``, ``covered`` and the axes."""
    return {
        "R": sub.R.astype(np.uint8),
        "covered": np.isfinite(sub.A),
        "models": np.array(sub.models),
        "questions": np.array(sub.questions),
        "axes": np.array(sub.axes),
    }


def gate_and_build(
    sub: Substrate,
    coverage_floor: float,
    degenerate_accuracy: float = DEGENERATE_ACCURACY,
    drop_degenerate: bool = False,
    footprints: Path | None = DEFAULT_FOOTPRINTS,
    release_dates: Path | None = DEFAULT_RELEASE_DATES,
) -> tuple[Slice, dict[str, Any]]:
    """The quality gates, then the slice. Returns ``(slice, QC report)``."""
    src = source_arrays(sub)
    report = quality_report(src, coverage_floor, degenerate_accuracy)
    log_quality_report(report)
    if not report["kept"]:
        raise ValueError(f"no model cleared the {coverage_floor:.0%} coverage floor")
    src = subset_source(src, report["kept"])
    if drop_degenerate:
        bad = degenerate_models(src["R"], src["covered"], degenerate_accuracy)
        if bad.size:
            report["dropped_degenerate"] = [str(src["models"][i]) for i in bad]
            src = subset_source(src, [i for i in range(src["R"].shape[0]) if i not in set(bad)])
    manifest = {"excluded": sub.excluded, "near_duplicates": sub.near_duplicates}
    report["kept"] = [str(m) for m in src["models"]]
    return build_slice(src, manifest, footprints, release_dates), report


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
class HashingEncoder:
    """Offline stand-in for the sentence encoder: hashed word/bigram counts, a fixed
    Gaussian projection to ``dim``, L2-normalised. For tests and the sample data only;
    the paper's banks are :class:`bullpen.data.text_banks.TextEncoder` (BGE)."""

    def __init__(self, dim: int, seed: int = 0) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = int(dim)
        self.vec = HashingVectorizer(n_features=2**12, ngram_range=(1, 2), alternate_sign=False)
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((2**12, self.dim)).astype(np.float32) / np.sqrt(self.dim)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        x = np.asarray(self.vec.transform(texts) @ self.proj, dtype=np.float32)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def make_encoder(spec: str, device: str | None = None):
    """``hashing:<dim>`` (offline) or ``hf`` (config/text_banks.yaml's encoder)."""
    if spec.startswith("hashing:"):
        return HashingEncoder(int(spec.split(":", 1)[1]))
    if spec != "hf":
        raise ValueError(f"unknown encoder {spec!r}: use 'hf' or 'hashing:<dim>'")
    from bullpen.config import load_text_banks
    from bullpen.data.text_banks import TextEncoder

    return TextEncoder(load_text_banks(), device=device)


def answer_banks(
    raw: Path,
    model_ids: list[str],
    question_ids: list[str],
    question_text: dict[str, str],
    encoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Ae [M, Q, D] float16, Ae_mask [M, Q], Qe [Q, D] float32)`` for one slice."""
    col = {q: j for j, q in enumerate(question_ids)}
    dim = int(encoder.dim)
    Ae = np.zeros((len(model_ids), len(question_ids), dim), dtype=np.float16)
    mask = np.zeros((len(model_ids), len(question_ids)), dtype=bool)
    for i, m in enumerate(model_ids):
        rows = load_results(responses_path(raw, m))
        cells = [(col[q], r.get("prediction") or "") for q, r in rows.items() if q in col]
        cells = [(j, t) for j, t in cells if t.strip()]
        if not cells:
            logger.warning(f"{m}: no answer text to embed")
            continue
        emb = encoder.encode([t for _, t in cells])
        idx = np.array([j for j, _ in cells])
        Ae[i, idx] = emb.astype(np.float16)
        mask[i, idx] = True
    Qe = encoder.encode([question_text.get(q, "") for q in question_ids]).astype(np.float32)
    logger.info(
        f"answer bank {Ae.shape} ({mask.mean():.1%} of cells embedded), question bank {Qe.shape}"
    )
    return Ae, mask, Qe


def save_banks(path: Path, Ae: np.ndarray, mask: np.ndarray, Qe: np.ndarray) -> None:
    """Write ``answers.npz`` in the layout :func:`bullpen.data.slice.load_slice` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Ae=Ae.astype(np.float16), Ae_mask=mask.astype(bool), Qe=Qe.astype(np.float32))
    logger.info(f"wrote {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
