"""Fold the per-model answer files ``collect.py`` writes into one gated model x question cut.

``<raw>/responses/<model>/results.jsonl`` holds one graded row per question. The gates:
bad-slug patterns, then the coverage gate; near-duplicate model pairs are reported, not
dropped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.config import AssemblyConfig

#: One JSONL per model under this directory.
RESPONSES_DIR = "responses"

#: A cell is correct when it is correct in more than half its draws.
MAJORITY = 0.5

#: ``D`` is uint8; a cell drawn more often than this records the cap.
MAX_RECORDED_DRAWS = 255


def responses_path(raw: Path, model_id: str) -> Path:
    """Where one model's graded answers live."""
    return raw / RESPONSES_DIR / model_id / "results.jsonl"


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """One model's answers so far, keyed by question id."""
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def collected_models(raw: Path) -> list[str]:
    """Every model with an answers file, in sorted order."""
    root = raw / RESPONSES_DIR
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "results.jsonl").is_file())


def model_vectors(
    rows: Mapping[str, Mapping[str, Any]], order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """``(A, D)`` for one model over ``order``: mean correctness (NaN = uncollected) and draws."""
    acc = np.full(len(order), np.nan, dtype=np.float32)
    draws = np.zeros(len(order), dtype=np.uint8)
    for j, qid in enumerate(order):
        row = rows.get(qid)
        if row is None:
            continue
        graded = row["correct"]
        values = [int(v) for v in graded] if isinstance(graded, list) else [int(graded)]
        acc[j] = float(np.mean(values))
        draws[j] = min(len(values), MAX_RECORDED_DRAWS)
    return acc, draws


def near_duplicates(models: Sequence[str], binary: np.ndarray, threshold: float) -> list[dict]:
    """Model pairs whose binary readouts agree at or above ``threshold`` (reported, not dropped)."""
    flagged = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            agreement = float(np.mean(binary[i] == binary[j]))
            if agreement >= threshold:
                flagged.append(
                    {
                        "model_a": models[i],
                        "model_b": models[j],
                        "agreement": round(agreement, 4),
                    }
                )
    flagged.sort(key=lambda d: -d["agreement"])
    return flagged


@dataclass
class Substrate:
    """The assembled cut, plus the manifest describing how it was gated."""

    A: np.ndarray
    R: np.ndarray
    D: np.ndarray
    models: list[str]
    questions: list[str]
    axes: list[str]
    excluded: list[dict] = field(default_factory=list)
    near_duplicates: list[dict] = field(default_factory=list)


def assemble(
    raw: Path,
    questions: Sequence[str],
    axes: Sequence[str],
    cfg: AssemblyConfig,
    model_ids: Sequence[str] | None = None,
) -> Substrate:
    """Fold the per-model answer files into one gated model x question cut."""
    names = list(model_ids) if model_ids is not None else collected_models(raw)
    n_q = len(questions)
    excluded: list[dict] = []

    kept: list[str] = []
    for model_id in names:
        if cfg.is_bad_slug(model_id):
            excluded.append({"slug": model_id, "reason": "bad-download / scaffold slug pattern"})
        else:
            kept.append(model_id)

    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    survivors: list[str] = []
    for model_id in kept:
        rows = load_results(responses_path(raw, model_id))
        acc, draws = model_vectors(rows, questions)
        coverage = float(np.isfinite(acc).mean()) if n_q else 0.0
        if coverage < cfg.min_coverage:
            excluded.append(
                {
                    "slug": model_id,
                    "reason": "coverage below gate",
                    "coverage": round(coverage, 4),
                }
            )
            continue
        vectors[model_id] = (acc, draws)
        survivors.append(model_id)

    survivors.sort()
    A = np.full((len(survivors), n_q), np.nan, dtype=np.float32)
    D = np.zeros((len(survivors), n_q), dtype=np.uint8)
    for i, model_id in enumerate(survivors):
        A[i], D[i] = vectors[model_id]
    R = (np.nan_to_num(A, nan=0.0) > MAJORITY).astype(np.uint8)

    logger.info(
        f"{len(names)} collected model(s): {len(survivors)} pass the "
        f"{cfg.min_coverage:.0%} coverage gate, {len(excluded)} excluded"
    )
    for entry in excluded:
        logger.warning(f"  excluded {entry['slug']}: {entry['reason']}")

    dupes = near_duplicates(survivors, R, cfg.dup_agreement)
    for dupe in dupes:
        logger.warning(
            f"  {dupe['model_a']} ~ {dupe['model_b']} agree {dupe['agreement']:.3f} — "
            f"possible identity leak, kept"
        )
    return Substrate(
        A=A,
        R=R,
        D=D,
        models=survivors,
        questions=list(questions),
        axes=list(axes),
        excluded=excluded,
        near_duplicates=dupes,
    )
