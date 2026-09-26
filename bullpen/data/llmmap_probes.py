"""The LLMmap probe bank: our models' embedded answers to LLMmap's own 8 published probes.

LLMmap (Pasquini, Kornaropoulos & Ateniese, USENIX Security 2025, arXiv 2407.15847)
fingerprints a model from its response text to 8 fixed queries (upstream
``confs/queries/default.json``, vendored in ``bullpen/competitors/llmmap/``).
``comp_llmmap_orig`` reads the embedded responses from ``llmmap_probe_bank.npz`` beside
the slice: ``E`` [M, 8, d] responses, ``mask`` [M, 8], ``Q`` [8, d] prompts.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

#: The bank's file name beside the slice.
LLMMAP_BANK_FILE = "llmmap_probe_bank.npz"


class LLMmapProbeError(RuntimeError):
    """The probe bank is missing."""


def load_probe_bank(path: Path) -> dict[str, Any]:
    """Read a probe bank ``.npz``."""
    path = Path(path)
    if not path.is_file():
        raise LLMmapProbeError(f"no LLMmap probe bank at {path}")
    with np.load(path, allow_pickle=True) as z:
        out = {
            "E": np.asarray(z["E"], dtype=np.float32),
            "mask": np.asarray(z["mask"], dtype=bool),
            "Q": np.asarray(z["Q"], dtype=np.float32),
            "model_ids": [str(m) for m in z["model_ids"]],
            "probe_ids": [str(p) for p in z["probe_ids"]],
            "prompt_sha256": str(z["prompt_sha256"]) if "prompt_sha256" in z.files else "",
        }
    return out


def aligned_probe_tables(
    model_ids: Sequence[str],
    path: Path | None = None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """The probe bank on a given model axis, or ``None`` when there is no bank.

    ``E`` is ``[M, 8, d]`` and ``mask`` ``[M, 8]`` over ``model_ids`` in their own
    order, zero-filled for a model the bank has no row for. ``Q`` is ``[8, d]``.
    ``required=True`` raises instead of returning ``None``.
    """
    try:
        if path is None:
            raise LLMmapProbeError("no LLMmap probe bank path given")
        bank = load_probe_bank(path)
    except LLMmapProbeError:
        if required:
            raise
        return None
    pos = {m: i for i, m in enumerate(bank["model_ids"])}
    n_p, d = bank["Q"].shape
    E = np.zeros((len(model_ids), n_p, d), dtype=np.float32)
    mask = np.zeros((len(model_ids), n_p), dtype=bool)
    for i, model_id in enumerate(model_ids):
        j = pos.get(model_id)
        if j is None:
            continue
        E[i] = bank["E"][j]
        mask[i] = bank["mask"][j]
    covered = int(mask.any(axis=1).sum())
    logger.info(
        f"LLMmap probe bank: {covered}/{len(model_ids)} model(s) on this axis have probe "
        f"responses ({mask.mean():.1%} of cells), prompts {bank['prompt_sha256'][:12]}"
    )
    return {"E": E, "mask": mask, "Q": np.asarray(bank["Q"], dtype=np.float32)}
