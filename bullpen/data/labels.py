"""Where a cut's model-axis label tables live, and one loader for them.

``external_labels.json`` and ``external_signals.json`` (published board scores),
``text_traits.npz`` (answer statistics) and ``group_h_labels.npz`` (reward-model scores)
live in ``labels/`` beside the slice, else in ``data/raw/``. A model a table does not
cover reads as NaN; :func:`load_labels` logs each table's coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from loguru import logger

from bullpen.config import RAW_DIR as _RAW_DIR
from bullpen.data.external_labels import DEFAULT_EXTERNAL_LABELS, load_external_labels
from bullpen.data.group_h_labels import DEFAULT_GROUP_H_LABELS, GroupHLabels, load_group_h_labels
from bullpen.data.text_traits import DEFAULT_TEXT_TRAITS, TextTraits, load_text_traits

RAW_DIR = _RAW_DIR
EXTERNAL_LABELS_NAME = DEFAULT_EXTERNAL_LABELS.name
EXTERNAL_SIGNALS_NAME = "external_signals.json"
TEXT_TRAITS_NAME = DEFAULT_TEXT_TRAITS.name
GROUP_H_LABELS_NAME = DEFAULT_GROUP_H_LABELS.name

#: The subdirectory beside a slice that holds its own label projections.
LABELS_DIRNAME = "labels"

LABEL_FILES: tuple[str, ...] = (EXTERNAL_LABELS_NAME, TEXT_TRAITS_NAME, GROUP_H_LABELS_NAME)


def labels_root_for(slice_path: Path | str | None) -> Path:
    """``<slice dir>/labels`` when it exists, else ``data/raw``."""
    if slice_path is not None:
        beside = Path(slice_path).resolve().parent / LABELS_DIRNAME
        if beside.is_dir():
            return beside
    return RAW_DIR


def load_labels(
    model_ids: Sequence[str], root: Path | str
) -> tuple[dict[str, np.ndarray], TextTraits | None, GroupHLabels | None]:
    """The label tables from ``root``, with each one's coverage of ``model_ids`` logged."""
    root = Path(root)
    external = load_external_labels(model_ids, root / EXTERNAL_LABELS_NAME)
    # the second board file shares the layout; table names are disjoint
    signals = load_external_labels(model_ids, root / EXTERNAL_SIGNALS_NAME)
    clash = set(external) & set(signals)
    if clash:
        raise ValueError(f"{sorted(clash)} are both a leaderboard and a scraped-board table")
    external = {**external, **signals}
    traits = load_text_traits(model_ids, root / TEXT_TRAITS_NAME)
    group_h = load_group_h_labels(model_ids, root / GROUP_H_LABELS_NAME)
    n = len(model_ids)
    parts = []
    if external:
        labelled = np.isfinite(np.vstack(list(external.values()))).any(axis=0)
        parts.append(f"external boards {int(labelled.sum())}/{n}")
    if traits is not None:
        parts.append(f"text traits {int(np.isfinite(traits.traits).any(axis=1).sum())}/{n}")
    if group_h is not None:
        parts.append(f"reward-model labels {int(group_h.covered().sum())}/{n}")
    missing = [
        name
        for name, obj in zip(LABEL_FILES, (external or None, traits, group_h), strict=True)
        if obj is None
    ]
    logger.info(
        f"labels from {root}: {', '.join(parts) or 'none'}"
        + (f"; absent: {missing}" if missing else "")
    )
    return external, traits, group_h
