"""Cached Hugging Face model cards (``data/raw/hf_model_cards/<model>.json``), one per model."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from bullpen.data.external_labels import RAW_DIR

DEFAULT_CARD_DIR = RAW_DIR / "hf_model_cards"

#: Every run of non-alphanumerics is one segment boundary.
_SEPARATORS = re.compile(r"[^a-z0-9]+")


def segments(text: str) -> str:
    """``text`` lower-cased with every run of non-alphanumerics folded to a dash."""
    return _SEPARATORS.sub("-", str(text).lower()).strip("-")


def card_path(model_key: str, card_dir: Path = DEFAULT_CARD_DIR) -> Path:
    """Where one model's card document lives."""
    return Path(card_dir) / f"{model_key}.json"


def read_card(model_key: str, card_dir: Path = DEFAULT_CARD_DIR) -> dict[str, Any] | None:
    """One cached document, or ``None`` when there is none."""
    path = card_path(model_key, card_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"{path} is malformed and is treated as missing")
        return None


def read_cards(
    model_ids: Sequence[str], card_dir: Path = DEFAULT_CARD_DIR
) -> dict[str, dict[str, Any]]:
    """``{model key: document}`` for the keys that have one, in ``model_ids`` order."""
    out = {}
    for model_key in model_ids:
        card = read_card(model_key, card_dir)
        if card is not None:
            out[str(model_key)] = card
    return out
