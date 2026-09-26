"""Published leaderboard scores joined onto the model axis (``external_labels.json``).

Names are joined by :func:`match_key` (org prefix, case and punctuation folded), then by the
hand-verified ``model_aliases.json``, never fuzzily. A model a board does not list is NaN.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from bullpen.config import RAW_DIR as _RAW_DIR

RAW_DIR = _RAW_DIR
DEFAULT_EXTERNAL_LABELS = RAW_DIR / "external_labels.json"

#: Keys of the stored documents.
SCORES_KEY = "scores"
ALIASES_KEY = "aliases"
TABLES_KEY = "tables"

#: Filename of the hand-verified alias map, inside the source directory.
ALIASES_NAME = "model_aliases.json"

#: Everything that is not a letter or a digit is dropped before comparing names.
_PUNCTUATION = re.compile(r"[^a-z0-9]")


class ExternalLabelError(RuntimeError):
    """A source table is malformed."""


def match_key(name: str) -> str:
    """The join key: a model name stripped of its org prefix, its case and its punctuation."""
    return _PUNCTUATION.sub("", str(name).replace("/", "__").split("__")[-1].lower())


def read_aliases(path: Path) -> dict[str, str]:
    """``{published key: slice key}`` from the hand-verified alias map; ``{}`` if absent."""
    if not path.exists():
        logger.warning(f"no alias map at {path}; the join is exact-match only")
        return {}
    raw = json.loads(path.read_text())
    entries = raw.get(ALIASES_KEY, raw)
    if not isinstance(entries, Mapping):
        raise ExternalLabelError(f"{path} has no {ALIASES_KEY!r} object")
    return {match_key(k): match_key(v) for k, v in entries.items()}


@dataclass(frozen=True)
class RowMatch:
    """Which upstream name each slice model was matched to, and how.

    ``source_of`` is positional on the model ids and ``None`` where nothing upstream
    claimed the model.
    """

    source_of: tuple[str | None, ...]
    via_alias: tuple[bool, ...]
    n_published: int
    unmatched_models: tuple[str, ...]
    unmatched_published: tuple[str, ...]

    @property
    def n_exact(self) -> int:
        return sum(
            1
            for name, alias in zip(self.source_of, self.via_alias, strict=True)
            if name and not alias
        )

    @property
    def n_alias(self) -> int:
        return sum(
            1 for name, alias in zip(self.source_of, self.via_alias, strict=True) if name and alias
        )

    @property
    def n_matched(self) -> int:
        return sum(1 for name in self.source_of if name is not None)

    def provenance(self) -> dict[str, int]:
        """The match counts recorded beside the joined values."""
        return {
            "n_published": self.n_published,
            "n_exact": self.n_exact,
            "n_alias": self.n_alias,
            "n_matched": self.n_matched,
        }


def match_rows(
    model_ids: Sequence[str],
    published: Iterable[str],
    aliases: Mapping[str, str],
) -> RowMatch:
    """Match every slice model to an upstream name: exact first, then alias, never fuzzy.

    Both sides are folded by :func:`match_key`; ``source_of`` carries the original
    upstream spelling. When two upstream names fold to one key, the first wins.
    """
    by_key: dict[str, str] = {}
    for name in published:
        by_key.setdefault(match_key(name), str(name))
    aliased = {slice_key: by_key[pub] for pub, slice_key in aliases.items() if pub in by_key}

    source_of: list[str | None] = []
    via_alias: list[bool] = []
    claimed: set[str] = set()
    unmatched: list[str] = []
    for model_id in model_ids:
        key = match_key(model_id)
        if key in by_key:
            source_of.append(by_key[key])
            via_alias.append(False)
            claimed.add(by_key[key])
        elif key in aliased:
            source_of.append(aliased[key])
            via_alias.append(True)
            claimed.update(
                by_key[pub] for pub, sl in aliases.items() if sl == key and pub in by_key
            )
        else:
            source_of.append(None)
            via_alias.append(False)
            unmatched.append(str(model_id))
    return RowMatch(
        source_of=tuple(source_of),
        via_alias=tuple(via_alias),
        n_published=len(by_key),
        unmatched_models=tuple(unmatched),
        unmatched_published=tuple(sorted(set(by_key.values()) - claimed)),
    )


def load_external_labels(
    model_ids: Sequence[str], path: Path = DEFAULT_EXTERNAL_LABELS
) -> dict[str, np.ndarray]:
    """``{table: [M] published score}`` for ``model_ids``, NaN where unlisted; ``{}`` if absent."""
    path = Path(path)
    if not path.exists():
        logger.warning(
            f"no external label projection at {path}; the leaderboard tasks have no target"
        )
        return {}
    tables = json.loads(path.read_text()).get(TABLES_KEY, {})
    out: dict[str, np.ndarray] = {}
    for name, block in tables.items():
        scores = block.get(SCORES_KEY, {})
        out[name] = np.array([float(scores.get(m, np.nan)) for m in model_ids])
    return out
