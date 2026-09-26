"""Per-model fields the correctness matrix does not carry: VRAM, release date, size, family.

Read from ``data/raw/model_footprints.json`` (weight bytes, stored and serving dtype,
parameter count, family) and ``data/raw/model_release_dates.json``, keyed by model id.
A model that cannot be resolved gets NaN or "".

Served VRAM (weights only, decimal GB)::

    vram_gb = weight_bytes * bytes_per_param(serve_quant) / bytes_per_param(quant) / 1e9

falling back to ``params_b * 1e9 * bytes_per_param(serve_quant) / 1e9`` when no
checkpoint size was measured. The chat flag is read from the model name.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from bullpen.config import RAW_DIR as _RAW_DIR

RAW_DIR = _RAW_DIR
DEFAULT_FOOTPRINTS = RAW_DIR / "model_footprints.json"
DEFAULT_RELEASE_DATES = RAW_DIR / "model_release_dates.json"
#: Optional per-suite benchmark publication dates (absent in the shipped data).
DEFAULT_BENCHMARK_DATES = RAW_DIR / "benchmark_publication_dates.json"

#: Key holding the per-model block in both files.
MODELS_KEY = "models"

#: Keys of the benchmark-date table.
SUITES_KEY = "suites"
PUBLISHED_KEY = "published"

#: Decimal GB, the unit of the VRAM budgets in ``battery.yaml``.
BYTES_PER_GB = 1e9

#: Parameters per unit of ``params_b``.
PARAMS_PER_B = 1e9

#: Bytes one parameter occupies in each dtype.
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "awq": 0.5,
    "gptq": 0.5,
}

#: Footprint fields read here; ``status`` marks estimated (unmeasured) entries.
WEIGHT_BYTES_KEY = "weight_bytes"
STORED_QUANT_KEY = "quant"
SERVE_QUANT_KEY = "serve_quant"
PARAMS_B_KEY = "params_b"
DIR_KEY = "dir"
HF_ID_KEY = "hf_id"
STATUS_KEY = "status"
FAMILY_KEY = "family"

#: Lower-cased substrings of a model name that declare an instruction tune.
CHAT_NAME_TOKENS: tuple[str, ...] = (
    "instruct",
    "chat",
    "-it",
    "sft",
    "dpo",
    "rlhf",
    "tulu",
    "vicuna",
    "alpaca",
    "orca",
    "hermes",
    "zephyr",
)

#: An ISO 8601 date is its first ten characters.
ISO_DATE_WIDTH = 10


def normalise_model_key(name: str) -> str:
    """The join key for a model name: bare repo name (org dropped), lower-cased."""
    return name.rsplit("/", 1)[-1].rsplit("__", 1)[-1].strip().lower()


def bytes_per_param(dtype: str, widths: Mapping[str, float] | None = None) -> float:
    """Bytes one parameter occupies when stored or served as ``dtype``."""
    table = BYTES_PER_PARAM if widths is None else widths
    try:
        return table[dtype.strip().lower()]
    except (AttributeError, KeyError):
        raise ValueError(f"unknown dtype {dtype!r}; known widths are {sorted(table)}") from None


def footprint_vram_gb(
    entry: Mapping[str, Any],
    widths: Mapping[str, float] | None = None,
    bytes_per_gb: float = BYTES_PER_GB,
) -> float:
    """Served VRAM of one footprint entry in GB (module docstring), or NaN."""
    served = entry.get(SERVE_QUANT_KEY)
    if not served:
        return float("nan")
    served_width = bytes_per_param(served, widths)

    measured = entry.get(WEIGHT_BYTES_KEY)
    stored = entry.get(STORED_QUANT_KEY)
    if measured and stored:
        return float(measured) * served_width / bytes_per_param(stored, widths) / bytes_per_gb

    params_b = entry.get(PARAMS_B_KEY)
    if params_b:
        return float(params_b) * PARAMS_PER_B * served_width / bytes_per_gb
    return float("nan")


def _footprint_rank(entry: Mapping[str, Any], key: str) -> tuple[int, int, str]:
    """Sort key choosing among footprint entries: measured, then exact directory, then name."""
    return (
        0 if entry.get(STATUS_KEY) is None else 1,
        0 if normalise_model_key(str(entry.get(DIR_KEY, ""))) == key else 1,
        str(entry.get(DIR_KEY, "")),
    )


def select_footprints(
    model_ids: Sequence[str], entries: Iterable[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The one footprint entry per model id, joined on name."""
    wanted = {normalise_model_key(m): m for m in model_ids}
    candidates: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        names = {normalise_model_key(str(entry.get(DIR_KEY, "")))}
        if entry.get(HF_ID_KEY):
            names.add(normalise_model_key(str(entry[HF_ID_KEY])))
        for name in names & wanted.keys():
            candidates.setdefault(name, []).append(entry)
    return {
        wanted[key]: dict(min(found, key=lambda e: _footprint_rank(e, key)))
        for key, found in candidates.items()
    }


def load_footprints(path: Path | str = DEFAULT_FOOTPRINTS) -> dict[str, dict[str, Any]]:
    """The tracked footprint file, as ``{model_id: entry}``."""
    payload = json.loads(Path(path).read_text())
    return dict(payload.get(MODELS_KEY, {}))


def vram_for_models(
    model_ids: Sequence[str], footprints: Mapping[str, Mapping[str, Any]]
) -> np.ndarray:
    """[M] served VRAM in GB, NaN for a model with no usable footprint."""
    vram = np.array([footprint_vram_gb(footprints.get(m, {})) for m in model_ids], dtype=np.float64)
    unknown = [m for m, gb in zip(model_ids, vram, strict=True) if not np.isfinite(gb)]
    if unknown:
        logger.warning(
            f"no VRAM for {len(unknown)}/{len(model_ids)} model(s); they stay NaN and "
            f"are excluded from the knapsack task: {unknown}"
        )
    return vram


def parse_release_date(raw: str | None) -> str:
    """An ISO date from an ISO timestamp, or "" for anything unusable."""
    if not raw:
        return ""
    date = str(raw).strip()[:ISO_DATE_WIDTH]
    try:
        year, month, day = (int(part) for part in date.split("-"))
    except ValueError:
        logger.warning(f"unparseable release date {raw!r} — dropped")
        return ""
    if not (1 <= month <= 12 and 1 <= day <= 31):
        logger.warning(f"implausible release date {raw!r} — dropped")
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def load_release_dates(path: Path | str = DEFAULT_RELEASE_DATES) -> dict[str, str]:
    """The tracked date file, as ``{model_id: ISO date}``."""
    payload = json.loads(Path(path).read_text())
    return {
        model_id: parse_release_date(
            entry.get("created_at") if isinstance(entry, Mapping) else entry
        )
        for model_id, entry in payload.get(MODELS_KEY, {}).items()
    }


def dates_for_models(model_ids: Sequence[str], dates: Mapping[str, str]) -> list[str]:
    """[M] ISO release dates, "" for a model with no recorded date."""
    out = [dates.get(m, "") for m in model_ids]
    undated = [m for m, d in zip(model_ids, out, strict=True) if not d]
    if undated:
        logger.warning(
            f"no release date for {len(undated)}/{len(model_ids)} model(s); they are "
            f"train-only rows in the temporal split: {undated}"
        )
    return out


def params_b_for_models(
    model_ids: Sequence[str], footprints: Mapping[str, Mapping[str, Any]]
) -> np.ndarray:
    """[M] parameter count in billions, NaN for a model that states none."""
    params = np.array(
        [float(footprints.get(m, {}).get(PARAMS_B_KEY) or "nan") for m in model_ids],
        dtype=np.float64,
    )
    unknown = int((~np.isfinite(params)).sum())
    if unknown:
        logger.warning(
            f"no parameter count for {unknown}/{len(model_ids)} model(s); they stay NaN "
            "and are dropped by the size readout rather than imputed"
        )
    return params


def families_for_models(
    model_ids: Sequence[str], footprints: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """[M] publisher family from the footprint file, "" where it names none."""
    families = [str(footprints.get(m, {}).get(FAMILY_KEY) or "") for m in model_ids]
    unknown = [m for m, f in zip(model_ids, families, strict=True) if not f]
    if unknown:
        logger.warning(
            f"no family for {len(unknown)}/{len(model_ids)} model(s); they are excluded "
            f"from the family readouts rather than pooled into an 'other' class: {unknown}"
        )
    return families


def chat_flags_for_models(model_ids: Sequence[str]) -> np.ndarray:
    """[M] bool — True where the model name declares an instruction tune (else base)."""
    flags = np.array(
        [any(token in m.lower() for token in CHAT_NAME_TOKENS) for m in model_ids], dtype=bool
    )
    logger.info(f"{int(flags.sum())}/{len(model_ids)} model names declare an instruction tune")
    return flags


def load_benchmark_dates(path: Path | str = DEFAULT_BENCHMARK_DATES) -> dict[str, str]:
    """``{suite prefix: ISO publication date}`` from the benchmark-date table, ``{}`` if absent."""
    if not Path(path).exists():
        logger.info(
            f"{path} is absent — no eval axis carries a publication date, so the "
            "contamination readout reports a named skip"
        )
        return {}
    payload = json.loads(Path(path).read_text())
    dated = {
        str(suite): parse_release_date(
            entry.get(PUBLISHED_KEY) if isinstance(entry, Mapping) else entry
        )
        for suite, entry in payload.get(SUITES_KEY, {}).items()
    }
    return {suite: date for suite, date in dated.items() if date}


def dates_for_benchmarks(bench_names: Sequence[str], suites: Mapping[str, str]) -> list[str]:
    """[B] ISO publication date per benchmark (longest matching suite prefix), "" if none."""
    keys = sorted(suites, key=len, reverse=True)
    out = [
        next((suites[k] for k in keys if name == k or name.startswith(f"{k}_")), "")
        for name in bench_names
    ]
    logger.info(f"benchmark dates: {sum(bool(d) for d in out)}/{len(out)} eval axes dated")
    return out
