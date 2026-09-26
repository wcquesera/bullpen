"""The reader of ``config/``: every YAML file is parsed here into a frozen dataclass.

    config/battery.yaml, battery_quick.yaml   task-level constants of the evaluation battery
    config/splits.yaml            split and fold seeds
    config/process.yaml           the coverage and QC gates of ``process.py build``
    config/metrics.yaml           ridge grid and bootstrap defaults
    config/train.yaml             training defaults (``train.py``)
    config/evaluate.yaml          per-task chance levels and the aggregation seed (``eval.py``)
    config/collect.yaml           generation budgets and assembly gates (``collect.py``)
    config/grading.yaml           the code-execution sandbox and reasoning-model markers
    config/text_banks.yaml        the answer/question sentence encoder (``process.py embed``)
    config/benchmark_groups.yaml  the 13 benchmark groups of the transfer refits
    config/paper.yaml             the arm and task registry (read by :mod:`bullpen.runs`)

Every loader is cached per path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # the grading package imports this module; keep the cycle type-only
    from bullpen.grading.sandbox import Sandbox

REPO_ROOT = Path(__file__).resolve().parents[1]

#: overrides the config directory for every file this module reads
CONFIG_DIR_ENV = "BULLPEN_CONFIG_DIR"


class ConfigError(RuntimeError):
    """A ``config/`` file is missing, unparseable, or missing a required key."""


def config_dir() -> Path:
    """Where ``config/`` is read from, honouring the env override."""
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override) if override else REPO_ROOT / "config"


CONFIG_DIR = REPO_ROOT / "config"

#: The model metadata and label tree; ``train.py`` and ``eval.py`` set
#: ``BULLPEN_RAW_DIR`` from a run config's ``data.raw_dir``.
RAW_DIR_ENV = "BULLPEN_RAW_DIR"
RAW_DIR = Path(os.environ.get(RAW_DIR_ENV) or REPO_ROOT / "data" / "raw")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _read(name: str, path: Path | None, required: tuple[str, ...]) -> dict[str, Any]:
    """Parse one config file, raising :class:`ConfigError` on a missing file or key."""
    p = Path(path) if path is not None else config_dir() / name
    if not p.is_file():
        raise ConfigError(f"{p} is missing")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must parse to a mapping, got {type(raw).__name__}")
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        raise ConfigError(f"{p} is missing required key(s): {', '.join(missing)}")
    return raw


# --------------------------------------------------------------------------- #
# battery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatteryConfig:
    """``config/battery.yaml``: task-level constants of the evaluation battery."""

    neighbour_k: int
    neighbour_metric: str
    probe_sizes: tuple[int, ...]
    headline_probe_size: int
    vram_budgets_gb: tuple[int, ...]
    headline_budget_gb: int
    knapsack_draws: int
    half_split_draws: int
    reliability_draws: int
    ndcg_k: int
    ece_bins: int
    n_folds: int


_BATTERY_KEYS: tuple[str, ...] = tuple(BatteryConfig.__dataclass_fields__)


@lru_cache(maxsize=4)
def _battery(path_str: str | None) -> BatteryConfig:
    raw = _read("battery.yaml", Path(path_str) if path_str else None, _BATTERY_KEYS)
    cfg = BatteryConfig(
        neighbour_k=int(raw["neighbour_k"]),
        neighbour_metric=str(raw["neighbour_metric"]),
        probe_sizes=tuple(int(x) for x in raw["probe_sizes"]),
        headline_probe_size=int(raw["headline_probe_size"]),
        vram_budgets_gb=tuple(int(x) for x in raw["vram_budgets_gb"]),
        headline_budget_gb=int(raw["headline_budget_gb"]),
        knapsack_draws=int(raw["knapsack_draws"]),
        half_split_draws=int(raw["half_split_draws"]),
        reliability_draws=int(raw["reliability_draws"]),
        ndcg_k=int(raw["ndcg_k"]),
        ece_bins=int(raw["ece_bins"]),
        n_folds=int(raw["n_folds"]),
    )
    if cfg.headline_probe_size not in cfg.probe_sizes:
        raise ConfigError(
            f"headline_probe_size {cfg.headline_probe_size} is not among "
            f"probe_sizes {list(cfg.probe_sizes)}"
        )
    if cfg.headline_budget_gb not in cfg.vram_budgets_gb:
        raise ConfigError(
            f"headline_budget_gb {cfg.headline_budget_gb} is not among "
            f"vram_budgets_gb {list(cfg.vram_budgets_gb)}"
        )
    return cfg


def load_battery(path: Path | str | None = None) -> BatteryConfig:
    """Parse ``config/battery.yaml``."""
    return _battery(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitsConfig:
    """``config/splits.yaml``: split and fold-plan seeds."""

    holdout_frac: float
    split_seed: int
    #: seed of the ``random`` K-fold scheme (:mod:`bullpen.data.folds`)
    fold_seed: int


_SPLITS_KEYS: tuple[str, ...] = ("holdout_frac", "split_seed", "fold_seed")


@lru_cache(maxsize=4)
def _splits(path_str: str | None) -> SplitsConfig:
    raw = _read("splits.yaml", Path(path_str) if path_str else None, _SPLITS_KEYS)
    frac = float(raw["holdout_frac"])
    if not 0.0 < frac < 1.0:
        raise ConfigError(f"holdout_frac must be strictly between 0 and 1, got {frac}")
    return SplitsConfig(
        holdout_frac=frac,
        split_seed=int(raw["split_seed"]),
        fold_seed=int(raw["fold_seed"]),
    )


def load_splits(path: Path | str | None = None) -> SplitsConfig:
    """Parse ``config/splits.yaml``."""
    return _splits(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# process
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcessConfig:
    """``config/process.yaml``: the quality gates of ``process.py build``.

    ``coverage_floor`` drops a model that answered less of the question axis than
    this (above it, unanswered cells are written as incorrect);
    ``degenerate_accuracy`` flags a survivor whose accuracy suggests a refusal loop
    or template mismatch.
    """

    coverage_floor: float
    degenerate_accuracy: float


_PROCESS_KEYS: tuple[str, ...] = tuple(ProcessConfig.__dataclass_fields__)


@lru_cache(maxsize=4)
def _process(path_str: str | None) -> ProcessConfig:
    raw = _read("process.yaml", Path(path_str) if path_str else None, _PROCESS_KEYS)
    cfg = ProcessConfig(
        coverage_floor=float(raw["coverage_floor"]),
        degenerate_accuracy=float(raw["degenerate_accuracy"]),
    )
    if not 0.0 < cfg.coverage_floor <= 1.0:
        raise ConfigError(
            f"coverage_floor is a fraction and must be in (0, 1], got {cfg.coverage_floor}"
        )
    if not 0.0 <= cfg.degenerate_accuracy < 1.0:
        raise ConfigError(f"degenerate_accuracy must be in [0, 1), got {cfg.degenerate_accuracy}")
    return cfg


def load_process(path: Path | str | None = None) -> ProcessConfig:
    """Parse ``config/process.yaml``."""
    return _process(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricsConfig:
    """``config/metrics.yaml``: the shared readout constants."""

    ridge_alphas: tuple[float, ...]
    n_boot: int
    bootstrap_alpha: float
    neighbour_metrics: tuple[str, ...]
    target_reliability: float
    lambda_grid: tuple[float, ...]


_METRICS_KEYS: tuple[str, ...] = tuple(MetricsConfig.__dataclass_fields__)


@lru_cache(maxsize=4)
def _metrics(path_str: str | None) -> MetricsConfig:
    raw = _read("metrics.yaml", Path(path_str) if path_str else None, _METRICS_KEYS)
    cfg = MetricsConfig(
        ridge_alphas=tuple(float(x) for x in raw["ridge_alphas"]),
        n_boot=int(raw["n_boot"]),
        bootstrap_alpha=float(raw["bootstrap_alpha"]),
        neighbour_metrics=tuple(str(x) for x in raw["neighbour_metrics"]),
        target_reliability=float(raw["target_reliability"]),
        lambda_grid=tuple(float(x) for x in raw["lambda_grid"]),
    )
    if not 0.0 < cfg.bootstrap_alpha < 1.0:
        raise ConfigError(
            f"bootstrap_alpha is a two-sided level and must be in (0, 1), got {cfg.bootstrap_alpha}"
        )
    return cfg


def load_metrics(path: Path | str | None = None) -> MetricsConfig:
    """Parse ``config/metrics.yaml``."""
    return _metrics(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """``config/train.yaml``: training defaults a run config can override."""

    seeds: tuple[int, ...]
    dim: int
    #: length of the interview the -budget twins are fitted on
    interview_k: int
    #: which :mod:`bullpen.models.question_selection` criterion ranks that interview
    interview_selector: str


@lru_cache(maxsize=4)
def _train(path_str: str | None) -> TrainConfig:
    raw = _read(
        "train.yaml",
        Path(path_str) if path_str else None,
        ("seeds", "dim", "interview_k", "interview_selector"),
    )
    seeds = tuple(int(s) for s in raw["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"seeds must be distinct: {list(seeds)}")
    dim = int(raw["dim"])
    if dim < 1:
        raise ConfigError(f"dim must be at least 1, got {dim}")
    interview_k = int(raw["interview_k"])
    if interview_k < 1:
        raise ConfigError(f"interview_k must be at least 1, got {interview_k}")
    return TrainConfig(
        seeds=seeds,
        dim=dim,
        interview_k=interview_k,
        interview_selector=str(raw["interview_selector"]),
    )


def load_train(path: Path | str | None = None) -> TrainConfig:
    """Parse ``config/train.yaml``."""
    return _train(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluateConfig:
    """``config/evaluate.yaml``: per-task chance levels (three disjoint blocks) and seeds."""

    #: task -> analytic chance level for its metric family
    chance: Mapping[str, float]
    #: task -> the metric key under which the task reports its OWN measured null
    chance_from_metric: Mapping[str, str]
    #: tasks deliberately left unanchored
    unanchored: tuple[str, ...]
    #: seed for the pooling and paired bootstraps, never for a fit
    aggregate_seed: int
    #: chance-to-ceiling spans a raw metric may sit from chance before it is flagged
    divergence_factor: float
    #: models a published leaderboard must score before its task is scored
    min_external_coverage: int

    def declared(self) -> set[str]:
        """Every task this config says something about."""
        return set(self.chance) | set(self.chance_from_metric) | set(self.unanchored)


_EVALUATE_KEYS: tuple[str, ...] = (
    "chance",
    "chance_from_metric",
    "unanchored",
    "aggregate_seed",
    "divergence_factor",
    "min_external_coverage",
)


@lru_cache(maxsize=4)
def _evaluate(path_str: str | None) -> EvaluateConfig:
    raw = _read("evaluate.yaml", Path(path_str) if path_str else None, _EVALUATE_KEYS)
    chance = {str(k): float(v) for k, v in dict(raw["chance"]).items()}
    from_metric = {str(k): str(v) for k, v in dict(raw["chance_from_metric"]).items()}
    unanchored = tuple(str(t) for t in raw["unanchored"])
    blocks = (set(chance), set(from_metric), set(unanchored))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        both = sorted(blocks[a] & blocks[b])
        if both:
            raise ConfigError(
                f"task(s) {both} appear in two chance blocks of evaluate.yaml; each "
                f"task gets exactly one chance level"
            )
    divergence_factor = float(raw["divergence_factor"])
    if not divergence_factor > 1.0:
        raise ConfigError(
            f"evaluate.yaml: divergence_factor is {divergence_factor}, which flags every "
            f"cell outside the scoring range itself; it must exceed 1.0"
        )
    min_external_coverage = int(raw["min_external_coverage"])
    if min_external_coverage < 1:
        raise ConfigError(
            f"evaluate.yaml: min_external_coverage is {min_external_coverage}, which "
            f"admits a leaderboard readout fitted on no models at all"
        )
    return EvaluateConfig(
        chance=chance,
        chance_from_metric=from_metric,
        unanchored=unanchored,
        aggregate_seed=int(raw["aggregate_seed"]),
        divergence_factor=divergence_factor,
        min_external_coverage=min_external_coverage,
    )


def load_evaluate(path: Path | str | None = None) -> EvaluateConfig:
    """Parse ``config/evaluate.yaml``."""
    return _evaluate(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationConfig:
    """``collect.yaml: generation``: token budget and temperature, standard vs reasoning models."""

    standard_max_new_tokens: int
    thinking_max_new_tokens: int
    standard_temperature: float
    thinking_temperature: float
    top_p: float
    top_k: int
    draws: int
    batch_size: int
    chunk_size: int
    #: case-insensitive substrings that mark a model id as a reasoning model
    thinking_markers: tuple[str, ...]

    def is_thinking(self, model_id: str) -> bool:
        """Whether ``model_id`` gets the thinking budget and temperature."""
        low = model_id.lower()
        return any(m.lower() in low for m in self.thinking_markers)

    def budget(self, model_id: str) -> tuple[int, float]:
        """``(max_new_tokens, temperature)`` for one model."""
        if self.is_thinking(model_id):
            return self.thinking_max_new_tokens, self.thinking_temperature
        return self.standard_max_new_tokens, self.standard_temperature


@dataclass(frozen=True)
class AssemblyConfig:
    """``collect.yaml: assembly``: the gates that turn per-model answers into a cut."""

    #: share of the question axis a model must answer to enter the cut
    min_coverage: float
    #: correctness agreement at or above which a pair of models is reported
    dup_agreement: float
    #: case-insensitive substrings marking a slug as a bad download or a scaffold
    bad_slug_patterns: tuple[str, ...]

    def is_bad_slug(self, slug: str) -> bool:
        """Whether a slug matches one of the excluded patterns."""
        low = slug.lower()
        return any(p.lower() in low for p in self.bad_slug_patterns)


@dataclass(frozen=True)
class CollectConfig:
    """``config/collect.yaml``: generation budgets (``collect.py``) and assembly gates
    (``process.py build``)."""

    generation: GenerationConfig
    assembly: AssemblyConfig


_COLLECT_KEYS: tuple[str, ...] = ("generation", "assembly")

_GENERATION_KEYS: tuple[str, ...] = (
    "standard_max_new_tokens",
    "thinking_max_new_tokens",
    "standard_temperature",
    "thinking_temperature",
    "top_p",
    "top_k",
    "draws",
    "batch_size",
    "chunk_size",
    "thinking_markers",
)


def _resolve(path_str: str) -> Path:
    """A config path, relative to the checkout unless absolute."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _generation(raw: Mapping[str, Any]) -> GenerationConfig:
    missing = [k for k in _GENERATION_KEYS if raw.get(k) is None]
    if missing:
        raise ConfigError(f"collect.yaml: generation is missing key(s): {', '.join(missing)}")
    caps = {k: int(raw[k]) for k in ("standard_max_new_tokens", "thinking_max_new_tokens")}
    for name, value in caps.items():
        if value < 1:
            raise ConfigError(f"collect.yaml: generation.{name} must be at least 1, got {value}")
    counts = {k: int(raw[k]) for k in ("draws", "batch_size", "chunk_size", "top_k")}
    for name, value in counts.items():
        if value < 1:
            raise ConfigError(f"collect.yaml: generation.{name} must be at least 1, got {value}")
    markers = tuple(str(m) for m in raw["thinking_markers"])
    if not markers:
        raise ConfigError(
            "collect.yaml: generation.thinking_markers is empty; every model would then be "
            "capped at the non-thinking budget, which truncates reasoning chains into wrong "
            "answers"
        )
    return GenerationConfig(
        standard_temperature=float(raw["standard_temperature"]),
        thinking_temperature=float(raw["thinking_temperature"]),
        top_p=float(raw["top_p"]),
        thinking_markers=markers,
        **caps,
        **counts,
    )


def _assembly(raw: Mapping[str, Any]) -> AssemblyConfig:
    for key in ("min_coverage", "dup_agreement"):
        if raw.get(key) is None:
            raise ConfigError(f"collect.yaml: assembly is missing required key: {key}")
        value = float(raw[key])
        if not 0.0 < value <= 1.0:
            raise ConfigError(
                f"collect.yaml: assembly.{key} must be in (0, 1], got {value}; it is a share "
                f"of the question axis, not a count"
            )
    return AssemblyConfig(
        min_coverage=float(raw["min_coverage"]),
        dup_agreement=float(raw["dup_agreement"]),
        bad_slug_patterns=tuple(str(p) for p in raw.get("bad_slug_patterns") or ()),
    )


@lru_cache(maxsize=4)
def _collect(path_str: str | None) -> CollectConfig:
    raw = _read("collect.yaml", Path(path_str) if path_str else None, _COLLECT_KEYS)
    return CollectConfig(
        generation=_generation(dict(raw["generation"])),
        assembly=_assembly(dict(raw["assembly"])),
    )


def load_collect(path: Path | str | None = None) -> CollectConfig:
    """Parse ``config/collect.yaml``."""
    return _collect(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# benchmark groups (config/benchmark_groups.yaml)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _benchmark_groups(path_str: str | None) -> dict[str, tuple[str, ...]]:
    raw = _read("benchmark_groups.yaml", Path(path_str) if path_str else None, ("groups",))
    groups = {str(g): tuple(str(b) for b in members) for g, members in raw["groups"].items()}
    seen: dict[str, str] = {}
    for g, members in groups.items():
        for b in members:
            if b in seen:
                raise ConfigError(f"benchmark_groups.yaml: {b} is in both {seen[b]} and {g}")
            seen[b] = g
    return groups


def load_benchmark_groups(
    bench_names: Sequence[str] | None = None, path: Path | str | None = None
) -> dict[str, tuple[str, ...]]:
    """``{group: benchmarks}``; with ``bench_names``, checked to partition them exactly."""
    groups = _benchmark_groups(str(path) if path is not None else None)
    if bench_names is not None:
        grouped = {b for members in groups.values() for b in members}
        missing, extra = set(bench_names) - grouped, grouped - set(bench_names)
        if missing or extra:
            raise ConfigError(
                f"benchmark_groups.yaml does not partition the slice's benchmarks: "
                f"ungrouped {sorted(missing)}, unknown {sorted(extra)}"
            )
    return groups


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GradingConfig:
    """``config/grading.yaml``: the code sandbox and reasoning-model markers for grading."""

    #: ``auto`` / ``bwrap`` / ``none`` — the code-execution containment
    sandbox_backend: str
    #: whether code benchmarks are graded by running the model's code at all
    sandbox_execute: bool
    sandbox_timeout_seconds: int
    sandbox_memory_gb: int
    sandbox_max_file_mb: int
    #: case-insensitive substrings of a model id marking it a reasoning model,
    #: for the purpose of think-first answer extraction
    reasoning_markers: tuple[str, ...]

    def sandbox(self) -> Sandbox:
        """The :class:`bullpen.grading.sandbox.Sandbox` these settings describe."""
        from bullpen.grading.sandbox import Sandbox

        return Sandbox(
            backend_choice=self.sandbox_backend,  # type: ignore[arg-type]
            execute=self.sandbox_execute,
            timeout_seconds=self.sandbox_timeout_seconds,
            memory_gb=self.sandbox_memory_gb,
            max_file_mb=self.sandbox_max_file_mb,
        )


_GRADING_KEYS: tuple[str, ...] = ("sandbox", "reasoning_markers")

_SANDBOX_BACKENDS: tuple[str, ...] = ("auto", "bwrap", "none")

_SANDBOX_LIMIT_KEYS: tuple[str, ...] = ("timeout_seconds", "memory_gb", "max_file_mb")


def _sandbox(raw: Mapping[str, Any]) -> dict[str, Any]:
    backend = str(raw.get("backend", "auto"))
    if backend not in _SANDBOX_BACKENDS:
        raise ConfigError(
            f"grading.yaml: sandbox.backend must be one of {list(_SANDBOX_BACKENDS)}, "
            f"got {backend!r}"
        )
    execute = raw.get("execute", True)
    if not isinstance(execute, bool):
        raise ConfigError(f"grading.yaml: sandbox.execute must be a boolean, got {execute!r}")
    limits: dict[str, Any] = {}
    for key in _SANDBOX_LIMIT_KEYS:
        if raw.get(key) is None:
            raise ConfigError(f"grading.yaml: sandbox is missing required key: {key}")
        value = int(raw[key])
        if value < 1:
            raise ConfigError(f"grading.yaml: sandbox.{key} must be at least 1, got {value}")
        limits[key] = value
    return {"backend": backend, "execute": execute, **limits}


@lru_cache(maxsize=4)
def _grading(path_str: str | None) -> GradingConfig:
    raw = _read("grading.yaml", Path(path_str) if path_str else None, _GRADING_KEYS)
    sandbox = _sandbox(dict(raw["sandbox"]))
    markers = tuple(str(m) for m in raw["reasoning_markers"])
    if not markers:
        raise ConfigError(
            "grading.yaml: reasoning_markers is empty; every reasoning model would then be "
            "graded on its full response, letting an option letter be read out of a discarded "
            "chain of thought"
        )
    return GradingConfig(
        sandbox_backend=sandbox["backend"],
        sandbox_execute=sandbox["execute"],
        sandbox_timeout_seconds=sandbox["timeout_seconds"],
        sandbox_memory_gb=sandbox["memory_gb"],
        sandbox_max_file_mb=sandbox["max_file_mb"],
        reasoning_markers=markers,
    )


def load_grading(path: Path | str | None = None) -> GradingConfig:
    """Parse ``config/grading.yaml``."""
    return _grading(str(path) if path is not None else None)


# --------------------------------------------------------------------------- #
# text banks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TextBanksConfig:
    """``config/text_banks.yaml``: the sentence encoder of the answer and question banks.

    Questions must be embedded under the same encoder and pooling as the answers.
    """

    answer_bank_dir: Path
    answer_bank_slug: str
    question_source: Path
    encoder_hf_id: str
    encoder_dim: int
    encoder_prefix: str
    encoder_pooling: str
    max_seq_length: int
    question_char_cap: int
    batch_size: int
    coverage_warn_threshold: float


_TEXT_BANKS_KEYS: tuple[str, ...] = tuple(
    k for k in TextBanksConfig.__dataclass_fields__ if k != "encoder_prefix"
)

#: poolings :mod:`bullpen.data.text_banks` implements
POOLINGS: tuple[str, ...] = ("mean", "cls")


@lru_cache(maxsize=4)
def _text_banks(path_str: str | None) -> TextBanksConfig:
    raw = _read("text_banks.yaml", Path(path_str) if path_str else None, _TEXT_BANKS_KEYS)
    pooling = str(raw["encoder_pooling"])
    if pooling not in POOLINGS:
        raise ConfigError(f"encoder_pooling must be one of {POOLINGS}, got {pooling!r}")
    sizes = {
        "encoder_dim": int(raw["encoder_dim"]),
        "max_seq_length": int(raw["max_seq_length"]),
        "question_char_cap": int(raw["question_char_cap"]),
        "batch_size": int(raw["batch_size"]),
    }
    bad = sorted(k for k, v in sizes.items() if v < 1)
    if bad:
        raise ConfigError(f"{bad} must be at least 1 in text_banks.yaml")
    warn_at = float(raw["coverage_warn_threshold"])
    if not 0.0 <= warn_at <= 1.0:
        raise ConfigError(f"coverage_warn_threshold is a fraction in [0, 1], got {warn_at}")
    return TextBanksConfig(
        answer_bank_dir=_resolve(str(raw["answer_bank_dir"])),
        answer_bank_slug=str(raw["answer_bank_slug"]),
        question_source=_resolve(str(raw["question_source"])),
        encoder_hf_id=str(raw["encoder_hf_id"]),
        encoder_prefix=str(raw.get("encoder_prefix") or ""),
        encoder_pooling=pooling,
        coverage_warn_threshold=warn_at,
        **sizes,
    )


def load_text_banks(path: Path | str | None = None) -> TextBanksConfig:
    """Parse ``config/text_banks.yaml``."""
    return _text_banks(str(path) if path is not None else None)
