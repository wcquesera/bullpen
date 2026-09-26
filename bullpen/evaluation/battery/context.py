"""The :class:`TaskContext` every battery task receives, and the metric type aliases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from sklearn.model_selection import KFold

from bullpen.data.group_h_labels import GroupHLabels
from bullpen.evaluation.battery.constants import DEFAULT_N_FOLDS, TRANSFER_PERMUTATIONS
from bullpen.evaluation.metrics import LAMBDA_GRID, oof_ridge_predict

MetricValue = float | int | str | list[float] | list[int]
TaskMetrics = dict[str, MetricValue]
TaskFn = Callable[["TaskContext"], TaskMetrics]


def _permutation_orders(n: int, seed: int, permutations: int) -> list[np.ndarray]:
    """The row orders :func:`_permuted_null` draws for ``np.arange(n)``, materialised."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(n) for _ in range(permutations)]


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
@dataclass
class TaskContext:
    """Everything a battery task may read, and the splits every task must use."""

    encoder: object  # a fitted bullpen.models.base.Encoder
    A: np.ndarray  # [M, Q] accuracies in [0, 1] on the SCORED block
    R: np.ndarray  # [M, Q] binary correctness on the SCORED block
    bench: np.ndarray  # [Q] benchmark index per question
    bench_names: list[str] = field(default_factory=list)
    vram_gb: np.ndarray | None = None  # [M] weights for the portfolio task
    seed: int = 0
    n_folds: int = DEFAULT_N_FOLDS
    #: Shrinkage strengths the routing readouts select over.
    lambdas: tuple[float, ...] = LAMBDA_GRID
    #: [Q] global question ids of the scored block
    cols: np.ndarray | None = None
    #: [M, W] accuracies on the arm's own input block — what ``fold_in`` may read
    A_in: np.ndarray | None = None
    #: [W] global question ids of that input block
    in_cols: np.ndarray | None = None
    #: benchmark indices the per-benchmark transfer tasks score (``pairwise``, ``ranking``,
    #: ``cross_benchmark``).
    scored_benchmarks: np.ndarray | None = None
    #: [M] slice model ids. Only the metadata readouts need them, so a synthetic
    #: context may leave them empty and those tasks return a note.
    model_ids: list[str] = field(default_factory=list)
    #: [M] publisher family, "" where unknown — the two family readouts' labels.
    family: list[str] = field(default_factory=list)
    #: [M] parameter count in billions, NaN where the model states none.
    params_b: np.ndarray | None = None
    #: [M] bool — instruction-tuned per the model name. Lexical, so the False class
    #: carries unlabelled tunes (see :mod:`bullpen.data.enrichment`).
    is_chat: np.ndarray | None = None
    #: [M] ISO release date, "" where undated — the temporal readout's firewall.
    release_date: list[str] = field(default_factory=list)
    #: [B] ISO publication date per BENCHMARK, "" where undated.
    bench_dates: list[str] = field(default_factory=list)
    #: [M, Q, d] answer embeddings on the SCORED block, with the mask saying which cells carry
    #: an embedded answer rather than the zero vector standing in for one the backfill never
    #: reached.
    Ae: np.ndarray | None = None
    Ae_mask: np.ndarray | None = None
    #: [Q, d] QUESTION text embeddings on the SCORED block, in ``cols`` order — the item-side
    #: covariate the question-conditioned readout needs, and the only optional block here that
    #: is not on the model axis.
    Qe: np.ndarray | None = None
    #: ``{published column: [M] score}``, NaN where the board does not list the model
    external: Mapping[str, np.ndarray] = field(default_factory=dict)
    #: [M, T] per-model text traits from text_traits.npz, NaN across a whole
    #: row the scan has no profile for.
    traits: np.ndarray | None = None
    #: [T] channel names of ``traits``, in column order.
    trait_names: tuple[str, ...] = ()
    #: The reward-model and style measurements on their OWN question axis — the 408 eval
    #: columns the reward pass scored, which is not ``cols``.
    group_h: GroupHLabels | None = None

    def __post_init__(self) -> None:
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2-d [models, questions], got shape {self.A.shape}")
        if self.R.shape != self.A.shape:
            raise ValueError(f"R has shape {self.R.shape}, A has {self.A.shape}")
        if not np.isfinite(self.A).all():
            raise ValueError("A contains non-finite entries; impute or drop them before scoring")
        if self.bench.shape != (self.A.shape[1],):
            raise ValueError(
                f"bench must label all {self.A.shape[1]} questions, got shape {self.bench.shape}"
            )
        present = np.unique(self.bench)
        if present.min() < 0 or not np.array_equal(present, np.arange(present.size)):
            raise ValueError(
                "bench must be contiguous 0..B-1 benchmark indices; benchmark_scores "
                f"indexes columns by position, and it got {present[:5]}..."
            )
        if self.bench_names and len(self.bench_names) != present.size:
            raise ValueError(
                f"bench_names has {len(self.bench_names)} entries for {present.size} benchmarks"
            )
        if self.vram_gb is not None and self.vram_gb.shape != (self.A.shape[0],):
            raise ValueError(
                f"vram_gb must have one weight per model ({self.A.shape[0]}), "
                f"got shape {self.vram_gb.shape}"
            )
        self._check_metadata()
        if not 2 <= self.n_folds <= self.A.shape[0]:
            raise ValueError(
                f"n_folds must be in [2, {self.A.shape[0]}] for {self.A.shape[0]} models, "
                f"got {self.n_folds}"
            )
        self._resolve_columns()

    def _check_metadata(self) -> None:
        """Every optional per-model block is on the model axis, or it is not passed."""
        n_models = self.A.shape[0]
        for name in ("model_ids", "family", "release_date"):
            values = getattr(self, name)
            if values and len(values) != n_models:
                raise ValueError(
                    f"{name} has {len(values)} entries for {n_models} models; these "
                    "blocks are positional and are never realigned"
                )
        for name, column in self.external.items():
            if np.asarray(column).shape != (n_models,):
                raise ValueError(
                    f"external column {name!r} has shape {np.asarray(column).shape} for "
                    f"{n_models} models; these blocks are positional and are never realigned"
                )
        for name in ("params_b", "is_chat"):
            values = getattr(self, name)
            if values is not None and np.asarray(values).shape != (n_models,):
                raise ValueError(
                    f"{name} must have one entry per model ({n_models}), got shape "
                    f"{np.asarray(values).shape}"
                )
        if self.bench_dates and len(self.bench_dates) != self.n_benchmarks:
            raise ValueError(
                f"bench_dates has {len(self.bench_dates)} entries for {self.n_benchmarks} "
                "benchmarks; this block is positional and is never realigned"
            )
        if self.traits is not None and np.asarray(self.traits).shape != (
            n_models,
            len(self.trait_names),
        ):
            raise ValueError(
                f"traits must be [{n_models}, {len(self.trait_names)}] over the model axis "
                f"and its named channels, got shape {np.asarray(self.traits).shape}"
            )
        if self.group_h is not None and self.group_h.n_models != n_models:
            raise ValueError(
                f"the reward labels are on {self.group_h.n_models} models and this block "
                f"holds {n_models}; those matrices are positional and are never realigned"
            )
        if self.Qe is not None and np.asarray(self.Qe).shape[:1] != (self.A.shape[1],):
            raise ValueError(
                f"Qe must be [questions, d] over the {self.A.shape[1]}-column scored block, "
                f"got shape {np.asarray(self.Qe).shape}; this block is positional and is "
                "never realigned"
            )
        if self.Ae is None:
            return
        if np.asarray(self.Ae).ndim != 3 or np.asarray(self.Ae).shape[:2] != self.A.shape:
            raise ValueError(
                f"Ae must be [models, questions, d] over {self.A.shape}, got shape "
                f"{np.asarray(self.Ae).shape}"
            )
        if self.Ae_mask is None:
            raise ValueError(
                "Ae was passed without Ae_mask; the bank does not cover every cell, and "
                "without the mask a zero vector reads as an answer"
            )
        if np.asarray(self.Ae_mask).shape != self.A.shape:
            raise ValueError(
                f"Ae_mask must be [models, questions] over {self.A.shape}, got shape "
                f"{np.asarray(self.Ae_mask).shape}"
            )

    def _resolve_columns(self) -> None:
        """Default both column spaces to the encoder's own, and check they line up."""
        own = getattr(self.encoder, "cols", None)
        if self.cols is None:
            if own is None:
                raise RuntimeError("the encoder has no column ids; call fit() before scoring it")
            self.cols = np.asarray(own, dtype=int)
        else:
            self.cols = np.asarray(self.cols, dtype=int)
        if self.in_cols is None:
            if own is None:
                raise RuntimeError("the encoder has no column ids; call fit() before scoring it")
            self.in_cols = np.asarray(own, dtype=int)
        else:
            self.in_cols = np.asarray(self.in_cols, dtype=int)
        if self.A_in is None:
            self.A_in = self.A
        if self.cols.size != self.A.shape[1]:
            raise ValueError(
                f"cols names {self.cols.size} questions but the scored block has {self.A.shape[1]}"
            )
        if self.A_in.shape != (self.A.shape[0], self.in_cols.size):
            raise ValueError(
                f"the input block is {self.A_in.shape} but in_cols names "
                f"{self.in_cols.size} questions over {self.A.shape[0]} models"
            )

    @property
    def n_input_questions(self) -> int:
        """Width of the arm's own input block — the longest interview it can be given."""
        return int(np.asarray(self.in_cols).size)

    @property
    def Z(self) -> np.ndarray:
        """The frozen profile bank [M, d]."""
        bank = getattr(self.encoder, "X", None)
        if bank is None:
            raise RuntimeError("the encoder has no bank; call fit() before scoring it")
        if bank.shape[0] != self.A.shape[0]:
            raise ValueError(
                f"the bank holds {bank.shape[0]} models but A holds {self.A.shape[0]}; "
                "the encoder was fitted on a different slice"
            )
        return bank

    @property
    def n_models(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_questions(self) -> int:
        return int(self.A.shape[1])

    @property
    def n_benchmarks(self) -> int:
        return int(self.bench.max()) + 1

    @property
    def scored(self) -> list[int]:
        """The benchmark indices the transfer tasks score (all, unless a holdout run)."""
        if self.scored_benchmarks is None:
            return list(range(self.n_benchmarks))
        return [int(b) for b in self.scored_benchmarks]

    @cached_property
    def benchmark_scores(self) -> np.ndarray:
        """[M, B] per-benchmark mean accuracy. Cached: five tasks read it."""
        return np.stack(
            [self.A[:, self.bench == b].mean(axis=1) for b in range(self.n_benchmarks)], axis=1
        )

    @cached_property
    def decoded_cells(self) -> np.ndarray:
        """[M, Q] accuracies decoded from the bank alone, out of fold over MODELS."""
        return oof_ridge_predict(self.Z, self.A, n_folds=self.n_folds, seed=self.seed)

    @cached_property
    def permuted_decodes(self) -> list[np.ndarray]:
        """:attr:`decoded_cells` refitted on :data:`TRANSFER_PERMUTATIONS` model-permuted banks."""
        return [
            oof_ridge_predict(self.Z[order], self.A, n_folds=self.n_folds, seed=self.seed)
            for order in _permutation_orders(self.n_models, self.seed, TRANSFER_PERMUTATIONS)
        ]

    def model_folds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """K-fold over MODELS — the split every structural task must use."""
        splitter = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        return list(splitter.split(np.arange(self.n_models)))

    def question_halves(self, draw: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Two disjoint sorted halves of the question index, reproducible per ``draw``."""
        if draw < 0:
            raise ValueError(f"draw must be >= 0, got {draw}")
        if self.n_questions < 2:
            raise ValueError("a half-split needs at least 2 questions")
        rng = np.random.default_rng([self.seed, draw])
        perm = rng.permutation(self.n_questions)
        half = self.n_questions // 2
        return np.sort(perm[:half]), np.sort(perm[half:])
