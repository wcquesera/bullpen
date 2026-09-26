"""The encoder contract every arm implements.

An encoder consumes a ``(models x questions)`` correctness submatrix ``A`` with the
global ids of its columns and exposes::

    fit(A, cols)               learn item parameters and the training embeddings X
    fold_in(bits, cols, row)   place a model from a few observed cells; ``row`` is its
                               index in the full model table (for table-lookup arms)
    refit(bits, cols, row)     place a model from its whole observed row
    predict(theta, cols)       the method's reconstruction of a row, or ``None``
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - torch is imported lazily at call time
    import torch

#: What an encoder needs on a new interview B before it can place a model there,
#: cheapest first.
INTERVIEW_REQUIREMENTS: tuple[str, ...] = (
    "nothing",
    "training_responses",
    "training_grades_pool",
    "training_grades_on_b",
)

#: Default ridge strength for the analytic fold-in solves.
RIDGE: float = 1e-2

#: Torch intra-op threads while fitting (``BULLPEN_THREADS``). The tensors are tiny,
#: so one thread is fastest; the count also changes floating-point reduction order.
CPU_THREADS: int = int(os.environ.get("BULLPEN_THREADS", "1"))


def torch_device() -> torch.device:
    """CUDA if it is there, CPU otherwise. Imports torch lazily."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def on_device(module: torch.nn.Module) -> torch.nn.Module:
    """``module``, moved to the device this process computes on."""
    return module.to(torch_device())


def park_fitted_modules(obj: object) -> None:
    """Move every fitted torch module on ``obj`` (singly or in a tuple) to the host."""
    import torch

    for name, value in vars(obj).items():
        if isinstance(value, torch.nn.Module):
            setattr(obj, name, value.cpu())
        elif (
            isinstance(value, tuple)
            and value
            and all(isinstance(m, torch.nn.Module) for m in value)
        ):
            setattr(obj, name, tuple(m.cpu() for m in value))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def cpu_thread_cap(n: int | None = None) -> Iterator[None]:
    """Cap torch's intra-op threads for the duration (no-op on CUDA or for ``n <= 0``)."""
    import torch

    n = CPU_THREADS if n is None else n
    if torch.cuda.is_available() or n <= 0:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def capped_threads[F: Callable[..., Any]](fn: F) -> F:
    """Run ``fn`` under :func:`cpu_thread_cap`. Applied to every torch-backed fit."""

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        with cpu_thread_cap():
            return fn(*args, **kwargs)

    return inner  # type: ignore[return-value]


@dataclass
class Encoder(ABC):
    """Common interface. Subclasses fill :attr:`X` and optionally :meth:`predict`."""

    name: str
    dim: int = 32
    seed: int = 0
    #: [M, dim] training embeddings, filled by :meth:`fit`
    X: np.ndarray | None = field(default=None, repr=False)
    #: global question ids of the columns this arm was fitted on
    cols: np.ndarray | None = field(default=None, repr=False)
    _pos: dict[int, int] = field(default_factory=dict, repr=False)

    #: whether the method can use observed probe cells to place a new model
    uses_probe: bool = True
    #: whether the method has a generative decoder (native row prediction)
    has_decoder: bool = True
    #: what this arm needs on a new interview; one of :data:`INTERVIEW_REQUIREMENTS`
    interview_requirement: str = "training_grades_pool"
    #: whether the placement carries a variance (unused by the kept arms)
    is_distributional: bool = False
    #: number of correctness columns this arm read; set by :meth:`_prepare`
    input_width: int = 0

    #: Fields that are frozen banks cut from the slice, mapped to the
    #: :class:`~bullpen.data.slice.Slice` attribute they come from. They are dropped
    #: when an artifact is saved and re-attached from the slice on load.
    SHARED_BANKS: ClassVar[dict[str, str]] = {}

    def __post_init__(self) -> None:
        if self.interview_requirement not in INTERVIEW_REQUIREMENTS:
            raise ValueError(
                f"{self.name}: interview_requirement "
                f"{self.interview_requirement!r} is not one of {INTERVIEW_REQUIREMENTS}"
            )
        if self.dim < 1:
            raise ValueError(f"{self.name}: dim must be at least 1, got {self.dim}")

    # -- fitting ------------------------------------------------------------ #
    @abstractmethod
    def fit(self, A: np.ndarray, cols: np.ndarray) -> Encoder:
        """Learn from the ``[M, Q]`` correctness matrix ``A`` over global ``cols``."""

    def _prepare(self, A: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Validate one fit's inputs, record the column ids, return ``A`` as float64."""
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError(f"{self.name}: A must be 2-d [models, questions], got {A.shape}")
        if A.size == 0:
            raise ValueError(f"{self.name}: A is empty (shape {A.shape})")
        if not np.isfinite(A).all():
            raise ValueError(
                f"{self.name}: A contains non-finite entries; impute or drop the "
                "affected cells before fitting rather than letting them reach a solve"
            )
        cols = np.asarray(cols)
        if cols.ndim != 1:
            raise ValueError(f"{self.name}: cols must be 1-d, got shape {cols.shape}")
        if cols.shape[0] != A.shape[1]:
            raise ValueError(
                f"{self.name}: A has {A.shape[1]} columns but cols labels {cols.shape[0]} of them"
            )
        positions = {int(c): i for i, c in enumerate(cols)}
        if len(positions) != cols.shape[0]:
            raise ValueError(f"{self.name}: cols contains repeated question ids")
        self.cols = cols
        self._pos = positions
        self.input_width = int(cols.shape[0])
        return A

    def local(self, cols: np.ndarray) -> np.ndarray:
        """Map global question ids to positions in the fitted item table."""
        if self.cols is None:
            raise RuntimeError(f"{self.name}: local() called before fit()")
        cols = np.asarray(cols)
        unknown = sorted({int(c) for c in cols.ravel()} - self._pos.keys())
        if unknown:
            raise ValueError(
                f"{self.name} was fitted on {self.input_width} questions and was "
                f"asked about {len(unknown)} it has never seen, e.g. {unknown[:5]}"
            )
        return np.array([self._pos[int(c)] for c in cols], dtype=int)

    # -- placing a model ---------------------------------------------------- #
    @abstractmethod
    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Place a model from the cells it was observed on. ``cols`` are global ids."""

    def refit(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Place a model from its whole observed row; defaults to :meth:`fold_in`."""
        return self.fold_in(bits, cols, row)

    def metric_weights(self) -> np.ndarray | None:
        """Per-dimension weights on squared distances in this space, or ``None`` (Euclidean)."""
        return None

    # -- reading a model back out ------------------------------------------- #
    def predict(self, theta: np.ndarray, cols: np.ndarray) -> np.ndarray | None:
        """Native reconstruction of a row, or ``None`` if the method has no decoder."""
        return None


def ridge_solve(
    B: np.ndarray, y: np.ndarray, lam: float = RIDGE, prior: np.ndarray | None = None
) -> np.ndarray:
    """``argmin_theta ||B theta - y||^2 + lam ||theta - prior||^2``.

    An empty interview (``B`` with no rows) returns the prior.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError(f"B must be 2-d [observations, dim], got shape {B.shape}")
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape[0] != B.shape[0]:
        raise ValueError(f"B has {B.shape[0]} rows but y has {y.shape[0]} entries")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    d = B.shape[1]
    prior = np.zeros(d) if prior is None else np.asarray(prior, dtype=np.float64).ravel()
    if prior.shape[0] != d:
        raise ValueError(f"prior has {prior.shape[0]} entries, B has {d} columns")
    return prior + np.linalg.solve(B.T @ B + lam * np.eye(d), B.T @ (y - B @ prior))
