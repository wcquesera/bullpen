"""The task registry: :class:`Task`, :class:`TaskResult`, ``@task``, runner."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from loguru import logger

from bullpen.evaluation.battery.context import TaskContext, TaskFn, TaskMetrics
from bullpen.evaluation.groups import STRUCTURAL, require_group, select_tasks


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    """One registered battery task and the contract for reading its score."""

    name: str
    group: str
    primary: str
    lower_is_better: bool
    fn: TaskFn
    doc: str
    diagnostic: bool = False

    def run(self, ctx: TaskContext) -> TaskResult:
        """Score this task. Raises whatever the task body raises."""
        metrics = self.fn(ctx)
        if self.primary not in metrics:
            raise KeyError(f"task {self.name} omitted its primary metric {self.primary!r}")
        primary = metrics[self.primary]
        if not isinstance(primary, float | int):
            raise TypeError(
                f"task {self.name} returned a non-numeric primary metric {self.primary}={primary!r}"
            )
        return TaskResult(
            task=self.name,
            group=self.group,
            primary=self.primary,
            lower_is_better=self.lower_is_better,
            score=float(primary),
            metrics=metrics,
            diagnostic=self.diagnostic,
        )


@dataclass(frozen=True)
class TaskResult:
    """A scored task: the headline number, everything else it reported, and ``error``."""

    task: str
    group: str
    primary: str
    lower_is_better: bool
    score: float
    metrics: TaskMetrics
    diagnostic: bool = False
    error: str | None = None

    @classmethod
    def from_error(cls, task: Task, exc: Exception) -> TaskResult:
        return cls(
            task=task.name,
            group=task.group,
            primary=task.primary,
            lower_is_better=task.lower_is_better,
            score=float("nan"),
            metrics={},
            diagnostic=task.diagnostic,
            error=f"{type(exc).__name__}: {exc}",
        )


#: Registration order is reading order: structural, then downstream, then decision.
BATTERY: dict[str, Task] = {}


def task(
    name: str,
    primary: str,
    lower_is_better: bool = False,
    group: str = STRUCTURAL,
    diagnostic: bool = False,
) -> Callable[[TaskFn], TaskFn]:
    """Register a battery task. See the module docstring for the contract."""
    require_group(group)

    def wrap(fn: TaskFn) -> TaskFn:
        if name in BATTERY:
            raise ValueError(f"task {name!r} is already registered")
        BATTERY[name] = Task(
            name=name,
            group=group,
            primary=primary,
            lower_is_better=lower_is_better,
            diagnostic=diagnostic,
            fn=fn,
            doc=(fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return wrap


def run_battery(
    ctx: TaskContext,
    names: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
) -> dict[str, TaskResult]:
    """Run the battery (or a selection of it) against one fitted encoder."""
    selected = select_tasks(BATTERY.values(), names=names, groups=groups)
    results: dict[str, TaskResult] = {}
    import time as _time

    for i, name in enumerate(selected):
        spec = BATTERY[name]
        t0 = _time.monotonic()
        try:
            results[name] = spec.run(ctx)
        except Exception as exc:  # noqa: BLE001  (one bad task must not sink the sweep)
            logger.exception("task {} failed: {}", name, exc)
            results[name] = TaskResult.from_error(spec, exc)
        dt = _time.monotonic() - t0
        logger.info(
            "task {}/{} {} = {:.3f} [{:.1f}s]", i + 1, len(selected), name, results[name].score, dt
        )
    return results
