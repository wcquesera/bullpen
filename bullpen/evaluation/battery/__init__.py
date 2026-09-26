"""The evaluation battery: the tasks a fitted model embedding is scored on.

Each task is a function of a :class:`TaskContext` registered with ``@task``; it returns a
flat dict of metrics with a declared primary and an interval over the unit it holds out
(whole models or whole benchmarks). The submodules below register their tasks on import,
so their import order is the order of :data:`BATTERY`, in which tasks run and are written.
"""

# isort: off
from bullpen.evaluation.battery.constants import (
    CI_KEYS,
    DIAG_LIST_SEP,
    DIAGNOSTIC_PREFIX,
    KNAPSACK_DRAWS,
    MIN_BENCHMARKS_FOR_TRANSFER,
    TRANSFER_PERMUTATIONS,
)
from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.registry import BATTERY, Task, TaskResult, run_battery, task
from bullpen.evaluation.battery import shared, helpers, structural, traits, leaderboards, cards  # noqa: F401
from bullpen.evaluation.battery import group_h, downstream, decision, item_axis, survey  # noqa: F401
from bullpen.evaluation.battery.traits import TRAIT_TASKS
from bullpen.evaluation.battery.leaderboards import EXTERNAL_TABLE_BY_TASK
from bullpen.evaluation.battery.group_h import GROUP_H_TASKS
from bullpen.evaluation.battery import constants
# isort: on

__all__ = [
    "BATTERY",
    "CI_KEYS",
    "DIAGNOSTIC_PREFIX",
    "DIAG_LIST_SEP",
    "EXTERNAL_TABLE_BY_TASK",
    "GROUP_H_TASKS",
    "KNAPSACK_DRAWS",
    "MIN_BENCHMARKS_FOR_TRANSFER",
    "TRAIT_TASKS",
    "TRANSFER_PERMUTATIONS",
    "Task",
    "TaskContext",
    "TaskMetrics",
    "TaskResult",
    "constants",
    "run_battery",
    "task",
]
