"""Task groups: structural readouts, downstream uses and decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger

STRUCTURAL: str = "structural"
DOWNSTREAM: str = "downstream"
DECISION: str = "decision"


@dataclass(frozen=True)
class TaskGroup:
    """One group, and the instruction for reading a number that belongs to it."""

    key: str
    title: str
    #: how to read a score in this group
    reading: str


GROUPS: dict[str, TaskGroup] = {
    STRUCTURAL: TaskGroup(
        key=STRUCTURAL,
        title="Structural readouts",
        reading=(
            "Cross-validated linear readout of the frozen bank, held out over whole "
            "models or whole benchmarks. A low number is a property of the bank. Do "
            "not headline a high cross-benchmark rho: the score matrix is "
            "effectively rank-2, so competence alone already earns it."
        ),
    ),
    DOWNSTREAM: TaskGroup(
        key=DOWNSTREAM,
        title="Downstream uses",
        reading=(
            "Neighbour-informed: the other models' true rows supply question "
            "difficulty for free. Quote the difficulty-only null on the same rows, "
            "and name the neighbour metric — cosine flatters wide embeddings and "
            "collapses 1-d ones."
        ),
    ),
    DECISION: TaskGroup(
        key=DECISION,
        title="Decisions",
        reading=(
            "Scored through a decision rule as well as a representation, so a "
            "negative number does not localise to either. Report the plug-in and "
            "the shrunk rule together, and disbelieve a positive number here "
            "before publishing it."
        ),
    ),
}

#: Reading order of the groups.
GROUP_ORDER: tuple[str, ...] = (STRUCTURAL, DOWNSTREAM, DECISION)


@runtime_checkable
class GroupedTask(Protocol):
    """The two fields the grouping helpers need from a task."""

    @property
    def name(self) -> str: ...

    @property
    def group(self) -> str: ...


def tasks_in_group(tasks: Iterable[GroupedTask], group: str) -> list[str]:
    """Names of the given tasks that belong to ``group``, in the order supplied."""
    require_group(group)
    return [t.name for t in tasks if t.group == group]


def require_group(key: str) -> TaskGroup:
    """Look a group up, or explain which keys exist."""
    if key not in GROUPS:
        raise ValueError(f"unknown group {key!r}; known groups are {list(GROUP_ORDER)}")
    return GROUPS[key]


def select_tasks(
    tasks: Iterable[GroupedTask],
    names: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
) -> list[str]:
    """Task names selected by name and/or group (their union), in registration order."""
    ordered = list(tasks)
    known = {t.name for t in ordered}
    if names is None and groups is None:
        return [t.name for t in ordered]

    wanted: set[str] = set()
    if names is not None:
        requested = list(names)
        unknown = [n for n in requested if n not in known]
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}; registered: {sorted(known)}")
        wanted.update(requested)
    if groups is not None:
        for key in groups:
            members = tasks_in_group(ordered, key)
            if not members:
                logger.warning("group {} has no registered tasks", key)
            wanted.update(members)

    selected = [t.name for t in ordered if t.name in wanted]
    if not selected:
        raise ValueError("the selection matched no tasks")
    return selected
