"""Decision tasks: routing, replacement, portfolio selection."""

from __future__ import annotations

from bullpen.evaluation.battery.context import TaskContext, TaskMetrics
from bullpen.evaluation.battery.helpers import (
    _top1_routing,
)
from bullpen.evaluation.battery.registry import task
from bullpen.evaluation.battery.shared import (
    _mean_or_nan,
)
from bullpen.evaluation.groups import DECISION

#: The tasks whose numbers :data:`~bullpen.evaluation.metrics.LAMBDA_GRID` can reach.
LAMBDA_TASKS: tuple[str, ...] = ("routing_decomposition", "routing_with_text")


@task("stage5a_routing", primary="gap_closed", group=DECISION)
def stage5a_routing(ctx: TaskContext) -> TaskMetrics:
    """Top-1 routing from the decoded cells: send each question to the predicted best model."""
    out = _top1_routing(ctx, ctx.decoded_cells)
    if "routing_acc" in out:
        # the anchor: the same argmax over decodes from model-permuted banks
        out["gap_closed_permuted_null"] = _mean_or_nan(
            [float(_top1_routing(ctx, P)["gap_closed"]) for P in ctx.permuted_decodes]
        )
    return out
