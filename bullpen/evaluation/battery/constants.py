"""Module-level constants of the battery: sizes, draws, floors, seeds, units."""

from __future__ import annotations

import os

#: Neighbours used by every kNN readout.
NEIGHBOUR_K: int = 5

#: Neighbour metric for the kNN readouts.
NEIGHBOUR_METRIC: str = "cosine"

#: Interview sizes the cold-start curve is measured at, log-spaced.
PROBE_SIZES: tuple[int, ...] = (5, 10, 20, 50)


#: Prefix marking a metric as a DIAGNOSTIC.
DIAGNOSTIC_PREFIX: str = "diag_"

#: Joins a list-valued diagnostic into one cell.
DIAG_LIST_SEP: str = "+"


#: VRAM budgets (GB) the portfolio task is solved under — one consumer card, one
#: workstation card, one A100/H100, one small multi-GPU node.
VRAM_BUDGETS_GB: tuple[int, ...] = (24, 48, 80, 160)

#: The budget whose retention is the portfolio headline.
HEADLINE_BUDGET_GB: int = 24

#: Question half-splits averaged over by the portfolio task.
KNAPSACK_DRAWS: int = 10

#: Random affordable squads drawn per (draw, budget) to measure the knapsack null.
KNAPSACK_NULL_SQUADS: int = 20


#: Question half-splits averaged over by the specialisation task.
HALF_SPLIT_DRAWS: int = 5

#: Draws used to estimate the specialisation target's own test-retest reliability.
RELIABILITY_DRAWS: int = 20

#: Cut-off for the top-k ranking readout: a deployment shortlist, not a leaderboard.
NDCG_K: int = 10


#: A pairwise head predicts P(i beats j); 0.5 is the decision boundary and the
#: label threshold, so it is one constant rather than two literals.
DECISION_THRESHOLD: float = 0.5

#: Below this, "transfer to a held-out benchmark" has no held-out benchmark.
MIN_BENCHMARKS_FOR_TRANSFER: int = 2

#: The per-benchmark transfer tasks.
BENCHMARK_HOLDOUT_TASKS: tuple[str, ...] = ("pairwise", "ranking", "cross_benchmark")

#: Floor on a knapsack item weight, so a zero-GB entry cannot divide by zero in the
#: ratio-greedy gain.
MIN_ITEM_WEIGHT_GB: float = 1e-6

#: Floor on a feature's training standard deviation before it is divided out, so a
#: constant feature passes through as zero instead of blowing up.
STD_FLOOR: float = 1e-9

#: Default folds over MODELS.
DEFAULT_N_FOLDS: int = 5


#: The temporal firewall.
TEMPORAL_CUTOFF: str = "2024-01-01"

#: Below this many labelled models a supervised readout is a note, not a number.
MIN_LABELLED_FIT: int = 8
MIN_LABELLED_SCORE: int = 4

#: Members a family needs before it is a class. A family the head saw twice is not
#: a class it can learn, and leaving it in puts a guaranteed zero in a macro mean.
MIN_FAMILY_MEMBERS: int = 3

#: Distinct families needed before "which family is this" is a closed-set problem.
MIN_FAMILIES: int = 3

#: Label permutations behind the measured null for the two family readouts.
LABEL_PERMUTATIONS: int = 200

#: Permutations behind ``cross_benchmark``'s null.
TRANSFER_PERMUTATIONS: int = int(os.environ.get("BULLPEN_TRANSFER_PERMUTATIONS", "40"))

#: Permutations behind the published-leaderboard readouts' null.
EXTERNAL_PERMUTATIONS: int = 200

#: Permuted banks per half-split draw behind ``knapsack``'s null, at the headline budget only.
KNAPSACK_PERMUTATIONS: int = 40


#: Questions sampled for the answer-divergence target.
DIVERGENCE_QUESTIONS: int = 512

#: Co-covered sampled questions a model pair needs before its mean answer cosine is a
#: measurement.
MIN_PAIR_COVERAGE: int = 8


#: Benchmark-name substrings marking a multi-step reasoning axis, and a safety or truthfulness
#: one.
REASONING_AXIS_TOKENS: tuple[str, ...] = (
    "gsm8k",
    "mathqa",
    "asdiv",
    "logiqa",
    "mathematics",
    "algebra",
    "formal_logic",
)


#: Axes a tilt contrast needs on each side before it is a composite rather than one
#: benchmark wearing a group's name.
MIN_CONTRAST_AXES: int = 2


#: Inverse regularisation for every logistic head here.
LOGISTIC_C: float = 1.0
LOGISTIC_MAX_ITER: int = 2000

#: Floor on a denominator that is a spread between two policies. Below it the
#: policies are indistinguishable and the ratio is not a fraction of anything.
MIN_HEADROOM: float = 1e-9

#: Resamples behind every per-task interval.
CI_BOOT: int = 1000

#: The independent unit a task's interval is clustered on, written into every cell as
#: ``resample_unit``.
UNIT_MODELS: str = "models"
UNIT_BENCHMARKS: str = "benchmarks"
UNIT_QUESTIONS: str = "questions"
UNIT_BENCHMARK_CLUSTERS: str = "benchmark_clusters"

#: What a task with no measurement behind it writes in ``resample_unit``.
UNIT_NONE: str = "none"

#: Every key the interval contract adds to a cell. Named once so the battery test
#: that checks all of them are present cannot drift from what is written.
CI_KEYS: tuple[str, ...] = ("ci_lo", "ci_hi", "n_units", "resample_unit")

#: Bootstrap draws that must survive (return a finite statistic) before a percentile over them
#: is an interval rather than a summary of the draws that happened not to be degenerate.
MIN_CI_DRAWS: int = CI_BOOT // 2
