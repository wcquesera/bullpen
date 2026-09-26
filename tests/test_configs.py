"""Every shipped config parses, and the run configs name registered arms and tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from bullpen.config import (
    load_battery,
    load_benchmark_groups,
    load_collect,
    load_evaluate,
    load_grading,
    load_metrics,
    load_process,
    load_splits,
    load_text_banks,
    load_train,
)
from bullpen.runs import RUNS_DIR, load_paper, load_run_config, task_lists

REPO = Path(__file__).resolve().parents[1]
RUN_CONFIGS = [*sorted(RUNS_DIR.glob("*.yaml")), REPO / "tests" / "fixtures" / "synthetic.yaml"]


def test_package_configs_load():
    assert load_train().dim == 32
    assert load_evaluate().min_external_coverage > 0
    assert load_battery().n_folds == 5
    assert load_battery(REPO / "config" / "battery_quick.yaml").n_folds == 3
    assert load_metrics().n_boot > 0
    assert load_splits() is not None
    assert load_collect().generation.draws == 1
    assert load_grading() is not None
    assert load_process().coverage_floor == pytest.approx(0.9)
    assert load_text_banks().encoder_dim == 768
    groups = load_benchmark_groups()
    assert len(groups) == 13


def test_there_are_run_configs():
    names = {p.stem for p in RUN_CONFIGS}
    assert {
        "sample",
        "strict",
        "strict_competitors",
        "fast",
        "shuffled_text",
        "k_sweep",
        "synthetic",
    } <= names


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.stem)
def test_run_config_is_valid(path):
    rc = load_run_config(path)
    paper = load_paper()
    assert set(rc.arms) <= set(paper["arms"])
    assert set(rc.floor_arms) <= set(paper["arms"])
    assert all(a.startswith("null_random") for a in rc.floor_arms)
    main, transfer = task_lists(rc)
    assert main or transfer
    runs = rc.runs()
    if rc.fold_scheme is None:
        assert [r.name for r in runs] == ["fullfit"]
    else:
        assert len(runs) == rc.k_folds * len(rc.seeds)
        assert len({r.name for r in runs}) == len(runs)
    assert rc.battery.is_file()


def test_strict_matches_the_paper_protocol():
    """The headline numbers: 5 grouped folds x 3 seeds, d=32, K*=400, 10 floor draws."""
    rc = load_run_config(RUNS_DIR / "strict.yaml")
    assert (rc.fold_scheme, rc.k_folds, rc.seeds) == ("grouped_all", 5, (0, 1, 2))
    assert rc.dim == 32 and rc.interview_k == (400,) and rc.interview_selector == "fisher"
    assert rc.floor_draws == 10
    main_five = {
        "pca_bits32",
        "pls_svd_bitsqtext32",
        "text_mean768",
        "pls_svd_fusion32",
        "pls_svd_tri32",
    }
    assert main_five <= set(rc.arms)
    twins = {
        "pca_bits32__interview",
        "pls_svd_bitsqtext32__interview",
        "text_mean768__itext",
        "pls_svd_fusion32__itext",
        "pls_svd_tri32__itext",
    }
    assert twins <= set(rc.arms)


def test_k_sweep_writes_one_root_per_budget():
    rc = load_run_config(RUNS_DIR / "k_sweep.yaml")
    roots = list(rc.roots())
    assert [k for k, _ in roots] == [25, 50, 100, 200, 400]
    assert len({r for _, r in roots}) == 5


def test_shuffled_text_pairs_each_text_arm_with_its_twin():
    rc = load_run_config(RUNS_DIR / "shuffled_text.yaml")
    for arm in ("text_mean768", "pls_svd_fusion32", "pls_svd_tri32"):
        assert arm in rc.arms and f"{arm}_model_shuffled" in rc.arms
