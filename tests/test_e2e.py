"""The four entry points in sequence on the synthetic fixture: collect -> process -> train -> eval.

Offline end to end: collect grades the pre-recorded responses in
tests/fixtures/synthetic/recorded, process embeds with the hashing encoder, and the run is
tests/fixtures/synthetic.yaml (one fold of three, no bootstrap) redirected into a temporary
directory. Takes about two minutes.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "tests" / "fixtures" / "synthetic"
MODELS = SAMPLE.joinpath("models.txt").read_text().split()
N_MODELS, N_QUESTIONS, TEXT_DIM = len(MODELS), 160, 32
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
MAIN_FIVE = [
    "pca_bits32",
    "pls_svd_bitsqtext32",
    "text_mean768",
    "pls_svd_fusion32",
    "pls_svd_tri32",
]


def run(*args: str) -> None:
    proc = subprocess.run(
        [sys.executable, *args], cwd=REPO, env=ENV, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"{args[0]} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("sample")
    raw, cut, out = tmp / "raw", tmp / "cut", tmp / "results"
    probe = str(SAMPLE / "probe.jsonl")
    run(
        "collect.py",
        "--probe",
        probe,
        "--models-file",
        str(SAMPLE / "models.txt"),
        "--recorded",
        str(SAMPLE / "recorded"),
        "--raw",
        str(raw),
    )
    run(
        "process.py",
        "build",
        "--probe",
        probe,
        "--raw",
        str(raw),
        "--meta",
        str(SAMPLE / "raw"),
        "--labels",
        str(SAMPLE / "labels"),
        "--out",
        str(cut),
    )
    run(
        "process.py",
        "embed",
        "--probe",
        probe,
        "--raw",
        str(raw),
        "--out",
        str(cut),
        "--encoder",
        f"hashing:{TEXT_DIM}",
    )
    cfg = yaml.safe_load((REPO / "tests" / "fixtures" / "synthetic.yaml").read_text())
    cfg["data"].update(slice=str(cut / "slice.npz"), answers=str(cut / "answers.npz"))
    cfg["out"] = str(out)
    config = tmp / "sample.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run("train.py", "--config", str(config), "--run", "fold0_s0")
    run("eval.py", "--config", str(config), "--run", "fold0_s0", "--no-bootstrap")
    return {"raw": raw, "cut": cut, "out": out, "config": config}


def test_collect_writes_one_graded_file_per_model(pipeline):
    files = sorted((pipeline["raw"] / "responses").glob("*/results.jsonl"))
    assert len(files) == N_MODELS
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(rows) == N_QUESTIONS
    assert {"id", "axis", "correct", "prediction", "model"} <= set(rows[0])
    assert {r["correct"] for r in rows} <= {0, 1}


def test_process_builds_a_finite_slice_and_disjoint_blocks(pipeline):
    z = np.load(pipeline["cut"] / "slice.npz")
    assert z["R"].shape == (N_MODELS, N_QUESTIONS)
    assert set(np.unique(z["R"])) <= {0, 1}
    assert np.isfinite(z["A"]).all()
    acc = z["R"].mean(axis=1)
    assert 0.2 < acc.min() and acc.max() < 1.0
    assert np.isfinite(z["vram_gb"]).all()
    blocks = json.loads((pipeline["cut"] / "question_blocks.json").read_text())
    idx = [set(blocks[k]) for k in ("eval", "tune", "pool")]
    assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
    assert set().union(*idx) == set(range(N_QUESTIONS))
    assert (pipeline["cut"] / "labels" / "external_labels.json").is_file()


def test_embed_writes_banks_at_the_configured_width(pipeline):
    z = np.load(pipeline["cut"] / "answers.npz")
    assert z["Ae"].shape == (N_MODELS, N_QUESTIONS, TEXT_DIM)
    assert z["Qe"].shape == (N_QUESTIONS, TEXT_DIM)
    assert z["Ae_mask"].all()
    norms = np.linalg.norm(z["Ae"].astype(np.float32), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-2)


def test_train_fits_every_arm_without_touching_the_eval_block(pipeline):
    run_dir = pipeline["out"] / "fold0_s0"
    manifest = json.loads((run_dir / "fits" / "manifest.json").read_text())
    arms = {a["arm"] for a in manifest["arms"]}
    assert set(MAIN_FIVE) <= arms
    assert {f"{a}__interview" for a in ("pca_bits32", "pls_svd_bitsqtext32")} <= arms
    assert {f"{a}__itext" for a in ("text_mean768", "pls_svd_fusion32", "pls_svd_tri32")} <= arms
    assert not manifest["skipped"]
    train, test = set(manifest["split"]["train"]), set(manifest["split"]["test"])
    assert train and test and not (train & test)
    eval_cols = set(manifest["questions"]["eval"])
    for a in manifest["arms"]:
        payload = pickle.loads((run_dir / "fits" / a["file"]).read_bytes())
        assert not eval_cols & set(map(int, payload["cols"])), a["arm"]
        if "interview" in a["blocks"]:
            assert a["n_cols"] == 40
    widths = {a["arm"]: a["dim"] for a in manifest["arms"]}
    assert widths["pls_svd_tri32"] == 32 and widths["text_mean768"] == TEXT_DIM
    tri = pickle.loads((run_dir / "fits" / "pls_svd_tri32__s0.pkl").read_bytes())["encoder"]
    assert tuple(tri.block_widths) == (12, 10, 10)
    floor = json.loads((run_dir / "floor" / "fits" / "manifest.json").read_text())
    assert sorted(a["seed"] for a in floor["arms"]) == [0, 1, 2]


def test_grouped_folds_keep_a_family_on_one_side(pipeline):
    manifest = json.loads((pipeline["out"] / "fold0_s0" / "fits" / "manifest.json").read_text())
    family = {m: m.split("__")[0] for m in MODELS}
    train = {family[m] for m in manifest["split"]["train_model_ids"]}
    test = {family[m] for m in manifest["split"]["test_model_ids"]}
    assert not train & test


def test_eval_writes_skill_over_the_width_matched_floor(pipeline):
    scores = pd.read_csv(pipeline["out"] / "scores.csv")
    summary = pd.read_csv(pipeline["out"] / "summary.csv")
    for col in ("s", "floor_s", "s_over_floor", "floor_arm", "floor_draws"):
        assert col in scores.columns
    main = scores[scores.group.isna() | (scores.group == "")]
    assert set(MAIN_FIVE) <= set(main.arm)
    assert (main.floor_arm == "null_random32").all()
    scored = main[main.s_over_floor.notna()]
    assert len(scored) and (scored.floor_draws == 3).all()
    # the probe tasks need nothing but the bits: every arm scores them
    for task in ("score_regression", "specialisation", "correctness_forecasting"):
        cells = main[main.task == task]
        assert len(cells) and cells.s_over_floor.notna().all(), task
    # the two sample leaderboards and the categorical family target are scored
    assert {
        "open_llm_leaderboard",
        "chatbot_arena_elo",
        "family_classification",
        "base_vs_chat",
    } & set(main.task)
    assert set(scores.group.dropna()) >= {"mean", "arithmetic"}
    by_arm = main.groupby("arm").s_over_floor.mean()
    assert by_arm["pls_svd_tri32"] > by_arm["null_random32"]
    assert by_arm[MAIN_FIVE].gt(0).all()
    assert len(summary) and summary.n_runs.max() == 1
