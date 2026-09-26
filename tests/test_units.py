"""Fast unit tests of the core pieces: the question split, PLS-SVD widths, the floor,
grading, and the offline text encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bullpen.data.blocks import build_blocks, without_columns
from bullpen.floor import attach_floor, summary_table
from bullpen.models.pls_svd import split_width
from bullpen.models.registry import build_encoders
from bullpen.processing import HashingEncoder


# ---------------------------------------------------------------- question blocks
def _axes():
    # a deliberately unbalanced question axis: 600 / 300 / 100 columns
    return np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 100)


def test_question_blocks_partition_the_axis():
    blocks = build_blocks(_axes(), eval_size=130, tune_size=100, seed=1)
    e, t, p = (set(map(int, x)) for x in (blocks.eval_idx, blocks.tune_idx, blocks.pool_idx))
    assert not (e & t) and not (e & p) and not (t & p)
    assert e | t | p == set(range(1000))
    assert (blocks.n_eval, blocks.n_tune, blocks.n_pool) == (130, 100, 770)


def test_question_blocks_are_proportional_per_axis():
    axes = _axes()
    blocks = build_blocks(axes, eval_size=130, tune_size=100, seed=1)
    share = {a: (axes == a).mean() for a in "abc"}
    for idx, size in ((blocks.eval_idx, 130), (blocks.tune_idx, 100)):
        counts = pd.Series(axes[idx]).value_counts()
        for a in "abc":
            assert abs(counts[a] - size * share[a]) <= 1.5


def test_question_blocks_are_a_function_of_the_seed():
    a = build_blocks(_axes(), 130, 100, seed=3)
    b = build_blocks(_axes(), 130, 100, seed=3)
    c = build_blocks(_axes(), 130, 100, seed=4)
    assert np.array_equal(a.eval_idx, b.eval_idx)
    assert not np.array_equal(a.eval_idx, c.eval_idx)


def test_benchmark_holdout_only_touches_the_fitting_blocks():
    blocks = build_blocks(_axes(), 130, 100, seed=1)
    drop = np.arange(900, 1000)  # every "c" question
    held = without_columns(blocks, drop)
    assert np.array_equal(held.eval_idx, blocks.eval_idx)
    assert not np.isin(held.pool_idx, drop).any() and not np.isin(held.tune_idx, drop).any()


# ---------------------------------------------------------------- PLS-SVD widths
def test_split_width_bits_block_absorbs_the_remainder():
    assert split_width(32, blocks=2) == (16, 16)
    assert split_width(32, blocks=3) == (12, 10, 10)
    assert sum(split_width(9, blocks=3)) == 9
    with pytest.raises(ValueError):
        split_width(2, blocks=3)


def test_pls_svd_encoders_honour_the_width_and_block_split():
    rng = np.random.default_rng(0)
    n_models, n_q, d_text = 80, 150, 24
    theta = rng.normal(size=(n_models, 3))
    A = (theta @ rng.normal(size=(3, n_q)) + rng.normal(size=(n_models, n_q)) > 0).astype(float)
    text = rng.normal(size=(n_models, d_text))
    qtext = rng.normal(size=(n_q, d_text))
    rows = np.arange(60)
    cols = np.arange(n_q)
    encs = build_encoders(
        n_models=n_models,
        text_table=text,
        rows=rows,
        seed=0,
        dim=32,
        subset=["pls_svd_fusion32", "pls_svd_tri32", "pls_svd_bitsqtext32", "null_random32"],
        q_text_table=qtext,
    )
    widths = {
        "pls_svd_fusion32": (16, 16),
        "pls_svd_tri32": (12, 10, 10),
        "pls_svd_bitsqtext32": (16, 16),
    }
    for name, want in widths.items():
        enc = encs[name].fit(A[rows], cols)
        assert enc.X.shape == (60, 32)
        assert tuple(enc.block_widths) == want
        # a held-out model is placed without refitting the basis
        z = enc.refit(A[70], cols, 70)
        assert z.shape[-1] == 32 and np.isfinite(z).all()
    null = encs["null_random32"].fit(A[rows], cols)
    assert null.X.shape == (60, 32)


# ---------------------------------------------------------------- the floor
def _rows():
    return pd.DataFrame(
        {
            "method": ["arm32", "arm768", "null_random32", "null_random768"],
            "task": ["t"] * 4,
            "group": [""] * 4,
            "s": [0.50, 0.40, 0.10, 0.20],
            "diag_emb_dim": [32, 768, 32, 768],
        }
    )


def test_floor_is_the_nearest_width_single_draw_without_a_multi_draw():
    out = attach_floor(_rows()).set_index("method")
    assert out.at["arm32", "floor_arm"] == "null_random32"
    assert out.at["arm768", "floor_arm"] == "null_random768"
    assert out.at["arm32", "s_over_floor"] == pytest.approx(0.40)
    assert out.at["arm768", "s_over_floor"] == pytest.approx(0.20)
    assert out.at["arm32", "floor_draws"] == 1


def test_multi_draw_floor_replaces_the_single_draw():
    floors = pd.DataFrame(
        {
            "method": ["null_random32"],
            "task": ["t"],
            "group": [""],
            "s": [0.05],
            "draws": [10],
            "sd": [0.02],
        }
    )
    out = attach_floor(_rows(), floors).set_index("method")
    assert out.at["arm32", "floor_s"] == pytest.approx(0.05)
    assert out.at["arm32", "floor_draws"] == 10
    assert out.at["arm32", "s_over_floor"] == pytest.approx(0.45)
    # a width with no multi-draw floor keeps its own single draw
    assert out.at["arm768", "floor_s"] == pytest.approx(0.20)


def test_an_odd_width_reads_the_nearest_floor():
    rows = _rows()
    rows.loc[0, "diag_emb_dim"] = 49
    out = attach_floor(rows).set_index("method")
    assert out.at["arm32", "floor_arm"] == "null_random32"


def test_summary_averages_over_runs():
    a = attach_floor(_rows()).assign(run="fold0_s0")
    b = attach_floor(_rows().assign(s=[0.7, 0.4, 0.1, 0.2])).assign(run="fold1_s0")
    scores = pd.concat([a, b]).rename(columns={"method": "arm"})
    summary = summary_table(scores).set_index("arm")
    assert summary.at["arm32", "s_over_floor"] == pytest.approx(0.5)
    assert summary.at["arm32", "n_runs"] == 2


# ---------------------------------------------------------------- grading, text
def test_grading_reads_the_answer_protocols():
    from bullpen.collection import ProbeItem, grade_answer

    mcq = ProbeItem("q", "mmlu_anatomy", "?", "B", "bench:mmlu_anatomy", "mcq", 4)
    assert grade_answer(mcq, "The answer is (B).")
    assert grade_answer(mcq, "Final answer: \\boxed{B}")
    assert not grade_answer(mcq, "The answer is (C).")
    assert not grade_answer(mcq, "I'm sorry, but I can't help with that request.")
    free = ProbeItem("q", "asdiv", "?", "15", "bench:asdiv", "free")
    assert grade_answer(free, "The total is \\boxed{15}.")
    assert not grade_answer(free, "The total is \\boxed{16}.")


def test_hashing_encoder_is_unit_norm_at_its_width():
    enc = HashingEncoder(16)
    x = enc.encode(["the answer is B", "I think the answer is C", ""])
    assert x.shape == (3, 16)
    assert np.allclose(np.linalg.norm(x[:2], axis=1), 1.0, atol=1e-5)
    assert np.allclose(x[2], 0.0)
    again = HashingEncoder(16).encode(["the answer is B", "I think the answer is C", ""])
    assert np.array_equal(x, again)


# ---------------------------------------------------------------- packed answer bank
@pytest.fixture
def bank():
    rng = np.random.default_rng(0)
    Ae = rng.normal(size=(7, 5, 16)).astype(np.float32)
    Ae /= np.linalg.norm(Ae, axis=-1, keepdims=True)
    Ae[2, 3] = 0.0  # an uncovered cell: zero vector, mask False
    mask = np.ones((7, 5), dtype=bool)
    mask[2, 3] = False
    Qe = rng.normal(size=(5, 16)).astype(np.float32)
    return Ae.astype(np.float16), mask, Qe


def test_packed_bank_splits_under_the_cap_and_round_trips(tmp_path, bank):
    from bullpen.data.slice import _load_answers, bank_exists, packed_parts, save_packed_answers

    Ae, mask, Qe = bank
    path = tmp_path / "answers.core.npz"
    # 2 models x 5 questions x 16 dims of int8 per part -> 4 parts for 7 models
    parts = save_packed_answers(Ae, mask, Qe, path, part_mb=2 * 5 * 16 / 1e6)
    assert len(parts) == 4 and parts == packed_parts(path)
    assert not path.exists() and bank_exists(path)
    got, got_mask, got_Qe = _load_answers(path, (7, 5), mmap=True)
    assert got.dtype == np.float16 and got.shape == Ae.shape
    assert np.array_equal(got_mask, mask) and np.array_equal(got_Qe, Qe)
    a, b = got.astype(np.float32)[mask], Ae.astype(np.float32)[mask]
    cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    assert cos.min() > 0.999
    assert np.all(got[2, 3] == 0)


def test_a_whole_bank_is_preferred_over_its_parts(tmp_path, bank):
    from bullpen.data.slice import _load_answers, save_packed_answers

    Ae, mask, Qe = bank
    path = tmp_path / "answers.npz"
    save_packed_answers(Ae, mask, Qe, path)
    np.savez(path, Ae=Ae, Ae_mask=mask, Qe=Qe)
    got, _, _ = _load_answers(path, (7, 5), mmap=False)
    assert np.array_equal(got, Ae.astype(np.float32))


def test_a_packed_bank_with_a_missing_part_is_refused(tmp_path, bank):
    from bullpen.data.slice import _load_answers, save_packed_answers

    Ae, mask, Qe = bank
    parts = save_packed_answers(Ae, mask, Qe, tmp_path / "answers.npz", part_mb=2 * 5 * 16 / 1e6)
    parts[1].unlink()
    with pytest.raises(ValueError, match="starts at row"):
        _load_answers(tmp_path / "answers.npz", (7, 5), mmap=True)


def test_recorded_responses_read_xz(tmp_path):
    import json
    import lzma

    from bullpen.collection import recorded_responses

    rows = [{"id": "q1", "prediction": "The answer is (B)."}, {"id": "q2", "prediction": " x"}]
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (tmp_path / "org__m.jsonl.xz").write_bytes(lzma.compress(data))
    assert recorded_responses(tmp_path, "org__m") == {"q1": "The answer is (B).", "q2": " x"}
