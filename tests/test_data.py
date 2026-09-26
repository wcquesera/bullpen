"""The shipped core subset: 260 models x 1,181 core questions, aligned across its files,
every file under GitHub's 100 MB cap. Skipped where data/ holds no subset."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CKPT = DATA / "paper_ckpt"
CORE = DATA / "core"
N_MODELS, N_CORE = 260, 1181

pytestmark = pytest.mark.skipif(not (CKPT / "slice.npz").is_file(), reason="no data subset")


@pytest.fixture(scope="module")
def core_slice():
    from bullpen.data.slice import load_slice

    return load_slice(CKPT / "slice.npz", CKPT / "answers.npz", view="core")


def test_every_shipped_file_is_under_the_github_cap():
    big = [p for p in DATA.rglob("*") if p.is_file() and p.stat().st_size >= 100e6]
    assert not big


def test_core_view_is_finite_and_carries_the_answer_bank(core_slice):
    sl = core_slice
    assert sl.A.shape == (N_MODELS, N_CORE) and np.isfinite(sl.A).all()
    assert sl.Ae.shape == (N_MODELS, N_CORE, 768) and sl.Qe.shape == (N_CORE, 768)
    assert sl.Ae_mask.mean() > 0.99
    norms = np.linalg.norm(sl.Ae[sl.Ae_mask][:1000].astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=0.01)


def test_question_blocks_restrict_to_the_paper_core_split(core_slice):
    from bullpen.data.blocks import blocks_for_slice

    b = blocks_for_slice(core_slice, path=CKPT / "question_blocks.json")
    assert (b.n_eval, b.n_tune, b.n_pool) == (159, 122, 900)


def test_probe_and_responses_align_with_the_slice(core_slice):
    probe = [json.loads(line) for line in (CORE / "probe.jsonl").read_text().splitlines()]
    assert [r["id"] for r in probe] == core_slice.question_ids
    models = (CORE / "models.txt").read_text().split()
    assert models == core_slice.model_ids
    files = sorted((CORE / "responses").glob("*.jsonl.xz"))
    assert {f.name.removesuffix(".jsonl.xz") for f in files} == set(models)
    rows = [json.loads(x) for x in lzma.decompress(files[0].read_bytes()).decode().splitlines()]
    assert set(r["id"] for r in rows) <= set(core_slice.question_ids)
    assert all({"id", "prediction", "in_bank"} <= set(r) for r in rows)
