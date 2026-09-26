"""The all-mpnet-base-v2 question table (``data/question_text_mpnet.npz``).

Upstream EmbedLLM embeds each question with
``SentenceTransformer('all-mpnet-base-v2').encode(questions)``; :func:`build` makes the
same call on our probe's question text (our prompts, not upstream's harness strings, so
the released embeddings cannot be reused). The build needs ``sentence-transformers``
(``python -m bullpen.competitors.embedllm.bridge``); loading needs only numpy. The
EmbedLLM, LOCUS and IRT-family competitors read this table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bullpen.competitors.embedllm.textmf import FAITHFUL_QUESTION_ENCODER, FAITHFUL_TEXT_DIM

#: Where train.py looks for the table; if absent, the arms reading it are skipped.
DEFAULT_BRIDGE = Path(__file__).resolve().parents[3] / "data" / "question_text_mpnet.npz"
ENCODER_TAG: str = "all_mpnet_base_v2"


def load(path: Path | str, question_ids: list[str] | np.ndarray) -> np.ndarray:
    """``[Q, 768]`` table in the order of ``question_ids``, joined by id; missing ids raise."""
    z = np.load(Path(path), allow_pickle=False)
    if str(z["encoder"]) != ENCODER_TAG:
        raise ValueError(f"{path}: encoder {z['encoder']!s}, expected {ENCODER_TAG}")
    table = np.asarray(z["Qe"], dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != FAITHFUL_TEXT_DIM:
        raise ValueError(f"{path}: Qe shape {table.shape}, expected [Q, {FAITHFUL_TEXT_DIM}]")
    at = {str(q): i for i, q in enumerate(z["question_ids"])}
    want = [str(x) for x in question_ids]
    missing = [q for q in want if q not in at]
    if missing:
        raise ValueError(f"{path}: {len(missing)} questions not in the bridge, e.g. {missing[:3]}")
    return table[[at[q] for q in want]]


def load_if_present(
    question_ids: list[str] | np.ndarray, path: Path | str = DEFAULT_BRIDGE
) -> np.ndarray | None:
    """:func:`load`, or ``None`` when the bridge has not been built."""
    return load(path, question_ids) if Path(path).exists() else None


def build(probe_jsonl: Path, slice_npz: Path, out: Path, batch_size: int = 64) -> None:
    from sentence_transformers import SentenceTransformer

    text = {}
    with open(probe_jsonl) as fh:
        for line in fh:
            row = json.loads(line)
            text[row["id"]] = row["question"]
    ids = [str(x) for x in np.load(slice_npz, allow_pickle=True)["question_ids"]]
    missing = [q for q in ids if q not in text]
    if missing:
        raise ValueError(f"{len(missing)} slice questions have no text, e.g. {missing[:3]}")
    model = SentenceTransformer(FAITHFUL_QUESTION_ENCODER, device="cpu")
    emb = model.encode([text[q] for q in ids], batch_size=batch_size, show_progress_bar=True)
    np.savez(
        out, Qe=np.asarray(emb, dtype=np.float32), question_ids=np.array(ids), encoder=ENCODER_TAG
    )


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--slice", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.probe, a.slice, a.out)
