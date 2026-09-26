"""Regenerate ``tests/fixtures/synthetic/``: a small synthetic stand-in in the real formats.

    uv run python tests/fixtures/make_sample_data.py

Everything is invented: 72 fictional models (12 publisher families x 6) answer 160
templated questions on four benchmark axes. Correctness follows a latent-trait model
(general skill, a math/verbal specialisation, a family offset, log size), and the
answer TEXT carries each family's own style (answer template, verbosity, hedging,
refusals), so both the bits and the text channel have something to find. Output:

    probe.jsonl                 the questions, in the probe format collect.py reads
    models.txt                  the model ids
    recorded/<model>.jsonl      one pre-recorded response per question
    raw/model_footprints.json   family, size, VRAM per model (the grouped folds use family)
    raw/model_release_dates.json
    labels/external_labels.json two published-board stand-ins (numeric targets)
    benchmark_groups.yaml       the leave-one-group-out partition of the four axes

The categorical targets (family, base vs chat) come from the footprints and the names.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "synthetic"
SEED = 7

FAMILIES = ["aurora", "boreal", "cobalt", "dune", "ember", "fjord",
            "garnet", "harbor", "iris", "juniper", "kestrel", "lumen"]
SIZES = [1.5, 3.0, 7.0, 7.0, 13.0, 34.0]
VARIANTS = ["base", "chat", "base", "instruct", "chat", "instruct"]
AXES = ["asdiv", "mathqa", "logiqa", "mmlu_anatomy"]
N_PER_AXIS = 40
LETTERS = "ABCD"

TEMPLATES = [
    "The answer is ({x}).",
    "Answer: {x}",
    "Final answer: \\boxed{{{x}}}",
    "So the correct choice is ({x}).",
]
HEDGES = ["I think ", "Probably ", "If I am not mistaken, ", "It seems that "]
FILLER = ["Let me work through this carefully.", "First, read the question.",
          "We compare each option in turn.", "Consider the quantities involved.",
          "Checking the arithmetic once more.", "This follows from the definitions."]


def questions(rng: np.random.Generator) -> list[dict]:
    rows = []
    for axis in AXES:
        for k in range(N_PER_AXIS):
            qid = f"{axis}-{k}"
            a, b = int(rng.integers(2, 40)), int(rng.integers(2, 40))
            if axis == "asdiv":
                rows.append({
                    "id": qid, "axis": axis, "grader": f"bench:{axis}", "format": "free",
                    "question": f"A crate holds {a} pears and {b} more are added. How many pears "
                                "are in the crate?\n\nGive your final answer as a number inside "
                                "\\boxed{}.",
                    "gold": str(a + b),
                })
                continue
            if axis == "mathqa":
                right = a * b
                stem = f"What is {a} multiplied by {b}?"
                wrong = [right + d for d in (-b, b, a)]
            elif axis == "logiqa":
                right = a + 2 * b
                stem = (f"Every lantern in row {a} is lit, and row {a} has {b} pairs of lanterns "
                        f"plus {a} single ones. How many lanterns are lit?")
                wrong = [right - 1, right + b, a + b]
            else:
                right = a + b + 1
                stem = (f"A model skeleton has {a} rib pieces and {b} spine pieces plus one "
                        "skull. How many pieces in total?")
                wrong = [right + 1, right - 2, a * 2]
            opts = [right, *wrong]
            order = rng.permutation(4)
            options = [opts[i] for i in order]
            gold = LETTERS[int(np.flatnonzero(order == 0)[0])]
            lines = [f"{L}) {o}" for L, o in zip(LETTERS, options, strict=True)]
            text = stem + "\n" + "\n".join(lines)
            rows.append({
                "id": qid, "axis": axis, "grader": f"bench:{axis}", "format": "mcq",
                "n_choices": 4, "question": text + "\n\nAnswer with the letter of the option.",
                "gold": gold,
            })
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    probe = questions(rng)
    n_q = len(probe)
    is_math = np.array([p["axis"] in ("asdiv", "mathqa") for p in probe], dtype=float)
    difficulty = rng.normal(0.0, 1.0, n_q)
    loading = rng.normal(0.0, 0.8, n_q)

    models, footprints, dates, meta = [], {}, {}, {}
    for f_i, fam in enumerate(FAMILIES):
        fam_skill = rng.normal(0.0, 0.5)
        fam_spec = rng.normal(0.0, 0.8)
        style = {"template": f_i % len(TEMPLATES), "verbosity": int(rng.integers(0, 4)),
                 "hedge": float(rng.uniform(0.0, 0.6)), "refusal": float(rng.uniform(0.0, 0.08))}
        for size, variant in zip(SIZES, VARIANTS, strict=True):
            name = f"{fam}-{size:g}b-{variant}"
            mid = f"{fam}-labs__{name}"
            if mid in meta:
                mid = f"{fam}-labs__{name}-v2"
            models.append(mid)
            chat = variant != "base"
            meta[mid] = {
                "skill": fam_skill + 0.6 * np.log(size) + (0.3 if chat else 0.0)
                + rng.normal(0.0, 0.3),
                "spec": fam_spec + rng.normal(0.0, 0.3),
                "chat": chat, "style": style, "family": fam,
            }
            footprints[mid] = {"dir": mid, "hf_id": mid.replace("__", "/"), "family": fam,
                               "params_b": size, "quant": "fp16", "serve_quant": "bf16",
                               "vram_gb": round(2.02 * size, 2),
                               "weight_bytes": int(2.02e9 * size)}
            year = 2023 + int(meta[mid]["skill"] > 1.2)
            dates[mid] = {"created_at": f"{year}-{int(rng.integers(1, 13)):02d}-15"}

    (OUT / "recorded").mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "recorded").glob("*.jsonl"):
        old.unlink()
    with (OUT / "probe.jsonl").open("w") as fh:
        for row in probe:
            fh.write(json.dumps(row) + "\n")
    (OUT / "models.txt").write_text("\n".join(models) + "\n")

    for mid in models:
        m = meta[mid]
        st = m["style"]
        logit = m["skill"] - difficulty + loading * m["spec"] * (2 * is_math - 1)
        p = 1.0 / (1.0 + np.exp(-logit))
        right = rng.random(n_q) < p
        lines = []
        for q, item in enumerate(probe):
            if rng.random() < st["refusal"]:
                text = "I'm sorry, but I can't help with that request."
            else:
                if item["format"] == "free":
                    off = int(rng.integers(1, 5))
                    ans = item["gold"] if right[q] else str(int(item["gold"]) + off)
                    body = f"The total is \\boxed{{{ans}}}."
                else:
                    wrong = [L for L in LETTERS if L != item["gold"]]
                    ans = item["gold"] if right[q] else wrong[(q + len(m["family"])) % 3]
                    body = TEMPLATES[st["template"]].format(x=ans)
                n_fill = st["verbosity"] + (1 if m["chat"] else 0)
                fill = " ".join(FILLER[(q + i) % len(FILLER)] for i in range(n_fill))
                hedge = HEDGES[q % len(HEDGES)] if rng.random() < st["hedge"] else ""
                body = body[0].lower() + body[1:] if hedge else body
                text = (fill + " " if fill else "") + hedge + body
            lines.append(json.dumps({"id": item["id"], "prediction": text}))
        (OUT / "recorded" / f"{mid}.jsonl").write_text("\n".join(lines) + "\n")

    stamp = {"generated_at": "synthetic", "generated_by": "tests/fixtures/make_sample_data.py"}
    (OUT / "raw" / "model_footprints.json").write_text(json.dumps(
        {**stamp, "models": footprints, "n_matched": len(models), "n_models": len(models)},
        indent=1))
    (OUT / "raw" / "model_release_dates.json").write_text(json.dumps(
        {**stamp, "models": dates, "n_dated": len(models), "n_models": len(models)}, indent=1))

    board = {mid: round(45 + 8 * meta[mid]["skill"] + rng.normal(0, 2), 2)
             for mid in models if rng.random() < 0.9}
    arena = {mid: round(1000 + 60 * meta[mid]["skill"] + 40 * meta[mid]["style"]["verbosity"]
                        + rng.normal(0, 15), 1)
             for mid in models if meta[mid]["chat"]}
    tables = {
        name: {"n_alias": 0, "n_exact": len(scores), "n_matched": len(scores),
               "n_published": len(scores), "n_unmatched_published": 0, "scores": scores}
        for name, scores in (("open_llm_leaderboard", board), ("chatbot_arena_elo", arena))
    }
    (OUT / "labels" / "external_labels.json").write_text(json.dumps(
        {**stamp, "collected_at": "synthetic", "model_ids": models, "source": "synthetic",
         "tables": tables}, indent=1))
    (OUT / "benchmark_groups.yaml").write_text(
        "# The leave-one-group-out partition of the sample's four benchmark axes.\n"
        "groups:\n  arithmetic: [asdiv, mathqa]\n  logic: [logiqa]\n  anatomy: [mmlu_anatomy]\n"
    )
    print(f"wrote {OUT}: {len(models)} models x {n_q} questions")


if __name__ == "__main__":
    main()
