# Data: the core subset

Everything here is real data from the paper's frozen cut, restricted to its 1,181-question
core: 260 models (public Hugging Face ids) x 1,181 questions from 68 benchmarks. Every paper
run config reads the cut through this core view (`view: core`), so the subset is enough to
run Table 3's protocol (`config/runs/strict.yaml`, `strict_competitors.yaml`). About 270 MB;
no file exceeds 71 MB.

```
paper_ckpt/                       the frozen cut on the core axis, in the layout the run configs read
  slice.npz                       correctness bits A/R [260, 1181], benchmark index, model and
                                  question ids, VRAM, release dates, fullset/extset flags
  answers.core.part0{0,1,2}.npz   BGE-base answer embeddings [260, 1181, 768] as int8 with a
                                  per-cell scale, plus the question embeddings Qe [1181, 768]
  question_blocks.json            the frozen eval / tune / pool split, restricted to the core
                                  (159 / 122 / 900 questions)
  folds/                          publisher-grouped fold plans over models (seeds 0-4), one random plan
  labels/                         label tables: leaderboard labels, scraped boards, per-question
                                  reward-model/refusal/hedge labels, answer text traits
  llmmap_probe_bank.npz           embedded answers to LLMmap's 8 published probes (comp_llmmap_orig;
                                  247 of 260 models)
  qc_report.json                  the gates the cut was built under, and who they dropped
raw/                              model metadata and label sources joined at scoring time
  model_footprints.json, model_release_dates.json, hf_model_cards/,
  external_survey*/, external_labels_src/model_aliases.json
question_text_mpnet.npz   competitor question encoder (all-mpnet-base-v2) on the core questions
core/                             the raw material for the sample pipeline (README, "Sample pipeline")
  probe.jsonl                     the 1,181 questions: id, benchmark, prompt text, gold answer, grader
  models.txt                      the 260 model ids, in slice order
  responses/<model>.jsonl.xz      each model's response text per question ({id, prediction, in_bank})
```

**Answer embeddings.** The paper's bank is fp16 (476 MB); here each 768-d vector is stored
as int8 scaled by its own maximum (`process.py pack`). Cosine to the fp16 original is at least
0.9997 on every cell. On one strict run each arm's mean over tasks moved by at most 0.006 and
the ranking of arms held, while single task cells of the text arms moved by up to 0.09 (see
the main README). If the fp16 `answers.core.npz` is present beside the parts, the loaders
use it instead.

**Response text.** 285,520 of the 307,060 cells carry text. For 278,595 of them it is the
exact text the answer embedding was computed from (matched by SHA-1). `in_bank: false`
marks 6,925 cells whose logged text was regenerated after embedding, so it is the current
log text and not the embedded text. 21,540 cells (20,548 of them in 25 models) have no
text in the released logs. The correctness bits in `paper_ckpt/` cover every cell, and the
answer embeddings cover 99.8% of cells (`Ae_mask`).

**Not in the subset.** The 30,105 non-core questions (only the 104 fullset models answered
them), the full-axis answer bank, raw generation logs, and the question text of non-core
questions. These are needed for the full-view results (the 24,118-question pool K sweep) and
for rebuilding the cut from scratch. See the main README, "Full processed checkpoint".
