# BULLPEN

Code for "From the Zoo to the BULLPEN: Sample-Efficient LLM Profiling via Multi-Channel Model
Embeddings" (anonymous submission). BULLPEN embeds each language model from three channels of
its answers to a fixed probe (correctness bits, question text, answer text), fused by
closed-form multi-block PLS-SVD. Every score is a chance-anchored skill `s` on held-out models
and questions, read against a width-matched Gaussian floor (`null_random{d}`).

## Install

```bash
uv sync                    # Python 3.12; `uv sync --extra collect` adds transformers (real models, BGE)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # fits are bit-identical only at a fixed thread count
```

## Sample run (offline, real data, about 15 minutes)

```bash
uv run python collect.py --probe data/core/probe.jsonl --models-file data/core/models.txt \
    --recorded data/core/responses --raw results/sample/raw          # regrade shipped answers
uv run python process.py build --probe data/core/probe.jsonl --raw results/sample/raw \
    --meta data/raw --labels data/paper_ckpt/labels --out results/sample/cut
uv run python process.py embed --probe data/core/probe.jsonl --raw results/sample/raw \
    --out results/sample/cut --encoder hashing:32                   # --encoder hf for BGE-base
uv run python train.py --config config/runs/sample.yaml
uv run python eval.py  --config config/runs/sample.yaml --no-bootstrap
```

The sample draws its own question blocks and folds, so it illustrates the pipeline, not the
paper's numbers.

## Reproducing the paper

`train.py` fits a run config's arms per model split; `eval.py` scores them and writes
`results/<config>/summary.csv` (per arm and task: `s`, `floor_s`, `s_over_floor`, averaged over
runs). `--run fold0_s0` restricts either script to some of the 15 runs (5 publisher-grouped
folds x 3 seeds). Fits are bit-reproducible on CPU; the neural competitors use a GPU when one
is present, where `comp_llmmap_orig` is not bit-reproducible run to run.

```bash
for c in strict strict_competitors k_sweep; do    # Table 3 (BULLPEN rows, other rows), K sweep
    uv run python train.py --config config/runs/$c.yaml
    uv run python eval.py  --config config/runs/$c.yaml
done
```

Also: `shuffled_text.yaml` (shuffled-text controls), `strict_transfer.yaml`
(leave-one-benchmark-group-out transfer), `fast.yaml` (one fullfit run). `uv run pytest` runs
the tests, including an end-to-end run on a synthetic fixture.

## Data

`data/` holds a real subset of the paper's cut: 260 models x the 1,181 core questions, the view
every run config reads (`data/README.md` lists the files). Answer embeddings are stored as int8
(cosine >= 0.9997 to the fp16 originals). Bits-only arms read no answer embeddings; the text
arms' single-task scores can move slightly against the paper (arm means within about 0.006 on
one strict run). The full checkpoint (all 31,286 questions, the fp16 answer bank, raw generation
logs), needed for the full-view K sweep, will be linked here.

## Arm names

| config arm | paper |
|---|---|
| `pca_bits32`, `pls_svd_bitsqtext32`, `text_mean768` | PCA32-bits, BULLPEN-bits+q, BULLPEN-ans |
| `pls_svd_fusion32`, `pls_svd_tri32` | BULLPEN-bits+ans, BULLPEN-fused |
| `<arm>__interview`, `<arm>__itext` | the -budget twin (K-question Fisher interview) |
| `<arm>_model_shuffled`, `<arm>_question_shuffled` | shuffled-text controls |
| `null_random{d}`, `null_leaderboard`, `bits_irt2pl` | Gaussian floor, leaderboard null, IRT-2PL |
| `comp_embedllm`, `comp_locus`, `comp_routellm_mf` | EmbedLLM, LOCUS, RouteLLM-MF |
| `comp_irtrouter`, `comp_jeirt`, `comp_irtnet` | IRT-Router, JE-IRT, IrtNet |
| `comp_llmmap_orig`, `comp_llmdna` | LLMmap (retrained), LLM DNA |
