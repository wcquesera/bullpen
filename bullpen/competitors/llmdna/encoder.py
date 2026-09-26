"""``comp_llmdna``: LLM DNA (Wu et al., ICLR 2026, arXiv 2509.24496), retrained on our answers.

RepTrace (Algorithm 1) is training-free: embed each model's responses to ``t`` fixed
prompts, concatenate them into ``E_f in R^{p*t}`` and project with one Gaussian matrix
``A in R^{L x p*t}``, ``A_jk ~ N(0, 1/L)``: ``tau_f = A E_f``, with ``L = 128`` and
``t = 600``. Kept exactly: concatenation, the projection, ``L``, ``t``, no training.
Deviations: the prompts are 600 questions drawn (seeded) from the pool block; the
responses are those already in the answer store; the encoder is the store's BGE-base
(p = 768) instead of Qwen3-Embedding-8B; a missing response (0.2% of cells) takes the
training models' mean embedding on that prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from bullpen.models.base import Encoder

#: DNA dimensionality L (paper App. B.1).
LLMDNA_DIM: int = 128

#: Prompt count t (paper §5.1: six datasets x 100 random samples).
LLMDNA_PROMPTS: int = 600

#: Prompts read per chunk of the answer bank (bounds peak memory).
CHUNK: int = 50


@dataclass
class LLMDNAEncoder(Encoder):
    """Concatenated response embeddings over t fixed prompts, Gaussian-projected to L."""

    dim: int = LLMDNA_DIM
    #: ``[M_all, Q_all, Da]`` frozen answer embeddings (the slice's ``Ae``); read in
    #: ``fit`` and dropped afterwards, so the artifact carries only the DNA table
    answer_bank: np.ndarray | None = field(default=None, repr=False)
    #: ``[M_all, Q_all]`` bool coverage of ``answer_bank``; ``None`` = an all-zero
    #: embedding marks a missing cell
    answer_mask: np.ndarray | None = field(default=None, repr=False)
    #: which rows of ``answer_bank`` this fold's ``A`` rows are
    rows: np.ndarray | None = field(default=None, repr=False)
    n_prompts: int = LLMDNA_PROMPTS

    #: global ids of the t sampled prompts, sorted
    prompts: np.ndarray | None = field(default=None, repr=False)
    #: ``[M_all, L]`` DNA of every model on the model axis
    table: np.ndarray | None = field(default=None, repr=False)
    #: number of (model, prompt) cells filled with the training mean
    n_filled: int = 0

    uses_probe: bool = False
    has_decoder: bool = False
    #: every model answers the same fixed prompts; the interview is irrelevant
    interview_requirement: str = "nothing"

    def fit(self, A: np.ndarray, cols: np.ndarray) -> LLMDNAEncoder:
        """Sample the prompts from ``cols``, project every model's concatenated answers."""
        A = self._prepare(A, cols)
        if self.answer_bank is None:
            raise ValueError(
                f"{self.name} needs the answer bank; LLM DNA is a projection of a model's "
                "response embeddings, so without them it is not LLM DNA"
            )
        bank = self.answer_bank
        n_all = int(bank.shape[0])
        rows = np.arange(n_all) if self.rows is None else np.asarray(self.rows, int)
        if rows.shape[0] != A.shape[0]:
            raise ValueError(
                f"{self.name}: rows selects {rows.shape[0]} models but A has {A.shape[0]}"
            )
        cols = np.asarray(cols, int)
        rng = np.random.default_rng(self.seed)
        t = min(self.n_prompts, cols.size)
        if t < self.n_prompts:
            logger.warning(
                f"{self.name}: fitted block has {cols.size} columns, fewer than the paper's "
                f"t={self.n_prompts}; using all of them"
            )
        self.prompts = np.sort(rng.choice(cols, size=t, replace=False))
        p = int(bank.shape[2])
        L = int(self.dim)
        # Algorithm 1 line 1: A_jk ~ N(0, 1/L). Drawn chunk by chunk from one
        # generator in prompt order, so the matrix is fixed by (seed, prompts).
        proj_rng = np.random.default_rng([self.seed, L, t])
        table = np.zeros((n_all, L), dtype=np.float64)
        filled = 0
        for s in range(0, t, CHUNK):
            q = self.prompts[s : s + CHUNK]
            E = np.asarray(bank[:, q], dtype=np.float32)  # [M_all, c, p]
            if self.answer_mask is not None:
                ok = np.asarray(self.answer_mask[:, q], dtype=bool)
            else:
                ok = np.abs(E).sum(axis=2) > 0
            if not ok.all():
                tr_ok = ok[rows]
                tr = E[rows]
                denom = np.clip(tr_ok.sum(axis=0), 1, None)[:, None]
                mean = (tr * tr_ok[:, :, None]).sum(axis=0) / denom  # [c, p]
                miss = ~ok
                E[miss] = np.broadcast_to(mean[None], E.shape)[miss]
                filled += int(miss.sum())
            block = proj_rng.normal(0.0, 1.0 / np.sqrt(L), size=(L, q.size * p)).astype(np.float32)
            table += (E.reshape(n_all, -1) @ block.T).astype(np.float64)
        self.n_filled = filled
        logger.info(
            f"{self.name}: DNA of {n_all} models from t={t} prompts x p={p} -> L={L}; "
            f"{filled} of {n_all * t} cells filled with the training mean"
        )
        self.table = table
        self.X = table[rows]
        # the DNA table is all this arm needs; do not carry the shared bank
        self.answer_bank = None
        self.answer_mask = None
        return self

    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """The model's DNA; bits and interview unread (it is a function of its answers)."""
        if self.table is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        return np.asarray(self.table[row], dtype=np.float64)


__all__ = ["LLMDNA_DIM", "LLMDNA_PROMPTS", "LLMDNAEncoder"]
