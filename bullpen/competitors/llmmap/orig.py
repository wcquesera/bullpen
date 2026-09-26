"""``comp_llmmap_orig``: LLMmap retrained on our models, read on its own 8 published probes.

The network and training recipe are :class:`~bullpen.competitors.llmmap.encoder.LLMmapEncoder`'s
(trained on traces from the pool block; the backbone has no positional encoding, so it is
query-agnostic). The signature is LLMmap's inference setting: the CLS vector over the 8
``concat(query embedding, response embedding)`` tokens of the authors' published probes
(``queries_default.json``, upstream commit ``f661d55``), each response embedded with the
same BGE encoder as the answer bank. A model with no probe response takes the mean
training signature; one with some responses is embedded on those. Width 384.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from bullpen.competitors.llmmap.encoder import LLMMAP_DIM, LLMmapEncoder
from bullpen.models.base import capped_threads


@dataclass
class LLMmapOrigEncoder(LLMmapEncoder):
    """LLMmap, fingerprinting our models from LLMmap's own 8 published probes."""

    #: ``[M_all, n_probes, Da]`` embedded probe responses by global row; an
    #: all-zero cell is a probe the model was never asked or never answered
    probe_answers: np.ndarray | None = field(default=None, repr=False)
    #: ``[M_all, n_probes]`` — True where that response is real
    probe_mask: np.ndarray | None = field(default=None, repr=False)
    #: ``[n_probes, Dq]`` embedded probe PROMPTS, same encoder
    probe_questions: np.ndarray | None = field(default=None, repr=False)

    def _check_probe_bank(self) -> None:
        if self.probe_answers is None or self.probe_questions is None:
            raise ValueError(
                f"{self.name} needs the LLMmap probe bank (llmmap_probe_bank.npz beside the slice)"
            )
        if self.probe_answers.shape[1] != self.probe_questions.shape[0]:
            raise ValueError(
                f"{self.name}: {self.probe_answers.shape[1]} response slots but "
                f"{self.probe_questions.shape[0]} prompts — the bank is inconsistent"
            )

    @property
    def n_probes(self) -> int:
        self._check_probe_bank()
        return int(self.probe_questions.shape[0])

    def _probe_present(self, row: int) -> np.ndarray:
        """Indices of the published probes this model actually answered."""
        if self.probe_mask is not None:
            return np.flatnonzero(np.asarray(self.probe_mask[int(row)], dtype=bool))
        return np.flatnonzero(np.abs(self.probe_answers[int(row)]).sum(axis=1) > 0)

    def _probe_tokens(self, row: int, which: np.ndarray) -> np.ndarray:
        """``[len(which), Dq + Da]`` — the release's ``concat([q_emb, o_emb])`` per probe,
        normalised and rescaled as the parent does for a pool token."""
        which = np.asarray(which, int)
        q = np.asarray(self.probe_questions[which], dtype=np.float32)
        a = np.asarray(self.probe_answers[int(row)][which], dtype=np.float32)
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        return np.concatenate([q, a], axis=1) * self.token_norm

    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> LLMmapOrigEncoder:
        """Train as :class:`LLMmapEncoder`, then place every row on the published probes."""
        self._check_probe_bank()
        super().fit(A, cols)

        rows = np.arange(A.shape[0]) if self.rows is None else np.asarray(self.rows, int)
        X = np.zeros((len(rows), LLMMAP_DIM))
        answered = np.zeros(len(rows), dtype=bool)
        for i, r in enumerate(rows):
            which = self._probe_present(int(r))
            answered[i] = which.size > 0
            if which.size:
                X[i] = self._template(self._probe_tokens(int(r), which))
        if not answered.any():
            raise ValueError(
                f"{self.name}: none of the {len(rows)} training models has a response to "
                "LLMmap's probes; the collection has not reached this cut's models yet"
            )
        self._mean = X[answered].mean(axis=0)
        X[~answered] = self._mean
        n_full = int((np.asarray(self.probe_mask)[rows].sum(axis=1) == self.n_probes).sum())
        logger.info(
            f"{self.name}: signatures on LLMmap's {self.n_probes} published probes — "
            f"{int(answered.sum())}/{len(rows)} training models answered at least one, "
            f"{n_full} answered all of them"
        )
        self.X = X
        return self

    def signature(self, row: int) -> np.ndarray:
        """The model's CLS vector on the published probes; the mean when it has none."""
        self._fitted()
        which = self._probe_present(int(row))
        if which.size == 0:
            return self._mean.copy()
        return self._template(self._probe_tokens(int(row), which))


__all__ = ["LLMmapOrigEncoder"]
