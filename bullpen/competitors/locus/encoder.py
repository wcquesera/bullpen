"""``comp_locus``: the LOCUS set encoder (arXiv 2601.21082), trained on the training rows.

It reads a model's correctness bits, each paired with a frozen question embedding. A
model's embedding is one forward pass of the set encoder over its (question embedding,
bit) pairs, so a held-out model is placed without optimisation (the paper's test-time
onboarding, §3.2). The fit follows the authors' release: 1,024 random anchor questions
from the fitted columns (fewer when the block is small), AdamW (encoder 2e-4, decoder
8e-4, weight decay 5e-2), grad clip 1.0, query noise annealed 0.10 -> 0.05 over 500
epochs, up to 2,500 epochs, checkpoint by routing hit@1 with patience 500. Deviations:
the selection questions are the fitted block's non-anchor columns (upstream uses
EmbedLLM's validation split), and selection starts only after the noise anneal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np
from loguru import logger

from bullpen.competitors.locus.net import LocusConfig, LocusNet, routing_hit1
from bullpen.models.base import Encoder, capped_threads, on_device, torch_device

#: The published width (paper §3 "d = 128"; ``training_script.py:364``).
LOCUS_DIM: int = 128

#: Validation questions for checkpoint selection, drawn from the non-anchor fitted columns.
LOCUS_VAL_QUESTIONS: int = 2048

#: Epochs between progress lines.
LOG_EVERY: int = 50


@dataclass
class LocusEncoder(Encoder):
    """LOCUS: attention set encoder over (question embedding, correctness) pairs."""

    #: [Q_total, d_text] frozen question embeddings indexed by global question id
    q_text_emb: np.ndarray | None = field(default=None, repr=False)
    cfg: LocusConfig = field(default_factory=LocusConfig)
    val_questions: int = LOCUS_VAL_QUESTIONS
    #: global ids of the anchor questions every training embedding was read from
    anchor_cols: np.ndarray | None = field(default=None, repr=False)
    best_epoch: int = -1
    best_val_hit1: float = float("nan")
    _net: LocusNet | None = field(default=None, repr=False)

    interview_requirement: str = "training_grades_pool"

    #: the question table is re-attached from the slice on load
    SHARED_BANKS: ClassVar[dict[str, str]] = {"q_text_emb": "Qe"}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim != self.cfg.z_dim:
            self.cfg = replace(self.cfg, z_dim=self.dim)

    # -- helpers ------------------------------------------------------------ #
    def _q(self, cols: np.ndarray):  # -> torch.Tensor
        import torch

        if self.q_text_emb is None:
            raise ValueError(
                f"{self.name} needs a frozen question embedding table and was given none"
            )
        table = np.asarray(self.q_text_emb)
        ids = np.asarray(cols).ravel().astype(int)
        if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
            raise ValueError(
                f"{self.name}: q_text_emb covers ids 0..{table.shape[0] - 1}, asked for "
                f"{int(ids.min())}..{int(ids.max())}"
            )
        return torch.tensor(table[ids], dtype=torch.float32, device=torch_device())

    def _fitted(self) -> LocusNet:
        if self._net is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        return on_device(self._net)  # type: ignore[return-value]

    # -- fitting ------------------------------------------------------------ #
    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> LocusEncoder:
        import torch

        A = self._prepare(A, cols)
        cols = np.asarray(cols)
        n_models, n_cols = A.shape
        c = self.cfg
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        device = torch_device()
        gen = torch.Generator(device=device).manual_seed(self.seed)

        # the published 1,024 anchors, but keep at least a fifth of the block for selection
        n_anchor = min(c.anchors, n_cols - max(1, n_cols // 5))
        if n_anchor < c.anchors:
            logger.warning(
                f"{self.name}: {n_cols} fitted columns — {n_anchor} anchors instead of the "
                f"published {c.anchors}, keeping {n_cols - n_anchor} for checkpoint selection"
            )
        anchor_pos = np.sort(rng.choice(n_cols, size=n_anchor, replace=False))
        rest = np.setdiff1d(np.arange(n_cols), anchor_pos)
        if rest.size == 0:
            raise ValueError(
                f"{self.name}: all {n_cols} fitted columns are anchors, leaving no held-out "
                "questions to select a checkpoint on; LOCUS selects by routing on unseen "
                "questions, so give it more columns than anchors"
            )
        val_pos = np.sort(rng.choice(rest, size=min(self.val_questions, rest.size), replace=False))
        self.anchor_cols = cols[anchor_pos]

        Qa = self._q(cols[anchor_pos])
        Ya = torch.tensor(A[:, anchor_pos], dtype=torch.float32, device=device)
        Qv = self._q(cols[val_pos])
        Yv = torch.tensor(A[:, val_pos].T, dtype=torch.float32, device=device)  # [N, M]

        net = LocusNet(int(Qa.shape[1]), c).to(device)
        opt = net.optimizer()
        logger.info(
            f"{self.name}: {n_models} models, {n_anchor} anchors of {n_cols} fitted columns, "
            f"{val_pos.size} selection questions, d={c.z_dim}, up to {c.max_epochs} epochs "
            f"(patience {c.patience})"
        )
        # Deviation: select only after the query-noise anneal. With few training
        # models the validation signal is nearly flat, and selecting from epoch 0 can
        # keep the untrained network as "best".
        select_from = c.noise_end_epoch
        best, best_epoch, best_state = -float("inf"), -1, None
        for epoch in range(c.max_epochs):
            loss = net.train_step(Qa, Ya, opt, epoch, gen)
            hit1 = routing_hit1(net.score(net.embed(Qa, Ya), Qv), Yv)
            if hit1 > best and epoch >= select_from:
                best, best_epoch = hit1, epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if epoch % LOG_EVERY == 0:
                logger.debug(
                    f"{self.name}: epoch {epoch} bce {loss:.4f} val hit@1 {hit1:.4f} "
                    f"(best {best:.4f} @ {best_epoch})"
                )
            if epoch >= select_from and epoch - best_epoch >= c.patience:
                break
        assert best_state is not None
        net.load_state_dict(best_state)
        self.best_epoch, self.best_val_hit1 = best_epoch, best
        logger.info(f"{self.name}: best val hit@1 {best:.4f} at epoch {best_epoch}")
        self.X = net.embed(Qa, Ya).cpu().numpy().astype(np.float64)
        # LOCUS reads only its anchors of the fitted block
        self.input_width = int(n_anchor)
        self._net = net.cpu()
        return self

    # -- placing a model ---------------------------------------------------- #
    @capped_threads
    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """One forward pass of the encoder over the observed (question, bit) pairs.

        An empty interview returns the mean training embedding.
        """
        import torch

        net = self._fitted()
        self.local(cols)
        y = np.asarray(bits, dtype=np.float64).ravel()
        if y.shape[0] != np.asarray(cols).size:
            raise ValueError(f"{self.name}: {np.asarray(cols).size} columns but {y.shape[0]} bits")
        if y.shape[0] == 0:
            assert self.X is not None
            return self.X.mean(axis=0)
        Yt = torch.tensor(y[None, :], dtype=torch.float32, device=torch_device())
        return net.embed(self._q(cols), Yt)[0].cpu().numpy().astype(np.float64)

    def refit(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Whole-row placement: encoded on the anchors when the row covers them all (as
        :attr:`X` was), otherwise on every given cell."""
        cols = np.asarray(cols)
        bits = np.asarray(bits).ravel()
        if self.anchor_cols is not None:
            pos = {int(c): i for i, c in enumerate(cols)}
            if all(int(a) in pos for a in self.anchor_cols):
                idx = np.array([pos[int(a)] for a in self.anchor_cols])
                return self.fold_in(bits[idx], cols[idx], row)
        return self.fold_in(bits, cols, row)

    def predict(self, theta: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """P(correct) per question from the bilinear-MLP decoder."""
        import torch

        net = self._fitted()
        self.local(cols)
        z = torch.tensor(np.asarray(theta, dtype=np.float32)[None, :], device=torch_device())
        logits = net.score(z, self._q(cols))[:, 0]
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


__all__ = ["LOCUS_DIM", "LOCUS_VAL_QUESTIONS", "LocusEncoder"]
