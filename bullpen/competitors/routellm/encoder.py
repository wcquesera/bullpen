"""``comp_routellm_mf``: RouteLLM's matrix-factorisation router (Ong et al., ICLR 2025, §4.2).

A Bradley-Terry model over per-(model, query) scores (model in :mod:`.mf`)::

    delta(M, q)     = w2^T ( normalize(P[M]) * W1 e(q) )      # e(q) frozen text embedding
    P(M beats M'|q) = sigmoid(delta(M, q) - delta(M', q))

Preferences are built as the paper's golden-label augmentation (§4.1.1): on a question,
the model that got it right beats the one that got it wrong; ties are dropped. Pairs come
from the training rows only. The embedding is ``normalize(P[M])``; :meth:`predict` is
undefined (BT scores have no absolute calibration). Placing a held-out model is our
extension: one new ``P`` row fitted under the same loss against the frozen training rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from bullpen.models.base import Encoder, capped_threads, on_device, torch_device

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from bullpen.competitors.routellm.mf import MFModel

#: Paper §4.2: about 10 epochs, batch 64, Adam lr 3e-4, weight decay 1e-5.
MF_EPOCHS: int = 10
MF_BATCH_SIZE: int = 64
MF_LR: float = 3e-4
MF_WEIGHT_DECAY: float = 1e-5
#: Gaussian noise on the frozen prompt embedding during training (the release example's 0.1).
MF_ALPHA: float = 0.1
#: Preference pairs per fit, drawn once (paper §5: 65k Arena comparisons).
MF_N_PAIRS: int = 65_000
#: Upstream's model width (MatrixFactorizationRouter hidden_size=128).
MF_PUBLISHED_DIM: int = 128

#: Fold-in: full-batch Adam on one new P row, everything else frozen.
FOLDIN_STEPS: int = 200
FOLDIN_LR: float = 1e-2

#: A soft cell at or above this is "correct" when deriving a preference.
LABEL_THRESHOLD: float = 0.5

#: Candidate triples drawn per round while collecting non-tie pairs.
_DRAW_CHUNK: int = 1 << 16
#: Rounds before a pool with almost no disagreement is refused.
_MAX_DRAW_ROUNDS: int = 200


def correctness_pairs(
    bits: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(winner, loser, question)`` index arrays of ``n_pairs`` non-tie preferences.

    Triples ``(a, b, q)`` with ``a != b`` are drawn uniformly and ties discarded. When
    the pool holds at most ``n_pairs`` distinct non-tie pairs, each is used exactly
    once, in a seeded order.
    """
    bits = np.asarray(bits, dtype=bool)
    n_models, n_questions = bits.shape
    if n_models < 2:
        raise ValueError(f"preference pairs need at least 2 models, got {n_models}")
    col_sum = bits.sum(axis=0)
    if not ((col_sum > 0) & (col_sum < n_models)).any():
        raise ValueError(
            "no question separates any two models (every column is all-right or all-wrong), "
            "so there is no preference to learn"
        )
    if int((col_sum * (n_models - col_sum)).sum()) <= n_pairs:
        q, w, lo = [], [], []
        for j in np.flatnonzero((col_sum > 0) & (col_sum < n_models)):
            right, wrong = np.flatnonzero(bits[:, j]), np.flatnonzero(~bits[:, j])
            ww, ll = np.meshgrid(right, wrong, indexing="ij")
            w.append(ww.ravel())
            lo.append(ll.ravel())
            q.append(np.full(ww.size, j))
        order = rng.permutation(sum(x.size for x in q))
        return np.concatenate(w)[order], np.concatenate(lo)[order], np.concatenate(q)[order]
    win, lose, qs = [], [], []
    have = 0
    for _ in range(_MAX_DRAW_ROUNDS):
        a = rng.integers(0, n_models, _DRAW_CHUNK)
        b = (a + rng.integers(1, n_models, _DRAW_CHUNK)) % n_models
        q = rng.integers(0, n_questions, _DRAW_CHUNK)
        ya, yb = bits[a, q], bits[b, q]
        keep = ya != yb
        a, b, q, ya = a[keep], b[keep], q[keep], ya[keep]
        win.append(np.where(ya, a, b))
        lose.append(np.where(ya, b, a))
        qs.append(q)
        have += int(keep.sum())
        if have >= n_pairs:
            break
    else:
        raise ValueError(
            f"only {have} non-tie pairs after {_MAX_DRAW_ROUNDS * _DRAW_CHUNK} draws; "
            f"asked for {n_pairs}"
        )
    return (
        np.concatenate(win)[:n_pairs],
        np.concatenate(lose)[:n_pairs],
        np.concatenate(qs)[:n_pairs],
    )


@dataclass
class RouteLLMMFEncoder(Encoder):
    """RouteLLM's MF router fitted on correctness-derived preferences."""

    #: [Q_total, d_text] frozen question embeddings indexed by GLOBAL question id
    q_text_emb: np.ndarray | None = field(default=None, repr=False)
    epochs: int = MF_EPOCHS
    batch_size: int = MF_BATCH_SIZE
    lr: float = MF_LR
    weight_decay: float = MF_WEIGHT_DECAY
    alpha: float = MF_ALPHA
    n_pairs: int = MF_N_PAIRS
    foldin_steps: int = FOLDIN_STEPS
    foldin_lr: float = FOLDIN_LR
    _net: MFModel | None = field(default=None, repr=False)
    #: training bits, kept because a fold-in's preferences are AGAINST training models
    _train_bits: np.ndarray | None = field(default=None, repr=False)

    has_decoder: bool = False
    interview_requirement: str = "training_grades_pool"

    def _q_block(self, cols: np.ndarray) -> torch.Tensor:
        import torch

        if self.q_text_emb is None:
            raise ValueError(
                f"{self.name} needs a frozen question text table and was given none; "
                "pass q_text_emb=<[Q_total, d_text] table indexed by global question id>"
            )
        table = np.asarray(self.q_text_emb)
        if table.ndim != 2:
            raise ValueError(f"{self.name}: q_text_emb must be 2-d, got {table.shape}")
        ids = np.asarray(cols).ravel().astype(int)
        if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
            raise ValueError(
                f"{self.name}: q_text_emb covers ids 0..{table.shape[0] - 1}, asked for "
                f"{int(ids.min())}..{int(ids.max())}"
            )
        return torch.tensor(table[ids].astype(np.float32), device=torch_device())

    def _fitted(self) -> MFModel:
        if self._net is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        return on_device(self._net)

    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> RouteLLMMFEncoder:
        import torch

        from bullpen.competitors.routellm.mf import MFModel

        if self.epochs < 1 or self.batch_size < 1 or self.n_pairs < 1:
            raise ValueError(f"{self.name}: epochs, batch_size and n_pairs must be positive")
        A = self._prepare(A, cols)
        bits = A >= LABEL_THRESHOLD
        n_models, n_questions = bits.shape
        device = torch_device()

        rng = np.random.default_rng(self.seed)
        win, lose, qidx = correctness_pairs(bits, self.n_pairs, rng)
        torch.manual_seed(self.seed)
        q_frozen = self._q_block(cols)
        net = MFModel(self.dim, n_models, int(q_frozen.shape[1])).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        gen = torch.Generator(device=device).manual_seed(self.seed)

        t_win = torch.tensor(win, device=device)
        t_lose = torch.tensor(lose, device=device)
        t_q = torch.tensor(qidx, device=device)
        n = t_win.shape[0]
        logger.info(
            f"{self.name}: {n_models} models x {n_questions} questions, {n} preference pairs "
            f"(ties dropped), d_text={int(q_frozen.shape[1])}, dim={self.dim}, "
            f"{self.epochs} epochs x {(n + self.batch_size - 1) // self.batch_size} steps"
        )
        net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=device, generator=gen)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                out = net(t_win[b], t_lose[b], q_frozen[t_q[b]], alpha=self.alpha, generator=gen)
                loss = loss_fn(out, torch.ones_like(out))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * b.shape[0]
            logger.debug(f"{self.name}: epoch {epoch + 1}/{self.epochs} loss {total / n:.4f}")
        net.eval()
        with torch.no_grad():
            self.X = (
                torch.nn.functional.normalize(net.P.weight, p=2, dim=1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        self._net = net
        self._train_bits = bits
        return self

    @capped_threads
    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """One new ``P`` row under the BT loss against the frozen training rows.

        Full-batch Adam from the mean training ``P`` over every non-tie (new model,
        training model, question) pair.
        """
        import torch

        net = self._fitted()
        j = self.local(cols)
        y = np.asarray(bits, dtype=np.float64).ravel() >= LABEL_THRESHOLD
        if y.shape[0] != j.shape[0]:
            raise ValueError(f"{self.name}: {j.shape[0]} columns but {y.shape[0]} bits")
        device = torch_device()
        p0 = net.P.weight.detach().mean(dim=0)
        train = self._train_bits[:, j]  # [M, k]
        disagree = train != y[None, :]
        if not disagree.any():
            return torch.nn.functional.normalize(p0, dim=0).cpu().numpy().astype(np.float64)

        m_idx, j_idx = np.nonzero(disagree)
        # +1 where the new model won (it was right, the training model wrong)
        sign = torch.tensor(np.where(y[j_idx], 1.0, -1.0), dtype=torch.float32, device=device)
        jj = torch.tensor(j_idx, device=device)
        with torch.no_grad():
            q_proj = net.project(self._q_block(cols))
            d_train = net.score(torch.nn.functional.normalize(net.P.weight, dim=1), q_proj)
            d_pair = d_train[torch.tensor(m_idx, device=device), jj]
        w = net.classifier.weight.detach().reshape(-1)

        theta = torch.nn.Parameter(p0.clone())
        opt = torch.optim.Adam([theta], lr=self.foldin_lr)
        for _ in range(self.foldin_steps):
            d_new = (torch.nn.functional.normalize(theta, dim=0) * w) @ q_proj.T  # [k]
            loss = torch.nn.functional.softplus(-sign * (d_new[jj] - d_pair)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return torch.nn.functional.normalize(theta.detach(), dim=0).cpu().numpy().astype(np.float64)


__all__ = [
    "FOLDIN_LR",
    "FOLDIN_STEPS",
    "MF_ALPHA",
    "MF_BATCH_SIZE",
    "MF_EPOCHS",
    "MF_LR",
    "MF_N_PAIRS",
    "MF_PUBLISHED_DIM",
    "MF_WEIGHT_DECAY",
    "RouteLLMMFEncoder",
    "correctness_pairs",
]
