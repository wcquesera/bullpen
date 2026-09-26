"""Shared training loop of the question-text IRT competitors (IRT-Router, JE-IRT, IrtNet).

Each is a free per-model ability table, a network mapping a frozen question embedding to
item parameters, and a logistic response trained with BCE over every (model, question)
cell. This module holds the shared parts: the loop over shuffled cells, checkpoint
selection on questions held back from the fitted block, the fold-in of an unseen model
(penalised BCE with the item side frozen, a convex logistic regression) and prediction.
After the fit the item parameters of the fitted columns are cached and the network dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from loguru import logger

from bullpen.models.base import Encoder, capped_threads, torch_device

if TYPE_CHECKING:  # pragma: no cover
    import torch

#: Cell label threshold for the arms whose published loss is binary.
LABEL_THRESHOLD: float = 0.5

#: Fold-in: Adam steps, step size and L2 pull toward the mean training vector (ours).
FOLDIN_STEPS: int = 300
FOLDIN_LR: float = 5e-2
FOLDIN_L2: float = 1e-3

LOG_EVERY: int = 5


@dataclass
class TextIRTEncoder(Encoder):
    """Free model table x question-text item network, logistic response, BCE."""

    #: [Q_total, d_text] frozen question embeddings indexed by global question id
    q_text_emb: np.ndarray | None = field(default=None, repr=False)
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    #: fraction of fitted columns held back for checkpoint selection (0 = none)
    val_frac: float = 0.0
    #: epochs without improvement before stopping (0 = run every epoch)
    patience: int = 0
    #: True: BCE against the soft cell value; False: against ``A >= 0.5``
    soft_labels: bool = False
    #: optimiser steps per epoch; None = one pass over the fitted cells
    steps_per_epoch: int | None = None
    #: checkpoint rule on the held-back questions: "bce" (lowest) or "acc" (highest)
    select_on: str = "bce"
    foldin_steps: int = FOLDIN_STEPS
    foldin_lr: float = FOLDIN_LR
    foldin_l2: float = FOLDIN_L2
    #: [Q_fit, d] response-function weights and [Q_fit] offsets, cached after fit
    item_w: np.ndarray | None = field(default=None, repr=False)
    item_b: np.ndarray | None = field(default=None, repr=False)
    best_epoch: int = -1
    best_val: float = float("nan")

    interview_requirement: str = "training_grades_pool"

    #: the item network's name, for logs
    METHOD: ClassVar[str] = ""

    # -- hooks the three arms fill in -------------------------------------- #
    def build_item_net(self, d_text: int) -> torch.nn.Module:  # pragma: no cover
        raise NotImplementedError

    def init_model_table(self, table: torch.nn.Embedding) -> None:
        """Leave ``nn.Embedding``'s N(0, 1) unless the method says otherwise."""

    def build_model_side(self, n_models: int, device: torch.device) -> torch.nn.Module:
        """The module mapping training-row indices to raw model vectors (a free table)."""
        import torch

        # moved before the init (same draws on either device)
        table = torch.nn.Embedding(n_models, self.dim).to(device)
        self.init_model_table(table)
        return table

    def all_models(self, side: torch.nn.Module) -> torch.Tensor:
        """``[n_models, dim]`` raw vectors of every training row."""
        return side.weight

    def item_params(self, net: torch.nn.Module, q: torch.Tensor) -> tuple:  # pragma: no cover
        """``(w [B, d], b [B])`` such that ``logit = <w, theta> - b``."""
        raise NotImplementedError

    def after_step(self, net: torch.nn.Module) -> None:
        """Called after every optimiser step."""

    def model_vector(self, theta: torch.Tensor) -> torch.Tensor:
        """Map the raw table row to the vector the logit is linear in."""
        return theta

    # -- helpers ----------------------------------------------------------- #
    def _q(self, cols: np.ndarray) -> torch.Tensor:
        import torch

        if self.q_text_emb is None:
            raise ValueError(
                f"{self.name} needs a frozen question embedding table and was given none; "
                "its item parameters are a function of the question text"
            )
        table = np.asarray(self.q_text_emb)
        ids = np.asarray(cols).ravel().astype(int)
        if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
            raise ValueError(
                f"{self.name}: q_text_emb covers ids 0..{table.shape[0] - 1}, asked for "
                f"{int(ids.min())}..{int(ids.max())}"
            )
        return torch.tensor(table[ids], dtype=torch.float32, device=torch_device())

    def _item_table(self, net: torch.nn.Module, Q: torch.Tensor) -> tuple:
        import torch

        net.eval()
        ws, bs = [], []
        with torch.no_grad():
            for s in range(0, Q.shape[0], 4096):
                w, b = self.item_params(net, Q[s : s + 4096])
                ws.append(w)
                bs.append(b)
        net.train()
        return torch.cat(ws), torch.cat(bs)

    # -- fitting ----------------------------------------------------------- #
    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> TextIRTEncoder:
        import torch

        A = self._prepare(A, cols)
        cols = np.asarray(cols)
        n_models, n_cols = A.shape
        device = torch_device()
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        gen = torch.Generator(device=device).manual_seed(self.seed)

        n_val = round(self.val_frac * n_cols) if self.val_frac > 0 else 0
        perm = rng.permutation(n_cols)
        val_pos, train_pos = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        Q_all = self._q(cols)
        target = A if self.soft_labels else (A >= LABEL_THRESHOLD).astype(np.float64)
        Y = torch.tensor(target, dtype=torch.float32, device=device)
        tr = torch.tensor(train_pos, dtype=torch.long, device=device)
        va = torch.tensor(val_pos, dtype=torch.long, device=device)

        net = self.build_item_net(int(Q_all.shape[1])).to(device)
        table = self.build_model_side(n_models, device)
        opt = torch.optim.Adam(
            list(net.parameters()) + list(table.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = self.scheduler(opt)
        bce = torch.nn.BCEWithLogitsLoss()

        n_pairs = n_models * train_pos.size
        n_steps = self.steps_per_epoch or -(-n_pairs // self.batch_size)
        logger.info(
            f"{self.name} ({self.METHOD}): {n_models} models x {train_pos.size} questions "
            f"({n_val} held back for selection), d={self.dim}, up to {self.epochs} epochs "
            f"of {n_steps} steps at batch {self.batch_size}, lr {self.lr}, wd {self.weight_decay}"
        )
        best, best_state, since = float("inf"), None, 0
        # one "epoch" is steps_per_epoch optimiser steps over a stream of shuffled
        # passes (upstream's step count per epoch), or one pass when unset
        order, cursor = torch.randperm(n_pairs, device=device, generator=gen), 0
        for epoch in range(self.epochs):
            total, steps = 0.0, 0
            for _ in range(n_steps):
                if cursor >= n_pairs:
                    order, cursor = torch.randperm(n_pairs, device=device, generator=gen), 0
                b = order[cursor : cursor + self.batch_size]
                cursor += self.batch_size
                mi = b // train_pos.size
                qi = tr[b % train_pos.size]
                w, off = self.item_params(net, Q_all[qi])
                logit = (w * self.model_vector(table(mi))).sum(-1) - off
                loss = bce(logit, Y[mi, qi])
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.after_step(net)
                total += float(loss.detach())
                steps += 1
            msg = f"{self.name}: epoch {epoch + 1}/{self.epochs} train bce {total / steps:.4f}"
            if n_val:
                vloss, vacc = self._val_scores(net, table, Q_all[va], Y[:, va])
                msg += f" val bce {vloss:.4f} acc {vacc:.4f}"
                if sched is not None:
                    sched.step(vloss)
                val = vloss if self.select_on == "bce" else -vacc
                if val < best:
                    best, since, self.best_epoch = val, 0, epoch
                    best_state = (
                        {k: v.detach().clone() for k, v in net.state_dict().items()},
                        {k: v.detach().clone() for k, v in table.state_dict().items()},
                    )
                else:
                    since += 1
            if epoch % LOG_EVERY == 0 or epoch == self.epochs - 1:
                logger.info(msg)
            if n_val and self.patience and since >= self.patience:
                logger.info(f"{self.name}: no val improvement for {since} epochs, stopping")
                break
        if best_state is not None:
            net.load_state_dict(best_state[0])
            table.load_state_dict(best_state[1])
            self.best_val = best if self.select_on == "bce" else -best
            logger.info(
                f"{self.name}: best val {self.select_on} {self.best_val:.4f} "
                f"at epoch {self.best_epoch + 1}"
            )

        with torch.no_grad():
            self.X = self.model_vector(self.all_models(table)).cpu().numpy().astype(np.float64)
        w, b = self._item_table(net, Q_all)
        self.after_fit(table)
        self.item_w = w.cpu().numpy().astype(np.float32)
        self.item_b = b.cpu().numpy().astype(np.float32)
        # the network and the question table are not needed past this point
        self.q_text_emb = None
        return self

    def after_fit(self, side: torch.nn.Module) -> None:
        """Hook after the fit; the fold-in needs nothing from the model side."""

    def scheduler(self, opt: torch.optim.Optimizer):
        return None

    def _val_scores(self, net, table, Qv, Yv) -> tuple[float, float]:
        """BCE and accuracy at 0.5 on the held-back questions, every model."""
        import torch

        with torch.no_grad():
            w, b = self._item_table(net, Qv)
            logit = self.model_vector(self.all_models(table)) @ w.T - b[None, :]
            bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logit, Yv))
            acc = float(((logit > 0).float() == (Yv >= LABEL_THRESHOLD).float()).float().mean())
        return bce, acc

    # -- placing and reading models --------------------------------------- #
    def _fitted_items(self, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.item_w is None or self.item_b is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        j = self.local(cols)
        return self.item_w[j].astype(np.float64), self.item_b[j].astype(np.float64)

    @capped_threads
    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Penalised BCE for one model vector with the items frozen, pulled toward the mean."""
        import torch

        w, b = self._fitted_items(cols)
        y = np.asarray(bits, dtype=np.float64).ravel()
        if y.shape[0] != w.shape[0]:
            raise ValueError(f"{self.name}: {w.shape[0]} columns but {y.shape[0]} bits")
        assert self.X is not None
        mu = self.X.mean(axis=0)
        if y.shape[0] == 0:
            return mu.copy()
        if not self.soft_labels:
            y = (y >= LABEL_THRESHOLD).astype(np.float64)
        W = torch.tensor(w)
        B = torch.tensor(b)
        Yt = torch.tensor(y)
        m0 = torch.tensor(mu)
        v = torch.nn.Parameter(m0.clone())
        opt = torch.optim.Adam([v], lr=self.foldin_lr)
        for _ in range(self.foldin_steps):
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(W @ v - B, Yt)
                + self.foldin_l2 * (v - m0).pow(2).sum()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        return v.detach().numpy().astype(np.float64)

    def predict(self, theta: np.ndarray, cols: np.ndarray) -> np.ndarray:
        w, b = self._fitted_items(cols)
        z = np.clip(w @ np.asarray(theta, dtype=np.float64) - b, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))


__all__ = ["FOLDIN_L2", "FOLDIN_LR", "FOLDIN_STEPS", "LABEL_THRESHOLD", "TextIRTEncoder"]
