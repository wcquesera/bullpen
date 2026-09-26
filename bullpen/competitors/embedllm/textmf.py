"""``comp_embedllm``: EmbedLLM's TextMF at the authors' recipe (arXiv 2410.02223, §4.2-4.3).

Reimplemented from ``richardzhuang0412/EmbedLLM`` at ``9b27630`` (``algorithm/mf.py``):
a free model table ``P`` (``nn.Embedding`` default init), the frozen all-mpnet-base-v2
question vector plus N(0, 0.05^2) jitter through ``Linear(768, d)``, the Hadamard product
into ``Linear(d, 2)``, cross-entropy on correctness, Adam lr 1e-4 with coupled weight
decay 1e-5, 50 epochs of unshuffled batches of 2048, d = 232 (the release default).
Upstream has no path for an unseen model; the fold-in is :class:`TextMFEncoder`'s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from bullpen.models.base import capped_threads, torch_device
from bullpen.models.text_mf import LABEL_THRESHOLD, LOG_EVERY, TextMFEncoder

if TYPE_CHECKING:  # pragma: no cover
    import torch

#: ``mf.py`` argparse defaults (232 is the released default width).
FAITHFUL_DIM: int = 232
FAITHFUL_EPOCHS: int = 50
FAITHFUL_LR: float = 1e-4
FAITHFUL_BATCH_SIZE: int = 2048
FAITHFUL_ALPHA: float = 0.05
#: ``train(..., weight_decay=1e-5)`` default, passed to ``torch.optim.Adam`` (l.242, 243)
FAITHFUL_WEIGHT_DECAY: float = 1e-5

#: The sentence encoder upstream embeds question text with
#: (``data_preprocessing/get_question_embedding_tensor.py:7``; paper §4.2).
FAITHFUL_QUESTION_ENCODER: str = "sentence-transformers/all-mpnet-base-v2"
FAITHFUL_TEXT_DIM: int = 768

#: Upstream commit the recipe was read from.
UPSTREAM_COMMIT: str = "9b276305af1342188c2fc066e41d5dc8bdfea895"


def build_textmf(
    q_table: np.ndarray, n_models: int, dim: int, seed: int
) -> tuple[torch.nn.Embedding, torch.Tensor, torch.nn.Linear, torch.nn.Linear]:
    """``(P, Q_frozen, text_proj, classifier)`` with upstream's default inits, seeded."""
    import torch

    device = torch_device()
    torch.manual_seed(seed)
    P = torch.nn.Embedding(n_models, dim).to(device)
    q = torch.tensor(np.asarray(q_table, dtype=np.float32), device=device)
    text_proj = torch.nn.Linear(q.shape[1], dim).to(device)
    classifier = torch.nn.Linear(dim, 2).to(device)
    return P, q, text_proj, classifier


def train_textmf(
    models: np.ndarray,
    prompts: np.ndarray,
    labels: np.ndarray,
    q_table: np.ndarray,
    n_models: int,
    *,
    dim: int = FAITHFUL_DIM,
    seed: int = 0,
    epochs: int = FAITHFUL_EPOCHS,
    lr: float = FAITHFUL_LR,
    weight_decay: float = FAITHFUL_WEIGHT_DECAY,
    alpha: float = FAITHFUL_ALPHA,
    batch_size: int = FAITHFUL_BATCH_SIZE,
    on_epoch: Callable[[int, float, tuple], None] | None = None,
) -> tuple[torch.nn.Embedding, torch.nn.Linear, torch.nn.Linear]:
    """Upstream's ``train()`` over ``(model, prompt, label)`` rows, in the order given.

    ``prompts`` index rows of ``q_table``; ``on_epoch(epoch, mean_loss, modules)`` is
    called after every epoch.
    """
    import torch

    if epochs < 1 or batch_size < 1:
        raise ValueError(f"epochs and batch_size must be >= 1, got {epochs}, {batch_size}")
    device = torch_device()
    P, q_frozen, text_proj, classifier = build_textmf(q_table, n_models, dim, seed)
    params = list(P.parameters()) + list(text_proj.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    m_t = torch.as_tensor(np.asarray(models, dtype=np.int64), device=device)
    p_t = torch.as_tensor(np.asarray(prompts, dtype=np.int64), device=device)
    y_t = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    n = int(m_t.shape[0])
    n_steps = (n + batch_size - 1) // batch_size
    for epoch in range(epochs):
        total = 0.0
        for start in range(0, n, batch_size):
            mi = m_t[start : start + batch_size]
            q = q_frozen[p_t[start : start + batch_size]]
            if alpha:
                q = q + torch.randn_like(q) * alpha
            loss = loss_fn(classifier(P(mi) * text_proj(q)), y_t[start : start + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if on_epoch is not None:
            on_epoch(epoch, total / n_steps, (P, text_proj, classifier))
    return P, text_proj, classifier


@dataclass
class TextMFFaithfulEncoder(TextMFEncoder):
    """TextMF at upstream's optimiser, init, batch order and width.

    ``q_text_emb`` must be the all-mpnet-base-v2 table; ``fold_in`` and ``predict``
    are inherited.
    """

    dim: int = FAITHFUL_DIM
    epochs: int = FAITHFUL_EPOCHS
    lr: float = FAITHFUL_LR
    weight_decay: float = FAITHFUL_WEIGHT_DECAY
    alpha: float = FAITHFUL_ALPHA
    batch_size: int = FAITHFUL_BATCH_SIZE

    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> TextMFFaithfulEncoder:
        import torch

        A = self._prepare(A, cols)
        q_table = self._q_block(cols).cpu().numpy()
        if q_table.shape[1] != FAITHFUL_TEXT_DIM:
            raise ValueError(
                f"{self.name}: question bridge is {q_table.shape[1]}-d; the faithful arm "
                f"reads {FAITHFUL_QUESTION_ENCODER} ({FAITHFUL_TEXT_DIM}-d)"
            )
        n_models, n_questions = A.shape
        # upstream reads its training rows unshuffled, ordered by model then prompt
        models, prompts = np.divmod(np.arange(n_models * n_questions), n_questions)
        labels = (A.ravel() >= LABEL_THRESHOLD).astype(np.int64)

        def log(epoch: int, loss: float, _: tuple) -> None:
            if epoch % LOG_EVERY == 0 or epoch == self.epochs - 1:
                logger.debug(f"{self.name}: epoch {epoch + 1}/{self.epochs} loss {loss:.4f}")

        logger.info(
            f"{self.name}: {n_models} models x {n_questions} questions, d={self.dim}, "
            f"{self.epochs} epochs at batch {self.batch_size}, Adam wd={self.weight_decay}, "
            "unshuffled (upstream mf.py recipe)"
        )
        P, text_proj, classifier = train_textmf(
            models,
            prompts,
            labels,
            q_table,
            n_models,
            dim=self.dim,
            seed=self.seed,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            batch_size=self.batch_size,
            on_epoch=log,
        )
        with torch.no_grad():
            self.X = P.weight.detach().cpu().numpy().astype(np.float64)
        self._net = (text_proj, classifier)
        return self


__all__ = [
    "FAITHFUL_ALPHA",
    "FAITHFUL_BATCH_SIZE",
    "FAITHFUL_DIM",
    "FAITHFUL_EPOCHS",
    "FAITHFUL_LR",
    "FAITHFUL_QUESTION_ENCODER",
    "FAITHFUL_TEXT_DIM",
    "FAITHFUL_WEIGHT_DECAY",
    "UPSTREAM_COMMIT",
    "TextMFFaithfulEncoder",
    "build_textmf",
    "train_textmf",
]
