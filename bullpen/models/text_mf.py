"""EmbedLLM's TextMF matrix factorisation (Zhuang et al.), the base of ``comp_embedllm``::

    p      = P[model]                       # free per-model parameter, (dim,)
    q      = text_proj(Q_frozen[question])  # frozen text embedding + learned proj
    logits = classifier(p * q)              # Hadamard product, then linear -> 2
    loss   = CrossEntropy(logits, correct)

``q_text_emb`` is a ``[Q_total, d_text]`` table of frozen question embeddings indexed by
global question id. An unseen model has no row in ``P``; :meth:`TextMFEncoder.fold_in`
optimises one against its probe cells. The upstream recipe (optimiser, init, batch
order, width) is set in :mod:`bullpen.competitors.embedllm.textmf`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bullpen.models.base import Encoder, capped_threads, on_device, torch_device

if TYPE_CHECKING:  # pragma: no cover - torch is imported lazily at call time
    import torch

#: Training schedule (lr, epochs, batch and weight decay as upstream's ``algorithm/mf.py``).
TEXTMF_EPOCHS: int = 50
TEXTMF_LR: float = 1e-4
TEXTMF_WEIGHT_DECAY: float = 1e-5
TEXTMF_BATCH_SIZE: int = 2048

#: Standard deviation of the Gaussian jitter added to the frozen question vectors each
#: step (upstream's text-side regulariser).
TEXTMF_ALPHA: float = 0.05

#: Adam steps, step size and L2 for placing an unseen model.
FOLDIN_STEPS: int = 200
FOLDIN_LR: float = 1e-2
FOLDIN_L2: float = 1e-3

#: Epochs between progress lines.
LOG_EVERY: int = 10

#: Threshold turning a soft per-cell accuracy into the 0/1 label.
LABEL_THRESHOLD: float = 0.5


@dataclass
class TextMFEncoder(Encoder):
    """Free per-model embedding times a projected frozen question embedding.

    Holds the fold-in and prediction; the fit is the subclass's
    (:class:`bullpen.competitors.embedllm.textmf.TextMFFaithfulEncoder`).
    """

    #: [Q_total, d_text] frozen question embeddings, indexed by global question id
    q_text_emb: np.ndarray | None = field(default=None, repr=False)
    epochs: int = TEXTMF_EPOCHS
    lr: float = TEXTMF_LR
    weight_decay: float = TEXTMF_WEIGHT_DECAY
    alpha: float = TEXTMF_ALPHA
    batch_size: int = TEXTMF_BATCH_SIZE
    foldin_steps: int = FOLDIN_STEPS
    foldin_lr: float = FOLDIN_LR
    foldin_l2: float = FOLDIN_L2
    _net: tuple[torch.nn.Module, torch.nn.Module] | None = field(default=None, repr=False)

    interview_requirement: str = "training_grades_pool"

    def _q_block(self, cols: np.ndarray) -> torch.Tensor:
        """Frozen question vectors for the given global ids, as a device tensor."""
        import torch

        if self.q_text_emb is None:
            raise ValueError(
                f"{self.name} needs a frozen question text table and was given none; "
                "pass q_text_emb=<[Q_total, d_text] table indexed by global question id>"
            )
        table = np.asarray(self.q_text_emb, dtype=np.float64)
        if table.ndim != 2:
            raise ValueError(
                f"{self.name}: q_text_emb must be 2-d [Q_total, d_text], got {table.shape}"
            )
        ids = np.asarray(cols).ravel().astype(int)
        if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
            raise ValueError(
                f"{self.name}: q_text_emb covers question ids 0..{table.shape[0] - 1} but "
                f"was asked for {int(ids.min())}..{int(ids.max())}"
            )
        return torch.tensor(table[ids].astype(np.float32), device=torch_device())

    def _fitted(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        if self._net is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        return on_device(self._net[0]), on_device(self._net[1])

    @capped_threads
    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Optimise a fresh ``p`` (from zero, with an L2 pull) against the probe cells.

        An empty interview returns zero.
        """
        import torch

        text_proj, classifier = self._fitted()
        j = self.local(cols)
        y = np.asarray(bits, dtype=np.float64).ravel()
        if y.shape[0] != j.shape[0]:
            raise ValueError(f"{self.name}: {j.shape[0]} columns but {y.shape[0]} bits")
        if y.shape[0] == 0:
            return np.zeros(self.dim)

        device = torch_device()
        labels = torch.tensor((y >= LABEL_THRESHOLD).astype(np.int64), device=device)
        with torch.no_grad():
            q_proj = text_proj(self._q_block(cols))

        theta = torch.nn.Parameter(torch.zeros(self.dim, device=device))
        opt = torch.optim.Adam([theta], lr=self.foldin_lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(self.foldin_steps):
            logits = classifier(theta.unsqueeze(0) * q_proj)
            loss = loss_fn(logits, labels) + self.foldin_l2 * theta.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return theta.detach().cpu().numpy().astype(np.float64)

    def predict(self, theta: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """P(correct) per question, read off the classifier's positive class."""
        import torch

        text_proj, classifier = self._fitted()
        self.local(cols)
        device = torch_device()
        t = torch.tensor(np.asarray(theta, dtype=np.float64).astype(np.float32), device=device)
        with torch.no_grad():
            q_proj = text_proj(self._q_block(cols))
            probs = torch.softmax(classifier(t.unsqueeze(0) * q_proj), dim=-1)[:, 1]
        return probs.cpu().numpy().astype(np.float64)


__all__ = [
    "FOLDIN_L2",
    "FOLDIN_LR",
    "FOLDIN_STEPS",
    "LABEL_THRESHOLD",
    "TEXTMF_ALPHA",
    "TEXTMF_BATCH_SIZE",
    "TEXTMF_EPOCHS",
    "TEXTMF_LR",
    "TEXTMF_WEIGHT_DECAY",
    "TextMFEncoder",
]
