"""``comp_irtrouter``: IRT-Router's MIRT-Router (Song et al., arXiv 2506.01048).

Reimplemented from the paper (§4.1-4.2.1, §5.5) and the authors' release
(``github.com/Mercidaiha/IRT-Router`` at ``e8f258c``, ``router/MIRT.py``)::

    theta_M = W_theta e_M                     (Linear, no bias, width N = 25)
    a_q     = softplus(W_a e_q)               (Linear, no bias)
    b_q     = W_b e_q                         (Linear, no bias)
    P       = sigmoid(a_q . theta_M - b_q)    BCE against the cell's score

Faithful: response function, N = 25, Adam lr 1e-3 (the release default), batch 512,
9 epochs, no weight decay, no checkpoint selection. Deviations: ``e_M`` is the model's
one-hot instead of an embedded LLM profile (the profile fields are our own task labels),
so ``theta_M`` is a free per-model vector with ``nn.Linear``'s init; an epoch is
upstream's 955 optimiser steps; questions are embedded with all-mpnet-base-v2 instead of
bert-base-uncased; the test-time query warm-up is not used; a held-out model is placed by
the shared fold-in (BCE over its observed cells, item side frozen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from bullpen.competitors.irtfamily.common import TextIRTEncoder

if TYPE_CHECKING:  # pragma: no cover
    import torch

#: §5.5 "the Dimension N of MIRT-Router and NIRT-Router are both set to 25";
#: ``train_mirt.py`` ``knowledge_n = 25``.
IRTROUTER_DIM: int = 25
IRTROUTER_EPOCHS: int = 9
IRTROUTER_LR: float = 1e-3
IRTROUTER_BATCH_SIZE: int = 512
#: One upstream epoch in optimiser steps: ``data/train.csv`` at e8f258c has
#: 488,600 (query, LLM) rows, so ceil(488,600 / 512) = 955 steps.
IRTROUTER_STEPS_PER_EPOCH: int = 955
UPSTREAM_COMMIT: str = "e8f258ced4ec3c40d795403603acd8c1cdfb994d"


@dataclass
class IRTRouterEncoder(TextIRTEncoder):
    """MIRT-Router with a one-hot model input (no profile)."""

    dim: int = IRTROUTER_DIM
    epochs: int = IRTROUTER_EPOCHS
    lr: float = IRTROUTER_LR
    weight_decay: float = 0.0
    batch_size: int = IRTROUTER_BATCH_SIZE
    val_frac: float = 0.0
    soft_labels: bool = True
    steps_per_epoch: int | None = IRTROUTER_STEPS_PER_EPOCH

    METHOD: ClassVar[str] = "MIRT-Router"

    def build_item_net(self, d_text: int) -> torch.nn.Module:
        import torch

        net = torch.nn.Module()
        net.a = torch.nn.Linear(d_text, self.dim, bias=False)
        net.b = torch.nn.Linear(d_text, 1, bias=False)
        return net

    def init_model_table(self, table: torch.nn.Embedding) -> None:
        # theta = Linear(n_models, N, bias=False)(one_hot): nn.Linear's default
        # kaiming_uniform(a=sqrt(5)) is U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        import torch

        bound = 1.0 / math.sqrt(table.weight.shape[0])
        with torch.no_grad():
            table.weight.uniform_(-bound, bound)

    def item_params(self, net: torch.nn.Module, q: torch.Tensor) -> tuple:
        import torch

        return torch.nn.functional.softplus(net.a(q)), net.b(q).squeeze(-1)


__all__ = ["IRTROUTER_DIM", "UPSTREAM_COMMIT", "IRTRouterEncoder"]
