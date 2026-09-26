"""``comp_irtnet``: IrtNet (Chen et al., arXiv 2510.00844).

Reimplemented from the paper (§3.1-3.2, §4.1) and the authors' release
(``github.com/JianhaoChen-nju/IrtNet`` at ``6cf5eec``, ``MoEClassifier``)::

    v_q   = all-mpnet-base-v2(q) (+ N(0, 0.05) noise while training)
    h_q   = SharedExpert(v_q) + sum_i softmax(gate(v_q))_i Expert_i(v_q)
            Expert = Linear(768, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 256)
    a_q   = Linear(256, d)(h_q),  b_q = Linear(256, 1)(h_q)
    P     = sigmoid(a_q . theta_m - b_q),  theta_m = nn.Embedding(M, d)[m], d = 232

Faithful (the release's ``pipeline.sh``): 39 dense experts, Adam lr 1e-4, weight decay
1e-4, batch 2048, up to 30 epochs, ReduceLROnPlateau(0.1, 2), best-val-accuracy
checkpoint, early stop after 5 epochs, 8.4% of the fitted columns held out. Deviations:
the load-balancing bias is omitted (it cannot change the output with every expert
active); experts run as one batched einsum; model-table init N(0, 0.01^2) instead of
N(0, 1), since with ~15x fewer cells the default init leaves the selected table at its
random draw; a held-out model is placed by the shared fold-in (item side frozen).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from bullpen.competitors.irtfamily.common import TextIRTEncoder

if TYPE_CHECKING:  # pragma: no cover
    import torch

IRTNET_DIM: int = 232
IRTNET_EXPERTS: int = 39
IRTNET_EXPERT_HIDDEN: int = 512
IRTNET_EXPERT_OUT: int = 256
IRTNET_DROPOUT: float = 0.5
IRTNET_NOISE: float = 0.05
IRTNET_EPOCHS: int = 30
IRTNET_LR: float = 1e-4
IRTNET_WEIGHT_DECAY: float = 1e-4
IRTNET_BATCH_SIZE: int = 2048
IRTNET_PATIENCE: int = 5
#: Model-table init scale (upstream: N(0, 1)).
IRTNET_INIT_STD: float = 0.01
#: 3,000 validation questions of 35,673 (§4.1)
IRTNET_VAL_FRAC: float = 3000 / 35673
UPSTREAM_COMMIT: str = "6cf5eeced941226fcd9d0a50b10e4130f5b85eae"


def _build(d_text: int, dim: int, n_experts: int, hidden: int, out: int, dropout: float):
    import torch
    from torch import nn

    class DenseMoEIRT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(d_text, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out)
            )
            self.gate = nn.Linear(d_text, n_experts)
            # the routed experts' weights stacked, each initialised as nn.Linear is
            self.w1 = nn.Parameter(torch.empty(n_experts, d_text, hidden))
            self.b1 = nn.Parameter(torch.empty(n_experts, 1, hidden))
            self.w2 = nn.Parameter(torch.empty(n_experts, hidden, out))
            self.b2 = nn.Parameter(torch.empty(n_experts, 1, out))
            for w, b, fan_in in ((self.w1, self.b1, d_text), (self.w2, self.b2, hidden)):
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(w, -bound, bound)
                nn.init.uniform_(b, -bound, bound)
            self.drop = nn.Dropout(dropout)
            self.difficulty = nn.Linear(out, 1)
            self.discrimination = nn.Linear(out, dim)

        def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.training:
                v = v + torch.randn_like(v) * IRTNET_NOISE
            weights = torch.softmax(self.gate(v), dim=-1)  # [B, E]
            h = torch.relu(torch.einsum("bi,eih->ebh", v, self.w1) + self.b1)
            h = torch.einsum("ebh,eho->ebo", self.drop(h), self.w2) + self.b2  # [E, B, out]
            h_q = self.shared(v) + torch.einsum("ebo,be->bo", h, weights)
            return self.discrimination(h_q), self.difficulty(h_q).squeeze(-1)

    return DenseMoEIRT()


@dataclass
class IrtNetEncoder(TextIRTEncoder):
    """IrtNet: dense-MoE question network emitting 2PL item parameters."""

    dim: int = IRTNET_DIM
    epochs: int = IRTNET_EPOCHS
    lr: float = IRTNET_LR
    weight_decay: float = IRTNET_WEIGHT_DECAY
    batch_size: int = IRTNET_BATCH_SIZE
    val_frac: float = IRTNET_VAL_FRAC
    patience: int = IRTNET_PATIENCE
    soft_labels: bool = False
    select_on: str = "acc"
    init_std: float = IRTNET_INIT_STD
    n_experts: int = IRTNET_EXPERTS
    expert_hidden: int = IRTNET_EXPERT_HIDDEN
    expert_out: int = IRTNET_EXPERT_OUT
    dropout: float = IRTNET_DROPOUT

    METHOD: ClassVar[str] = "IrtNet"

    def build_item_net(self, d_text: int) -> torch.nn.Module:
        return _build(
            d_text, self.dim, self.n_experts, self.expert_hidden, self.expert_out, self.dropout
        )

    def init_model_table(self, table: torch.nn.Embedding) -> None:
        import torch

        torch.nn.init.normal_(table.weight, 0.0, self.init_std)

    def item_params(self, net: torch.nn.Module, q: torch.Tensor) -> tuple:
        return net(q)

    def scheduler(self, opt):
        import torch

        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)


__all__ = ["IRTNET_DIM", "UPSTREAM_COMMIT", "IrtNetEncoder"]
