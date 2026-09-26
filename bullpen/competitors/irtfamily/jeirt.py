"""``comp_jeirt``: JE-IRT, joint-embedding item response theory (arXiv 2509.22888).

Reimplemented from the paper (§3 Eq. 2-6, §4.1-4.3, App. H)::

    E_Q   = g(f_base(Q))     frozen sentence encoder f_base, adapter g:
                             Linear(768, 1536) -> ReLU -> Linear(1536, d)
    E_M   = T[M]             free embedding table; the per-model code at d = 256
    P     = sigmoid(E_Q . E_M / |E_Q|  -  |E_Q|)     (Eq. 2-3), BCE on binary cells (Eq. 6)

The direction of ``E_Q`` is the question's topic, its norm its difficulty. Faithful:
response function, adapter shape, all-mpnet-base-v2 base encoder, d = 256, best-val-loss
checkpoint over 100 epochs with 10% of the fitted columns held out for selection.
Unstated in the paper, so ours: Adam lr 1e-3, batch 2048, no weight decay, ReLU,
model-table init N(0, 0.01^2) (the default N(0, 1) init leaves the selected table at its
random draw), early stop after 10 epochs without improvement. A held-out model is placed
by the shared fold-in (item side frozen, as in §4.3) with an L2 pull to the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from bullpen.competitors.irtfamily.common import TextIRTEncoder

if TYPE_CHECKING:  # pragma: no cover
    import torch

JEIRT_PAPER_DIM: int = 256
JEIRT_EPOCHS: int = 100
JEIRT_LR: float = 1e-3
JEIRT_BATCH_SIZE: int = 2048
JEIRT_VAL_FRAC: float = 0.1
JEIRT_PATIENCE: int = 10
#: Model-table init scale (not stated in the paper).
JEIRT_INIT_STD: float = 0.01

#: Floor on |E_Q| inside the direction, so a zero adapter output has no NaN.
_NORM_EPS: float = 1e-8


@dataclass
class JEIRTEncoder(TextIRTEncoder):
    """JE-IRT: projected ability along the question direction minus its norm."""

    dim: int = JEIRT_PAPER_DIM
    epochs: int = JEIRT_EPOCHS
    lr: float = JEIRT_LR
    weight_decay: float = 0.0
    batch_size: int = JEIRT_BATCH_SIZE
    val_frac: float = JEIRT_VAL_FRAC
    patience: int = JEIRT_PATIENCE
    soft_labels: bool = False
    select_on: str = "bce"
    init_std: float = JEIRT_INIT_STD

    METHOD: ClassVar[str] = "JE-IRT"

    def build_item_net(self, d_text: int) -> torch.nn.Module:
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(d_text, 2 * d_text),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * d_text, self.dim),
        )

    def init_model_table(self, table: torch.nn.Embedding) -> None:
        import torch

        torch.nn.init.normal_(table.weight, 0.0, self.init_std)

    def item_params(self, net: torch.nn.Module, q: torch.Tensor) -> tuple:
        e = net(q)
        norm = e.norm(dim=-1)
        return e / norm.clamp_min(_NORM_EPS).unsqueeze(-1), norm


__all__ = ["JEIRT_PAPER_DIM", "JEIRTEncoder"]
