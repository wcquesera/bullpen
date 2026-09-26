"""LLMmap's inference network, vendored from the authors' release.

Source: ``github.com/pasquini-dario/LLMmap`` at commit
``f661d550e3409d1bdf7e00b34bcec15296c5215e`` ("LLMmap0.2", 2025-07-24).
Copyright (c) 2024 pasquini-dario, MIT License — see ``LICENSE`` beside this file.

Vendored verbatim in behaviour (module names, parameter names and forward pass
are the release's, so the released ``data/pretrained_models/default/model.pt``
loads into :class:`InferenceModelLLMmap` with ``strict=True``):

* ``ClassToken``, ``TransformerBlock``, ``InferenceModelLLMmap``,
  ``euclidean_distance``, ``SiameseNetwork``
  — ``LLMmap/inference_model_archs.py:66-187``
* ``ContrastiveLoss`` — ``LLMmap/trainer.py:103-129``
* ``DEFAULT_HP`` — ``LLMmap/inference_model_archs.py:14-27``, with the
  training defaults of ``confs/default.json``

Changes from the release, none of which alter a forward pass:

* ``make_norm``'s partial/callable branch is dropped (the release only ever
  passes a string);
* :meth:`InferenceModelLLMmap.features` is split out of ``forward`` so the
  closed-set head can expose the same CLS vector (with the release default
  ``with_add_dense_class = False`` both paths return the same tensor);
* torch is imported lazily through :func:`build`.

Label convention, read from the code rather than the docstring: the release's
``ContrastiveLoss`` docstring says ``0 = same``, but its
``DatasetFactorySiamese.__getitem__`` (``LLMmap/dataset.py:161-176``) emits
``label = 1`` for the positive (same-LLM) pair. The trained behaviour is the
code's, so ``y = 1`` means SAME here.
"""

from __future__ import annotations

from functools import cache
from typing import Any

#: ``LLMmap/inference_model_archs.py:14-27`` with the optimiser of
#: ``confs/default.json``. ``emb_size`` is the text encoder's width; the release
#: uses ``intfloat/multilingual-e5-large-instruct`` (1024).
DEFAULT_HP: dict[str, Any] = {
    "num_blocks": 3,
    "feature_size": 384,
    "norm_layer": "BatchNorm1d",
    "num_heads": 4,
    "activation": "gelu",
    "optimizer": {"name": "Adam", "params": {"lr": 1e-4}},
    "with_add_dense_class": False,
    "emb_size": 1024,
    "num_queries": 8,
}


@cache
def build() -> dict[str, Any]:
    """The release's module classes, built once on first use."""
    import torch
    from torch import nn

    def get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "gelu":
            return nn.GELU()
        if name == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation: {name!r}")

    norms = {"BatchNorm1d": nn.BatchNorm1d, "LayerNorm": nn.LayerNorm}

    def make_norm(norm_cfg: str, dim: int) -> nn.Module:
        if norm_cfg not in norms:
            raise ValueError(f"Unknown norm_layer {norm_cfg!r}. Known: {list(norms)}")
        return norms[norm_cfg](dim)

    class ClassToken(nn.Module):
        def __init__(self, feature_size: int):
            super().__init__()
            self.token = nn.Parameter(torch.randn(1, 1, feature_size))

        def forward(self, x):  # (B, S, F)
            return self.token.expand(x.size(0), -1, -1)

    class TransformerBlock(nn.Module):
        def __init__(self, hp: dict):
            super().__init__()
            f = hp["feature_size"]
            act = get_activation(hp["activation"])
            self.norm1 = make_norm(hp["norm_layer"], f)
            self.attn = nn.MultiheadAttention(f, hp["num_heads"], batch_first=True)
            self.norm2 = make_norm(hp["norm_layer"], f)
            self.mlp = nn.Sequential(nn.Linear(f, f), act)

        def _apply_norm(self, norm, x):
            if isinstance(norm, nn.BatchNorm1d):
                return norm(x.transpose(1, 2)).transpose(1, 2)
            return norm(x)

        def forward(self, x):  # (B, S, F)
            x_norm = self._apply_norm(self.norm1, x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            return x + self.mlp(self._apply_norm(self.norm2, x))

    class InferenceModelLLMmap(nn.Module):
        def __init__(self, hp: dict = DEFAULT_HP, *, is_for_siamese: bool = False):
            super().__init__()
            f = hp["feature_size"]
            act = get_activation(hp["activation"])
            self.cls_token = ClassToken(f)
            self.proj = nn.Linear(hp["emb_size"] * 2, f)
            self.act = act
            self.blocks = nn.ModuleList(TransformerBlock(hp) for _ in range(hp["num_blocks"]))
            if not is_for_siamese:
                if hp["with_add_dense_class"]:
                    self.pre_head = nn.Sequential(nn.Linear(f, f // 2), act)
                    head_in = f // 2
                else:
                    self.pre_head = nn.Identity()
                    head_in = f
                self.head = nn.Linear(head_in, hp["num_classes"])
            else:
                self.pre_head = nn.Identity()
                self.head = nn.Identity()

        def features(self, traces):  # (B, Q, emb_size * 2) -> (B, F)
            x = self.act(self.proj(traces))
            x = torch.cat([self.cls_token(x), x], dim=1)  # prepend [CLS], no pos-enc
            for blk in self.blocks:
                x = blk(x)
            return x[:, 0]

        def forward(self, traces):
            return self.head(self.pre_head(self.features(traces)))

    def euclidean_distance(a, b, eps: float = 1e-8):
        return torch.sqrt(((a - b) ** 2).sum(dim=1, keepdim=True) + eps)

    class SiameseNetwork(nn.Module):
        def __init__(self, feature_extractor: nn.Module):
            super().__init__()
            self.f = feature_extractor
            self.bn = nn.BatchNorm1d(1)
            self.fc = nn.Linear(1, 1)

        def forward(self, x):  # (B, 2, Q, emb_size * 2) -> (B, 1) in (0, 1)
            dist = euclidean_distance(self.f(x[:, 0]), self.f(x[:, 1]))
            return torch.sigmoid(self.fc(self.bn(dist)))

    class ContrastiveLoss(nn.Module):
        """The release's loss; ``y = 1`` is the SAME-model pair (module docstring)."""

        def __init__(self, margin: float = 1.0):
            super().__init__()
            self.margin = margin

        def forward(self, y_pred, y_true):
            y_true = y_true.float().view_as(y_pred)
            square_pred = y_pred.pow(2)
            margin_square = torch.clamp(self.margin - y_pred, min=0.0).pow(2)
            return ((1.0 - y_true) * square_pred + y_true * margin_square).mean()

    return {
        "InferenceModelLLMmap": InferenceModelLLMmap,
        "SiameseNetwork": SiameseNetwork,
        "ContrastiveLoss": ContrastiveLoss,
        "euclidean_distance": euclidean_distance,
    }


__all__ = ["DEFAULT_HP", "build"]
