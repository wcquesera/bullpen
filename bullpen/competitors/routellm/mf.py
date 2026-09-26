# Copyright 2024 LMSYS / the RouteLLM authors. Licensed under the Apache License,
# Version 2.0 (see LICENSE in this directory).
#
# Vendored from https://github.com/lm-sys/RouteLLM at commit
# 0b64fdafe049e596a3f5657c219329f24af24198,
# routellm/routers/matrix_factorization/train_matrix_factorization.py:44-101
# (``MFModel_Train``) and model.py:74-126 (``MFModel``, the inference twin).
#
# Modifications:
#   * the frozen prompt table ``Q`` is passed in as a tensor rather than read from
#     an ``.npy`` path, and is not an ``nn.Embedding`` (it has no gradient either
#     way; this only keeps it out of ``state_dict``);
#   * the in-place ``prompt_embed += noise`` is written out of place (numerically
#     the same; it keeps a caller's prompt tensor unmutated);
#   * noise is drawn from an explicit generator so a fit is reproducible per seed;
#   * ``score`` exposes delta(M, q) (paper Eq. 12) for a matrix of models and
#     prompts at once, which the fold-in needs;
#   * ``load_published`` reads the released ``PyTorchModelHubMixin`` checkpoints,
#     whose layers are wrapped in ``nn.Sequential`` (``text_proj.0.weight``).
# The forward computation is otherwise line-for-line upstream's.
"""RouteLLM's matrix-factorisation router, the model definition only.

Module level (not built inside a factory) so a fitted encoder pickles.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


class MFModel(torch.nn.Module):
    """``MFModel_Train`` minus the ``Q`` table; delta(M, q) = w2^T (v_m * W1 v_q)."""

    def __init__(self, dim: int, num_models: int, text_dim: int, use_proj: bool = True) -> None:
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        if use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert text_dim == dim, (
                f"text_dim {text_dim} must be equal to dim {dim} if not using projection"
            )
        self.classifier = torch.nn.Linear(dim, 1, bias=False)  # bias should be False!

    def project(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        return self.text_proj(prompt_embed) if self.use_proj else prompt_embed

    def forward(
        self,
        model_win: torch.Tensor,
        model_loss: torch.Tensor,
        prompt_embed: torch.Tensor,
        alpha: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        model_win_embed = F.normalize(self.P(model_win), p=2, dim=1)
        model_loss_embed = F.normalize(self.P(model_loss), p=2, dim=1)
        if alpha:
            # adding noise to stablize the training
            noise = torch.randn(prompt_embed.shape, device=prompt_embed.device, generator=generator)
            prompt_embed = prompt_embed + noise * alpha
        prompt_embed = self.project(prompt_embed)
        return self.classifier((model_win_embed - model_loss_embed) * prompt_embed).squeeze(-1)

    def score(self, model_embed: torch.Tensor, prompt_proj: torch.Tensor) -> torch.Tensor:
        """delta(M, q) for ``[n, dim]`` L2-normalised models x ``[k, dim]`` projected prompts."""
        w = self.classifier.weight.reshape(-1)
        return (model_embed * w) @ prompt_proj.T


def load_published(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a released checkpoint's ``nn.Sequential`` keys onto :class:`MFModel`'s."""
    renamed = {
        "text_proj.0.weight": "text_proj.weight",
        "classifier.0.weight": "classifier.weight",
    }
    return {renamed.get(k, k): v for k, v in state.items()}


__all__ = ["MFModel", "load_published"]
