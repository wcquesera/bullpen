"""LOCUS network: a set-transformer encoder over (query embedding, correctness) pairs
and a bilinear-MLP correctness decoder.

Reimplemented from Patel, Cocke & Joshi, *LOCUS* (arXiv 2601.21082, §2, Eqs. 1-4) and
checked against the authors' release (``github.com/patel-shivam/locus_code_release`` at
``e4ebf69``; no licence, so nothing is copied). Comments cite ``<file>:<line>`` there;
where paper and code differ, the code is followed. The blocks are the Set Transformer's
(Lee et al. 2019) with the release's two idiosyncrasies kept: a ReLU between the
multihead block's two LayerNorms, and a single-Linear token map over
``[q, y, q * (2y - 1)]`` (so the configured encoder dropout never fires).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    """Linear layers with ReLU (and dropout) between them, none after the last.

    ``encoder_attention_modules.py:18-33``. A two-entry ``sizes`` is one Linear,
    which is why the encoder's dropout is inert.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MAB(nn.Module):
    """Multihead attention block, ``encoder_attention_modules.py:48-114``."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        if d % heads:
            raise ValueError(f"width {d} is not divisible by {heads} heads")
        self.d, self.h = d, heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.fc_o = nn.Linear(d, d, bias=False)
        self.ln0, self.ln1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.rff = _mlp([d, 2 * d, d])

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        mask_q: torch.Tensor | None = None,
        mask_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, SQ, _ = Q.shape
        hd = self.d // self.h

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.h, hd).transpose(1, 2)

        q, k, v = split(self.W_q(Q)), split(self.W_k(K)), split(self.W_v(K))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if mask_k is not None:
            att = att.masked_fill(~mask_k[:, None, None, :], float("-inf"))
        H = (att.softmax(dim=-1) @ v).transpose(1, 2).contiguous().view(B, SQ, self.d)
        out = torch.relu(self.ln0(self.fc_o(H) + Q))
        out = self.ln1(out + self.rff(out))
        if mask_q is not None:
            out = out.masked_fill(~mask_q[..., None], 0.0)
        return out


class ISAB(nn.Module):
    """Induced set attention, ``encoder_attention_modules.py:129-142``."""

    def __init__(self, d: int, heads: int, m: int) -> None:
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m, d))
        nn.init.xavier_uniform_(self.I)
        self.mab1 = MAB(d, heads)
        self.mab2 = MAB(d, heads)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        H = self.mab1(self.I.expand(X.size(0), -1, -1), X, mask_k=mask)
        return self.mab2(X, H, mask_q=mask)


class PMA(nn.Module):
    """Learned-query pooling (paper Eq. 3), ``encoder_attention_modules.py:145-154``."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        # upstream leaves S at its randn draw, no xavier (line 148)
        self.S = nn.Parameter(torch.randn(1, 1, d))
        self.mab = MAB(d, heads)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.mab(self.S.expand(X.size(0), -1, -1), X, mask_k=mask)


class LocusSetEncoder(nn.Module):
    """``F_theta``: a set of (q, y) pairs -> one ``d``-vector (paper Eqs. 1-3)."""

    def __init__(self, q_dim: int, d: int, heads: int, layers: int, m_induce: int) -> None:
        super().__init__()
        self.input = _mlp([2 * q_dim + 1, d])
        self.layers = nn.ModuleList(ISAB(d, heads, m_induce) for _ in range(layers))
        self.pma = PMA(d, heads)
        self.out = _mlp([d, d])

    def forward(self, q: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``q`` [B, A, Dq], ``y`` [B, A, 1] in [0, 1], ``mask`` [B, A] bool -> [B, d]."""
        h = self.input(torch.cat([q, y, q * (2 * y - 1)], dim=-1))
        for layer in self.layers:
            h = layer(h, mask)
        return self.out(self.pma(h, mask).squeeze(1))


class BilinearMLPDecoder(nn.Module):
    """``G_psi`` (paper Eq. 4), ``decoder_and_losses.py:13-29``."""

    def __init__(self, d: int, q_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.U = nn.Linear(q_dim, d)
        self.mlp_head = _mlp([3 * d, hidden, 1], dropout=dropout)

    def forward(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """``z`` [..., d], ``q`` [..., Dq] broadcastable -> logits [...]."""
        uq = self.U(q)
        z = z.expand_as(uq)
        return self.mlp_head(torch.cat([z, uq, z * uq], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class LocusConfig:
    """The release's CLI defaults, ``training_script.py:259-395`` (L = 2 layers, as run)."""

    z_dim: int = 128
    heads: int = 4
    layers: int = 2
    m_induce: int = 64
    decoder_hidden: int = 64
    decoder_dropout: float = 0.1
    anchors: int = 1024
    enc_lr: float = 2e-4
    dec_lr: float = 8e-4
    enc_weight_decay: float = 5e-2
    dec_weight_decay: float = 5e-2
    noise_start: float = 0.10
    noise_end: float = 0.05
    noise_end_epoch: int = 500
    max_epochs: int = 2500
    patience: int = 500
    batch_models: int = 128
    grad_clip: float = 1.0


def noise_std(epoch: int, cfg: LocusConfig) -> float:
    """Cosine anneal ``noise_start -> noise_end`` over ``[0, noise_end_epoch]``.

    ``training_script.py:80-111``; the same schedule drives encoder and decoder
    query noise (both defaults 0.10 -> 0.05).
    """
    if epoch >= cfg.noise_end_epoch:
        return cfg.noise_end
    w = 0.5 * (1.0 + math.cos(math.pi * epoch / cfg.noise_end_epoch))
    return cfg.noise_end + (cfg.noise_start - cfg.noise_end) * w


class LocusNet(nn.Module):
    """Encoder and decoder together; ``router_model.py:54-178`` minus the unused knobs."""

    def __init__(self, q_dim: int, cfg: LocusConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = LocusSetEncoder(q_dim, cfg.z_dim, cfg.heads, cfg.layers, cfg.m_induce)
        self.decoder = BilinearMLPDecoder(cfg.z_dim, q_dim, cfg.decoder_hidden, cfg.decoder_dropout)

    def optimizer(self) -> torch.optim.Optimizer:
        """AdamW with separate encoder/decoder groups, ``training_script.py:489-494``."""
        c = self.cfg
        return torch.optim.AdamW(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": c.enc_lr,
                    "weight_decay": c.enc_weight_decay,
                },
                {
                    "params": self.decoder.parameters(),
                    "lr": c.dec_lr,
                    "weight_decay": c.dec_weight_decay,
                },
            ]
        )

    @torch.no_grad()
    def embed(self, Qa: torch.Tensor, Ya: torch.Tensor, batch: int = 64) -> torch.Tensor:
        """Deterministic forward pass: ``Qa`` [A, Dq], ``Ya`` [M, A] -> Z [M, d].

        ``evaluation.py:14-55``: every model sees the same anchor set, all unmasked.
        """
        self.eval()
        A = Qa.shape[0]
        out = []
        for s in range(0, Ya.shape[0], batch):
            y = Ya[s : s + batch]
            B = y.shape[0]
            mask = torch.ones(B, A, dtype=torch.bool, device=Qa.device)
            out.append(self.encoder(Qa.expand(B, A, -1), y.unsqueeze(-1), mask))
        return torch.cat(out)

    @torch.no_grad()
    def score(self, Z: torch.Tensor, Q: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """Logits [N, M] for every question against every model, ``evaluation.py:58-106``."""
        self.eval()
        rows = []
        for s in range(0, Q.shape[0], chunk):
            uq = self.decoder.U(Q[s : s + chunk])  # [n, d]
            n, M = uq.shape[0], Z.shape[0]
            z = Z.unsqueeze(0).expand(n, M, -1)
            u = uq.unsqueeze(1).expand(n, M, -1)
            rows.append(self.decoder.mlp_head(torch.cat([z, u, z * u], dim=-1)).squeeze(-1))
        return torch.cat(rows)

    def train_step(
        self,
        Qa: torch.Tensor,
        Ya: torch.Tensor,
        opt: torch.optim.Optimizer,
        epoch: int,
        gen: torch.Generator,
    ) -> float:
        """One epoch over the training models, ``train_utils.py:16-69``.

        The decoder targets are the anchors themselves (upstream's per-model target
        permutation is a no-op on the mean BCE). Noise is added to the encoder and
        decoder query copies only; ``gen`` must live on the device of ``Qa``/``Ya``.
        """
        self.train()
        std = noise_std(epoch, self.cfg)
        M, A = Ya.shape
        order = torch.randperm(M, generator=gen, device=gen.device)
        loss_fn = nn.BCEWithLogitsLoss()
        total, n = 0.0, 0
        for s in range(0, M, self.cfg.batch_models):
            idx = order[s : s + self.cfg.batch_models]
            B = idx.numel()
            y = Ya[idx]
            q = Qa.expand(B, A, -1)
            q_enc = q + torch.randn(q.shape, generator=gen, device=gen.device) * std
            q_dec = q + torch.randn(q.shape, generator=gen, device=gen.device) * std
            mask = torch.ones(B, A, dtype=torch.bool, device=Ya.device)
            z = self.encoder(q_enc, y.unsqueeze(-1), mask)
            logits = self.decoder(z.unsqueeze(1), q_dec)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.cfg.grad_clip)
            opt.step()
            total += float(loss.detach())
            n += 1
        return total / max(n, 1)


def routing_hit1(logits: torch.Tensor, Y: torch.Tensor) -> float:
    """Fraction of questions whose argmax-logit model answered correctly.

    ``evaluation.py:162-196`` with k = 1, over every question (zero-positive
    questions count as misses there too). ``logits``, ``Y`` are [N, M].
    """
    top = logits.argmax(dim=1)
    return float((Y.gather(1, top[:, None]).squeeze(1) >= 0.5).float().mean())
