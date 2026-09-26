"""Answer-embedding banks: pooling a model's answers into one text vector, and the encoder.

The answer bank ``Ae`` is ``[M, Q, d]`` unit-norm answer embeddings with a coverage
mask ``Ae_mask``; an uncovered cell is a zero vector and a False in the mask. Pooling
uses covered cells only.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from bullpen.config import TextBanksConfig

#: The dtype a bank is computed in.
BANK_COMPUTE_DTYPE = np.float32

#: Guards the division when a pooled row is (numerically) zero.
_NORM_FLOOR = 1e-12


def pooled_model_text(
    Ae: np.ndarray, mask: np.ndarray, cols: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``([M, d] text block, [M] bool has-text)``: each model's mean covered answer embedding.

    The mean is renormalised to unit length; ``cols`` restricts it to one question
    block. A model with no covered cell in ``cols`` gets a NaN row.
    """
    Ae = np.asarray(Ae)
    mask = _narrow(Ae, mask, cols)
    return _pool(Ae, mask, unit_answers=False, renormalise=True)


def _narrow(Ae: np.ndarray, mask: np.ndarray, cols: np.ndarray | None) -> np.ndarray:
    """The coverage mask validated against the bank and cut to ``cols``."""
    mask = np.asarray(mask, dtype=bool)
    if Ae.ndim != 3:
        raise ValueError(f"answer bank must be 3-d [models, questions, d], got {Ae.shape}")
    if mask.shape != Ae.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match bank {Ae.shape[:2]}")
    if cols is None:
        return mask
    # only the mask is narrowed: fancy-indexing the bank itself would copy
    # gigabytes in order to drop a few thousand columns.
    keep = np.zeros(mask.shape[1], dtype=bool)
    keep[np.asarray(cols, dtype=int)] = True
    return mask & keep


def _pool(
    Ae: np.ndarray, mask: np.ndarray, unit_answers: bool, renormalise: bool
) -> tuple[np.ndarray, np.ndarray]:
    """One pooled row per model over its covered answers; NaN for a model with none."""
    has_text = mask.sum(axis=1) > 0
    T = np.full((Ae.shape[0], Ae.shape[2]), np.nan, dtype=np.float64)
    # row by row: a whole-array cast of the fp16 bank would double its memory
    for i in np.flatnonzero(has_text):
        block = Ae[i, np.flatnonzero(mask[i])].astype(np.float64)
        if unit_answers:
            block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), _NORM_FLOOR)
        pooled = block.mean(axis=0)
        T[i] = pooled / max(float(np.linalg.norm(pooled)), _NORM_FLOOR) if renormalise else pooled
    if not has_text.all():
        logger.warning(
            f"{int((~has_text).sum())} of {has_text.size} models have no covered answer on "
            "these columns; their text rows are NaN and no text arm may be fitted over them"
        )
    return T, has_text


class TextEncoder:
    """The frozen sentence encoder of the answer and question banks (``config/text_banks.yaml``).

    BGE-base through transformers: prefix, left truncation to the window,
    attention-mask mean pooling (or CLS), L2 norm in float32. ``tokenizer`` and
    ``model`` may be injected; otherwise they are loaded from ``cfg.encoder_hf_id``.
    """

    def __init__(self, cfg: TextBanksConfig, device: str | None = None, tokenizer=None, model=None):
        import torch

        self.cfg = cfg
        if tokenizer is None or model is None:  # pragma: no cover - needs a model download
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_hf_id)
            model = AutoModel.from_pretrained(cfg.encoder_hf_id)
        # keep the TAIL of a long text: a generation's answer is at its end
        tokenizer.truncation_side = "left"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = tokenizer
        self.model = model.to(torch.device(device)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self) -> int:
        return int(self.cfg.encoder_dim)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """L2-normalised ``[n, encoder_dim]`` float32 rows, in input order."""
        import torch

        batch_size = batch_size or self.cfg.batch_size
        out = np.empty((len(texts), self.dim), dtype=BANK_COMPUTE_DTYPE)
        for s in range(0, len(texts), batch_size):
            chunk = [self.cfg.encoder_prefix + t for t in texts[s : s + batch_size]]
            enc = self.tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_seq_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            if self.cfg.encoder_pooling == "cls":
                pooled = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            rows = torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy()
            if rows.shape[1] != self.dim:
                raise ValueError(
                    f"{self.cfg.encoder_hf_id} returned {rows.shape[1]}-d, config says {self.dim}"
                )
            out[s : s + batch_size] = rows
        return out
