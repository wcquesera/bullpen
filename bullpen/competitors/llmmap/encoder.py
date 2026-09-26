"""LLMmap's open-set inference model (Pasquini et al., USENIX Security 2025), retrained.

LLMmap fingerprints a model from its responses to 8 fixed queries: each (query,
response) pair is embedded, the 8 tokens go through a small transformer (3 blocks, 4
heads, 384-d, CLS readout, no positional encoding), and the CLS vector is the signature.
The network, Siamese head and contrastive loss are vendored (``net.py``, MIT, commit
``f661d55``); the training recipe is the release's (Adam lr 1e-4, batch 128, 500k
pairs/epoch, up to 50 epochs, early stopping with patience 5).

Retraining on our answers changes: the training traces are 8-question draws from each
model's :data:`TRACE_POOL` top-ranked answered pool questions, ranked by §5 Eq. 2 (mean
pairwise cosine distance among the training models' answers) and ordered greedily over
the :data:`SHORTLIST`; the text encoder is the answer bank's BGE-base; tokens are
L2-normalised and rescaled to :data:`TOKEN_NORM`. This class signs a model on the first 8
ranked questions; :class:`~bullpen.competitors.llmmap.orig.LLMmapOrigEncoder`
(``comp_llmmap_orig``) signs it on LLMmap's own published probes. Width 384.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from loguru import logger

from bullpen.competitors.llmmap.net import DEFAULT_HP, build
from bullpen.models.base import Encoder, capped_threads, on_device, torch_device

#: The published signature width (paper §6.2 "m = 384"; ``DEFAULT_HP``).
LLMMAP_DIM: int = 384

#: Queries per trace (paper §5, Table G.2; ``confs/queries/default.json``).
N_QUERIES: int = 8

#: Candidates carried into the greedy ordering.
SHORTLIST: int = 256

#: Per-model trace pool: the model's this-many highest-ranked answered queries.
TRACE_POOL: int = 64

#: A question is a query candidate only if this fraction of the training models
#: answered it (relaxed automatically when fewer than the shortlist clear it).
MIN_COLUMN_COVERAGE: float = 0.70

#: Fraction of training-model pairs the greedy ordering treats as confusable.
CONFUSABLE_FRAC: float = 0.05

#: Norm both halves of every token are rescaled to: the median L2 norm of the release's
#: multilingual-e5-large-instruct states. At unit norm the learned CLS token dominates the
#: shared BatchNorm and the signature is nearly constant.
TOKEN_NORM: float = 20.7

#: The release's ``confs/default.json`` training budget.
PAIRS_PER_EPOCH: int = 500_000
PAIRS_PER_EVAL: int = 5_000
BATCH_SIZE: int = 128
MAX_EPOCHS: int = 50
EARLY_STOP_PATIENCE: int = 5
LR: float = 1e-4
MARGIN: float = 1.0

#: Fraction of trace positions held out for validation pairs.
VAL_FRACTION: float = 0.15

#: Columns read per chunk when scanning the answer bank.
SCAN_CHUNK: int = 512


@dataclass
class LLMmapEncoder(Encoder):
    """LLMmap: transformer over (query embedding, response embedding) tokens, Siamese-trained."""

    #: ``[M_all, Q_all, Da]`` frozen answer embeddings by global id; an all-zero
    #: cell is an unanswered one
    answer_bank: np.ndarray | None = field(default=None, repr=False)
    #: ``[Q_all, Dq]`` frozen question embeddings by global id
    question_bank: np.ndarray | None = field(default=None, repr=False)
    #: which rows of ``answer_bank`` this fold's ``A`` rows are
    rows: np.ndarray | None = field(default=None, repr=False)

    n_queries: int = N_QUERIES
    shortlist: int = SHORTLIST
    trace_pool: int = TRACE_POOL
    min_coverage: float = MIN_COLUMN_COVERAGE
    pairs_per_epoch: int = PAIRS_PER_EPOCH
    pairs_per_eval: int = PAIRS_PER_EVAL
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOP_PATIENCE
    lr: float = LR
    token_norm: float = TOKEN_NORM

    #: global question ids in query-strategy order (greedy pool first)
    ranking: np.ndarray | None = field(default=None, repr=False)
    #: the fixed probe set, ``ranking[:n_queries]``
    probes: np.ndarray | None = field(default=None, repr=False)
    history: list[dict] = field(default_factory=list, repr=False)
    best_epoch: int = -1
    _net: object | None = field(default=None, repr=False)
    #: the backbone's weights and hyperparameters, what a pickle carries instead of it
    _net_state: dict | None = field(default=None, repr=False)
    _hp: dict | None = field(default=None, repr=False)
    #: mean training template, the placement of a model that answered nothing
    _mean: np.ndarray | None = field(default=None, repr=False)

    interview_requirement: str = "nothing"
    has_decoder: bool = False

    SHARED_BANKS: ClassVar[dict[str, str]] = {"answer_bank": "Ae", "question_bank": "Qe"}

    # -- the two frozen banks ----------------------------------------------- #
    def _check_banks(self) -> None:
        if self.answer_bank is None or self.question_bank is None:
            raise ValueError(
                f"{self.name} needs both the answer bank and the question bank; LLMmap "
                "reads (query, response) text pairs, so without them it is not LLMmap"
            )

    def _answers(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """``[len(rows), len(cols), Da]`` float32, one open-mesh read."""
        return np.asarray(
            self.answer_bank[np.ix_(np.asarray(rows, int), np.asarray(cols, int))],
            dtype=np.float32,
        )

    def _answered(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """``[len(rows), len(cols)]`` bool, scanned in column chunks."""
        cols = np.asarray(cols, int)
        out = np.zeros((len(rows), len(cols)), dtype=bool)
        for s in range(0, len(cols), SCAN_CHUNK):
            e = self._answers(rows, cols[s : s + SCAN_CHUNK])
            out[:, s : s + SCAN_CHUNK] = np.abs(e).sum(axis=2) > 0
        return out

    def _tokens(self, row: int, qids: np.ndarray) -> np.ndarray:
        """``[len(qids), Dq + Da]`` — the release's ``concat([q_emb, o_emb])`` per query."""
        qids = np.asarray(qids, int)
        q = np.asarray(self.question_bank[qids], dtype=np.float32)
        a = self._answers(np.array([row]), qids)[0]
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        return np.concatenate([q, a], axis=1) * self.token_norm

    # -- query strategy ----------------------------------------------------- #
    def _discrepancy(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per column: mean pairwise cosine distance among answering rows, and coverage."""
        score = np.zeros(len(cols))
        cover = np.zeros(len(cols))
        for s in range(0, len(cols), SCAN_CHUNK):
            e = self._answers(rows, cols[s : s + SCAN_CHUNK])
            norm = np.linalg.norm(e, axis=2)
            ok = norm > 0
            e = np.where(ok[:, :, None], e / np.clip(norm, 1e-12, None)[:, :, None], 0.0)
            n = ok.sum(axis=0).astype(np.float64)
            sq = (e.sum(axis=0) ** 2).sum(axis=1)
            good = n >= 2
            d = np.zeros_like(n)
            d[good] = 1.0 - (sq[good] - n[good]) / (n[good] * (n[good] - 1.0))
            del e
            score[s : s + SCAN_CHUNK] = d
            cover[s : s + SCAN_CHUNK] = n / len(rows)
        return score, cover

    def _greedy(self, rows: np.ndarray, shortlist: np.ndarray) -> np.ndarray:
        """Order ``shortlist`` greedily by the confusability of the hardest model pairs."""
        e = self._answers(rows, shortlist).astype(np.float64)
        norm = np.linalg.norm(e, axis=2, keepdims=True)
        e = np.where(norm > 0, e / np.clip(norm, 1e-12, None), 0.0)
        grams = np.einsum("mqd,nqd->qmn", e, e, optimize=True)
        n = len(rows)
        off = ~np.eye(n, dtype=bool)
        k_hard = max(round(CONFUSABLE_FRAC * off.sum()), 1)

        def confusability(G: np.ndarray) -> float:
            d = np.sqrt(np.clip(np.diag(G), 1e-12, None))
            vals = (G / np.outer(d, d))[off]
            return float(np.partition(vals, -k_hard)[-k_hard:].mean())

        chosen: list[int] = []
        G = np.zeros((n, n))
        remaining = list(range(len(shortlist)))
        while remaining:
            objs = [confusability(G + grams[c]) for c in remaining]
            best = remaining.pop(int(np.argmin(objs)))
            G += grams[best]
            chosen.append(int(shortlist[best]))
        return np.array(chosen, dtype=int)

    def _pool(
        self, row: int, cols: np.ndarray | None = None, limit: int | None = None
    ) -> np.ndarray:
        """The model's answered queries in ranking order (restricted to ``cols`` if given)."""
        ranking = self.ranking if cols is None else self.ranking[np.isin(self.ranking, cols)]
        limit = self.trace_pool if limit is None else limit
        out: list[int] = []
        for s in range(0, len(ranking), SCAN_CHUNK):
            block = ranking[s : s + SCAN_CHUNK]
            ok = self._answered(np.array([row]), block)[0]
            out.extend(block[ok].tolist())
            if len(out) >= limit:
                break
        return np.array(out[:limit], dtype=int)

    # -- fitting ------------------------------------------------------------ #
    def _fitted(self):
        if self._net is None and self._net_state is not None:
            # rebuilt after unpickling; see __getstate__
            net = build()["InferenceModelLLMmap"](self._hp, is_for_siamese=True)
            net.load_state_dict(self._net_state)
            self._net = net.eval()
        if self._net is None:
            raise RuntimeError(f"{self.name}: fit() has not been called")
        return on_device(self._net)

    def __getstate__(self) -> dict:
        """Pickle the network as a CPU state dict (its class is local to ``net.build``)."""
        state = self.__dict__.copy()
        net = state.get("_net")
        if net is not None:
            state["_net_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        state["_net"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @capped_threads
    def fit(self, A: np.ndarray, cols: np.ndarray) -> LLMmapEncoder:
        import torch

        A = self._prepare(A, cols)
        self._check_banks()
        cols = np.asarray(cols, int)
        rows = np.arange(A.shape[0]) if self.rows is None else np.asarray(self.rows, int)
        if rows.shape[0] != A.shape[0]:
            raise ValueError(
                f"{self.name}: rows selects {rows.shape[0]} models but A has {A.shape[0]}"
            )
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        device = torch_device()
        k = self.n_queries

        # 1. query strategy, training rows only
        score, cover = self._discrepancy(rows, cols)
        answerable = np.flatnonzero(cover * len(rows) >= 2)
        if answerable.size < k:
            raise ValueError(
                f"{self.name}: only {answerable.size} fitted columns were answered by two "
                f"or more training models; a trace needs {k}"
            )
        floor = self.min_coverage
        eligible = answerable[cover[answerable] >= floor]
        if eligible.size < min(self.shortlist, answerable.size):
            floor = float(
                np.sort(cover[answerable])[::-1][min(self.shortlist, answerable.size) - 1]
            )
            eligible = answerable[cover[answerable] >= floor]
            logger.warning(
                f"{self.name}: too few columns at coverage {self.min_coverage}; "
                f"relaxed the candidate floor to {floor:.3f}"
            )
        order = eligible[np.argsort(-score[eligible], kind="stable")]
        short = cols[order[: self.shortlist]]
        pool = self._greedy(rows, short)
        rest = answerable[np.argsort(-score[answerable], kind="stable")]
        rest = cols[rest][~np.isin(cols[rest], pool)]
        self.ranking = np.concatenate([pool, rest])

        # 2. traces for every training model
        dq = int(np.asarray(self.question_bank).shape[1])
        da = int(self.answer_bank.shape[2])
        P = self.trace_pool
        tokens = np.zeros((len(rows), P, dq + da), dtype=np.float32)
        n_valid = np.zeros(len(rows), dtype=int)
        for i, r in enumerate(rows):
            q = self._pool(int(r))
            n_valid[i] = len(q)
            if len(q):
                tokens[i, : len(q)] = self._tokens(int(r), q)

        # 3. Siamese training on trace positions
        perm = rng.permutation(P)
        n_val = max(round(VAL_FRACTION * P), k)
        val_pos, train_pos = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        def usable(positions: np.ndarray) -> np.ndarray:
            return np.flatnonzero((positions[None, :] < n_valid[:, None]).sum(axis=1) >= k)

        tr_models, va_models = usable(train_pos), usable(val_pos)
        if len(tr_models) < 2:
            raise ValueError(
                f"{self.name}: {len(tr_models)} training models have {k} answered queries "
                "in the training trace positions; a contrastive fit needs two"
            )
        if len(va_models) < 2:
            va_models = tr_models
            val_pos = train_pos
            logger.warning(
                f"{self.name}: too few models for validation pairs; validating on train positions"
            )
        dropped = len(rows) - len(tr_models)
        logger.info(
            f"{self.name}: {len(pool)} greedy-ordered queries, ranking {len(self.ranking)}; "
            f"{len(tr_models)}/{len(rows)} training models form {k}-query traces "
            f"({dropped} too sparse), pool {P} positions ({len(train_pos)} train / "
            f"{len(val_pos)} val), device={device}"
        )

        def sample(n: int, models: np.ndarray, positions: np.ndarray, g: np.random.Generator):
            same = g.integers(2, size=n).astype(bool)
            a = g.choice(models, size=n)
            off = g.integers(1, len(models), size=n)
            b = np.where(same, a, models[(np.searchsorted(models, a) + off) % len(models)])
            idx = np.empty((n, 2, k), dtype=int)
            for j, who in enumerate((a, b)):
                keys = g.random((n, len(positions)))
                keys[positions[None, :] >= n_valid[who][:, None]] = -1.0
                idx[:, j] = positions[np.argsort(-keys, axis=1)[:, :k]]
            return np.stack([a, b], 1), idx, same.astype(np.float32)

        def gather(m: np.ndarray, idx: np.ndarray):
            return torch.from_numpy(tokens[m[:, :, None], idx]).to(device)

        parts = build()
        hp = dict(DEFAULT_HP, emb_size=dq)
        if dq != da:
            raise ValueError(f"{self.name}: query width {dq} != answer width {da}")
        backbone = parts["InferenceModelLLMmap"](hp, is_for_siamese=True)
        self._hp = hp
        siam = parts["SiameseNetwork"](backbone).to(device)
        crit = parts["ContrastiveLoss"](margin=MARGIN)
        opt = torch.optim.Adam(siam.parameters(), lr=self.lr)
        val = sample(self.pairs_per_eval, va_models, val_pos, np.random.default_rng(self.seed + 1))

        best, best_state, bad = np.inf, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            m, idx, y = sample(self.pairs_per_epoch, tr_models, train_pos, rng)
            siam.train()
            tot = 0.0
            for s in range(0, self.pairs_per_epoch, self.batch_size):
                sl = slice(s, s + self.batch_size)
                if len(y[sl]) < 2:  # BatchNorm needs two samples in train mode
                    continue
                opt.zero_grad()
                loss = crit(siam(gather(m[sl], idx[sl]))[:, 0], torch.from_numpy(y[sl]).to(device))
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(y[sl])
            siam.eval()
            vl = acc = 0.0
            with torch.no_grad():
                vm, vidx, vy = val
                for s in range(0, len(vy), self.batch_size):
                    sl = slice(s, s + self.batch_size)
                    p = siam(gather(vm[sl], vidx[sl]))[:, 0]
                    t = torch.from_numpy(vy[sl]).to(device)
                    vl += float(crit(p, t)) * len(vy[sl])
                    acc += float(((p > 0.5).float() == t).sum())
            vl /= len(vy)
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": tot / self.pairs_per_epoch,
                    "val_loss": vl,
                    "val_acc": acc / len(vy),
                }
            )
            logger.info(f"{self.name} epoch {epoch:02d} {self.history[-1]}")
            if vl < best - 1e-6:
                best, bad, self.best_epoch = vl, 0, epoch
                best_state = {n: v.detach().clone() for n, v in siam.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            siam.load_state_dict(best_state)
        backbone.eval()
        self._net = backbone.cpu()

        # 4. signatures on the fixed probe set, LLMmap's inference setting
        self.probes = self.ranking[:k].copy()
        X = np.zeros((len(rows), LLMMAP_DIM))
        for i in range(len(rows)):
            X[i] = self._template(tokens[i, : min(n_valid[i], k)])
        empty = n_valid == 0
        if empty.any():
            X[empty] = X[~empty].mean(axis=0)
            logger.warning(
                f"{self.name}: {int(empty.sum())} training models answered no query; mean template"
            )
        self._mean = X[~empty].mean(axis=0)
        self.X = X
        return self

    # -- placing a model ---------------------------------------------------- #
    def _template(self, toks: np.ndarray) -> np.ndarray:
        """Mean signature over disjoint ``n_queries`` traces; a short remainder only if alone.

        Called with at most ``n_queries`` tokens by the arm, so this is one trace."""
        import torch

        n = len(toks)
        if n == 0:
            return np.full(LLMMAP_DIM, np.nan)
        k = self.n_queries
        chunks = [toks[s : s + k] for s in range(0, n - n % k, k)] or [toks]
        net = self._fitted()
        dev = next(net.parameters()).device
        with torch.no_grad():
            feats = [
                net.features(torch.from_numpy(np.ascontiguousarray(c[None])).to(dev))
                .cpu()
                .numpy()[0]
                for c in chunks
            ]
        return np.mean(feats, axis=0).astype(np.float64)

    def _place(self, row: int, qids: np.ndarray) -> np.ndarray:
        if len(qids) == 0:
            return self._mean.copy()
        return self._template(self._tokens(row, qids))

    def signature(self, row: int) -> np.ndarray:
        """The model's signature on the fixed probe set (next answered question on a gap)."""
        self._fitted()
        return self._place(int(row), self._pool(int(row), limit=self.n_queries))

    @capped_threads
    def fold_in(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """The fixed-probe signature; the interview is unread (mean signature if none)."""
        return self.signature(row)

    @capped_threads
    def refit(self, bits: np.ndarray, cols: np.ndarray, row: int) -> np.ndarray:
        """Same as :meth:`fold_in`: the signature does not depend on ``cols``."""
        return self.signature(row)


__all__ = ["LLMMAP_DIM", "N_QUERIES", "LLMmapEncoder"]
