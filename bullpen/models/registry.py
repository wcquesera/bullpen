"""The encoder registry: every arm the paper reports, by name.

:func:`build_encoders` returns fresh, unfitted encoders for one model split. An arm
whose input block is missing (no answer bank, no question-text table, ...) is left
out with a warning rather than substituted, and a requested name that was not built
is reported by name.

Arm naming: a ``__interview`` / ``__itext`` suffix is the -budget twin, fitted on the
K interview questions only (``__itext`` also pools the answer text over them); a
``_model_shuffled`` twin reads another model's answers and a ``_question_shuffled``
twin reads another question's embedding (the shuffled-text controls);
``null_random{d}`` is the width-d Gaussian floor.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger

from bullpen.competitors.embedllm.textmf import FAITHFUL_DIM, TextMFFaithfulEncoder
from bullpen.competitors.fixed_table import FixedEncoder, LeaderboardEncoder
from bullpen.competitors.irt import IRTEncoder
from bullpen.competitors.irtfamily import (
    IRTNET_DIM,
    IRTROUTER_DIM,
    JEIRT_PAPER_DIM,
    IrtNetEncoder,
    IRTRouterEncoder,
    JEIRTEncoder,
)
from bullpen.competitors.llmdna import LLMDNA_DIM, LLMDNAEncoder
from bullpen.competitors.llmmap import LLMMAP_DIM, LLMmapOrigEncoder
from bullpen.competitors.locus import LOCUS_DIM, LocusEncoder
from bullpen.competitors.routellm import RouteLLMMFEncoder
from bullpen.competitors.routellm.encoder import MF_PUBLISHED_DIM
from bullpen.models.base import Encoder
from bullpen.models.pls_svd import CHANNEL_NAMES, ChannelPCAEncoder, PLSSVDFusionEncoder

#: ``factory(n_models=..., rows=..., seed=..., dim=...) -> Encoder``
EncoderFactory = Callable[..., Encoder]

#: The run width the ``null_random32`` floor is drawn at.
DEFAULT_DIM: int = 32

#: Width of the mean answer embedding (BGE-base), and of its floor.
MODEL_TEXT_DIM: int = 768

#: Width of the unidimensional 2PL.
IRT_2PL_DIM: int = 1

#: Offset applied to the seed of a Gaussian floor's table.
NULL_SEED_OFFSET: int = 10_000

INTERVIEW_SUFFIX: str = "__interview"
QUESTION_SHUFFLED_SUFFIX: str = "_question_shuffled"

#: The question-text IRT competitors at their published widths.
IRT_FAMILY_DIMS: dict[str, int] = {
    "comp_irtrouter": IRTROUTER_DIM,
    "comp_jeirt": JEIRT_PAPER_DIM,
    "comp_irtnet": IRTNET_DIM,
}
IRT_FAMILY_ARMS: tuple[str, ...] = tuple(IRT_FAMILY_DIMS)
IRT_FAMILY_CLASSES: dict[str, type] = {
    "comp_irtrouter": IRTRouterEncoder,
    "comp_jeirt": JEIRTEncoder,
    "comp_irtnet": IrtNetEncoder,
}

#: Channel-group arms: ``name -> (encoder class, channels)``. PCA32-bits is a z-scored
#: 32-d PCA of the bits; BULLPEN-bits+q and BULLPEN-fused are K-block PLS-SVD fits.
CHANNEL_GROUP_ARMS: dict[str, tuple[type, tuple[str, ...]]] = {
    "pca_bits32": (ChannelPCAEncoder, ("bits",)),
    "pls_svd_bitsqtext32": (PLSSVDFusionEncoder, ("bits", "qtext")),
    "pls_svd_tri32": (PLSSVDFusionEncoder, CHANNEL_NAMES),
}
#: Channel-group arms with a question-shuffled control twin.
QUESTION_SHUFFLED_BASES: tuple[str, ...] = ("pls_svd_bitsqtext32", "pls_svd_tri32")


def matched_null(d: int) -> str:
    """Name of the width-``d`` Gaussian floor."""
    return f"null_random{int(d)}"


#: Widths of the Gaussian floors besides the run's own (``null_random32``).
FLOOR_WIDTHS: tuple[int, ...] = (
    FAITHFUL_DIM,  # 232: comp_embedllm and IrtNet
    LLMMAP_DIM,  # 384
    LOCUS_DIM,  # 128: LOCUS, RouteLLM-MF, LLM DNA
    MODEL_TEXT_DIM,  # 768: text_mean768
    IRTROUTER_DIM,  # 25
    JEIRT_PAPER_DIM,  # 256
)

#: Every arm, in fitting order.
ENCODER_ORDER: list[str] = [
    "pca_bits32",
    "pca_bits32__interview",
    "pls_svd_bitsqtext32",
    "pls_svd_bitsqtext32__interview",
    "comp_embedllm",
    "bits_irt2pl",
    *IRT_FAMILY_ARMS,
    "pls_svd_fusion32",
    "pls_svd_fusion32_model_shuffled",
    "pls_svd_tri32",
    "pls_svd_tri32_model_shuffled",
    "comp_routellm_mf",
    "text_mean768__itext",
    "pls_svd_fusion32__itext",
    "pls_svd_tri32__itext",
    "text_mean768",
    "text_mean768_model_shuffled",
    "pls_svd_bitsqtext32_question_shuffled",
    "pls_svd_tri32_question_shuffled",
    "comp_locus",
    "comp_llmmap_orig",
    "comp_llmdna",
    "null_leaderboard",
    "null_random32",
    *(matched_null(d) for d in FLOOR_WIDTHS),
]

#: Budget twins that read the answer text pooled over the K interview questions.
INTERVIEW_TEXT_ARMS: tuple[str, ...] = (
    "text_mean768__itext",
    "pls_svd_fusion32__itext",
    "pls_svd_tri32__itext",
)

#: Arms fitted on the K interview instead of the pool.
INTERVIEW_ARMS: frozenset[str] = frozenset(
    {"pca_bits32__interview", "pls_svd_bitsqtext32__interview", *INTERVIEW_TEXT_ARMS}
)

INTERVIEW_BLOCK: str = "interview"

#: How an arm's width is decided: ``RUN_DIM`` (the run's ``dim``), ``INPUT_DIM`` (the
#: input table's width) or a fixed published width.
RUN_DIM: str = "run"
INPUT_DIM: str = "input"
ARM_DIM: dict[str, int | str] = dict.fromkeys(ENCODER_ORDER, RUN_DIM) | {
    "comp_embedllm": FAITHFUL_DIM,
    "bits_irt2pl": IRT_2PL_DIM,
    "null_leaderboard": 1,
    "text_mean768": INPUT_DIM,
    "text_mean768_model_shuffled": INPUT_DIM,
    "text_mean768__itext": INPUT_DIM,
    "comp_locus": LOCUS_DIM,
    "comp_routellm_mf": MF_PUBLISHED_DIM,
    "comp_llmmap_orig": LLMMAP_DIM,
    "comp_llmdna": LLMDNA_DIM,
    **IRT_FAMILY_DIMS,
    **{matched_null(d): d for d in FLOOR_WIDTHS},
}


def arm_dim(name: str) -> int | str:
    """The width policy for ``name``: :data:`RUN_DIM`, :data:`INPUT_DIM`, or a fixed int."""
    return ARM_DIM.get(name, RUN_DIM)


def arm_fit_block(name: str) -> str:
    """The question block ``name`` is fitted on: ``pool`` or ``interview``."""
    return INTERVIEW_BLOCK if name in INTERVIEW_ARMS else "pool"


def arm_blocks(name: str) -> tuple[str, ...]:
    """Every question block one arm consumes, fitting block first."""
    return (arm_fit_block(name),)


def artifact_file(directory: Path | str, arm: str, seed: int) -> Path:
    """``<directory>/<arm>__s<seed>.pkl``."""
    return Path(directory) / f"{arm}__s{seed}.pkl"


DERANGEMENT_TRIES: int = 1000


def deranged_models(n_models: int, train_rows: np.ndarray, seed: int = 0) -> np.ndarray:
    """A derangement of the model axis, closed on each side of the split.

    Row ``r`` reads row ``perm[r] != r``; training rows are permuted among training
    rows and the rest among the rest. Each block is redrawn until it has no fixed
    point, so the draw is uniform over derangements and deterministic in ``seed``.
    """
    if n_models < 1:
        raise ValueError(f"n_models must be at least 1, got {n_models}")
    train = np.unique(np.asarray(train_rows, dtype=int))
    if train.size and (train[0] < 0 or train[-1] >= n_models):
        raise ValueError(f"train_rows must lie in [0, {n_models}), got {train[0]}..{train[-1]}")
    rng = np.random.default_rng(seed)
    perm = np.arange(n_models)
    rest = np.setdiff1d(perm, train)
    for label, block in (("train", train), ("non-train", rest)):
        if block.size == 1:
            raise ValueError(
                f"the {label} block holds one model (row {int(block[0])}); no derangement exists"
            )
        if block.size == 0:
            continue
        for _ in range(DERANGEMENT_TRIES):
            draw = rng.permutation(block)
            if not (draw == block).any():
                break
        else:  # pragma: no cover - probability ~e^-275
            raise RuntimeError(f"no derangement of {block.size} rows in {DERANGEMENT_TRIES} draws")
        perm[block] = draw
    return perm


def deranged_questions(n_questions: int, seed: int) -> np.ndarray:
    """A derangement of the question axis: question ``q`` reads row ``perm[q] != q``."""
    return deranged_models(n_questions, np.arange(n_questions), seed)


def _gaussian_floor(name: str, width: int, n_models: int, rows, seed: int, offset: int):
    table = np.random.default_rng(seed + NULL_SEED_OFFSET + offset).normal(size=(n_models, width))
    return FixedEncoder(name, dim=width, seed=seed, table=table, rows=rows)


def _derange(text_table: np.ndarray, rows: np.ndarray | None, seed: int, name: str):
    try:
        return deranged_models(
            text_table.shape[0], np.arange(text_table.shape[0]) if rows is None else rows, seed
        )
    except ValueError as exc:
        logger.warning(f"skipping {name}: {exc}")
        return None


def _channel_group_arms(
    *,
    seed: int,
    dim: int,
    rows: np.ndarray | None,
    subset: list[str] | None,
    text_table: np.ndarray | None,
    q_text_table: np.ndarray | None,
    interview_text_table: np.ndarray | None,
) -> dict[str, Encoder]:
    """Every :data:`CHANNEL_GROUP_ARMS` arm with its twins, when its input blocks exist."""
    out: dict[str, Encoder] = {}
    for name, (cls, channels) in CHANNEL_GROUP_ARMS.items():
        reads_text = "anstext" in channels
        shuffled = f"{name}_model_shuffled" if reads_text else None
        budget = f"{name}__itext" if reads_text else f"{name}{INTERVIEW_SUFFIX}"
        q_twin = f"{name}{QUESTION_SHUFFLED_SUFFIX}" if name in QUESTION_SHUFFLED_BASES else None
        family = [n for n in (name, shuffled, budget, q_twin) if n is not None]
        wanted = [n for n in family if subset is None or n in subset]
        missing = [
            block
            for block, need, have in (
                ("question text bridge", "qtext" in channels, q_text_table is not None),
                ("pooled answer-text block", reads_text, text_table is not None),
            )
            if need and not have
        ]
        if missing:
            if wanted:
                logger.warning(f"{' and '.join(missing)} unavailable — skipping {wanted}")
            continue
        kw = dict(
            dim=dim,
            seed=seed,
            rows=rows,
            channels=channels,
            q_text_emb=None if q_text_table is None else np.asarray(q_text_table),
        )
        out[name] = cls(name, text_table=text_table if reads_text else None, **kw)
        if shuffled is not None and (subset is None or shuffled in subset):
            derange = _derange(text_table, rows, seed, shuffled)
            if derange is not None:
                out[shuffled] = cls(shuffled, text_table=text_table[derange], **kw)
        if not reads_text:
            out[budget] = cls(budget, **kw)
        elif interview_text_table is not None:
            out[budget] = cls(budget, text_table=np.asarray(interview_text_table), **kw)
        if q_twin is not None and (subset is None or q_twin in subset):
            # a row permutation, not a permuted table: q_text_emb is re-attached from
            # the slice on load, which would undo a permuted copy
            out[q_twin] = cls(
                q_twin,
                text_table=text_table if reads_text else None,
                q_row_perm=deranged_questions(len(q_text_table), seed),
                **kw,
            )
    return out


def build_encoders(
    n_models: int,
    text_table: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    seed: int = 0,
    dim: int = DEFAULT_DIM,
    subset: list[str] | None = None,
    q_text_table: np.ndarray | None = None,
    q_text_table_mpnet: np.ndarray | None = None,
    answer_table: np.ndarray | None = None,
    answer_mask: np.ndarray | None = None,
    interview_text_table: np.ndarray | None = None,
    llmmap_probe_bank: dict[str, np.ndarray] | None = None,
) -> dict[str, Encoder]:
    """Fresh (unfitted) encoders for one split, in :data:`ENCODER_ORDER`.

    ``rows`` are the training rows of the full model axis; ``text_table`` is the
    ``[M_all, d]`` mean answer embedding pooled over the pool block and
    ``interview_text_table`` the same pooled over the K interview; ``q_text_table``
    is the ``[Q_all, d]`` BGE question embedding and ``q_text_table_mpnet`` the
    all-mpnet one the EmbedLLM, LOCUS and IRT-family competitors read;
    ``answer_table`` / ``answer_mask`` are the ``[M_all, Q_all, d]`` answer bank and
    its coverage; ``llmmap_probe_bank`` holds the answers to LLMmap's own probes.
    """
    if n_models < 1:
        raise ValueError(f"n_models must be at least 1, got {n_models}")
    rand_table = np.random.default_rng(seed + NULL_SEED_OFFSET).normal(size=(n_models, dim))
    enc: dict[str, Encoder] = {
        "bits_irt2pl": IRTEncoder("bits_irt2pl", dim=IRT_2PL_DIM, seed=seed),
        "null_leaderboard": LeaderboardEncoder("null_leaderboard", seed=seed),
        "null_random32": FixedEncoder(
            "null_random32", dim=dim, seed=seed, table=rand_table, rows=rows
        ),
        **{
            matched_null(w): _gaussian_floor(matched_null(w), w, n_models, rows, seed, w)
            for w in FLOOR_WIDTHS
        },
    }
    if q_text_table is not None:
        enc["comp_routellm_mf"] = RouteLLMMFEncoder(
            "comp_routellm_mf", dim=MF_PUBLISHED_DIM, seed=seed, q_text_emb=np.asarray(q_text_table)
        )
    elif subset is None or "comp_routellm_mf" in subset:
        logger.warning("question text table unavailable — skipping comp_routellm_mf")
    if q_text_table_mpnet is not None:
        mpnet = np.asarray(q_text_table_mpnet)
        enc["comp_embedllm"] = TextMFFaithfulEncoder("comp_embedllm", seed=seed, q_text_emb=mpnet)
        enc["comp_locus"] = LocusEncoder("comp_locus", dim=LOCUS_DIM, seed=seed, q_text_emb=mpnet)
        for name, cls in IRT_FAMILY_CLASSES.items():
            enc[name] = cls(name, dim=IRT_FAMILY_DIMS[name], seed=seed, q_text_emb=mpnet)
    elif subset is None or {"comp_embedllm", "comp_locus", *IRT_FAMILY_ARMS} & set(subset):
        logger.warning(
            "all-mpnet-base-v2 question table (data/question_text_mpnet.npz) unavailable — "
            "skipping comp_embedllm, comp_locus and the IRT-family competitors"
        )
    if answer_table is not None and q_text_table is not None and llmmap_probe_bank is not None:
        enc["comp_llmmap_orig"] = LLMmapOrigEncoder(
            "comp_llmmap_orig",
            dim=LLMMAP_DIM,
            seed=seed,
            answer_bank=answer_table,
            question_bank=np.asarray(q_text_table),
            rows=rows,
            probe_answers=llmmap_probe_bank["E"],
            probe_mask=llmmap_probe_bank.get("mask"),
            probe_questions=llmmap_probe_bank["Q"],
        )
    elif subset is None or "comp_llmmap_orig" in subset:
        logger.warning(
            "answer bank, question table or LLMmap probe bank unavailable — "
            "skipping comp_llmmap_orig"
        )
    if answer_table is not None:
        enc["comp_llmdna"] = LLMDNAEncoder(
            "comp_llmdna",
            dim=LLMDNA_DIM,
            seed=seed,
            answer_bank=answer_table,
            answer_mask=answer_mask,
            rows=rows,
        )
    elif subset is None or "comp_llmdna" in subset:
        logger.warning("answer bank unavailable — skipping comp_llmdna")

    text_arms = (
        "text_mean768",
        "text_mean768_model_shuffled",
        "pls_svd_fusion32",
        "pls_svd_fusion32_model_shuffled",
    )
    text_arms_wanted = subset is None or bool(set(text_arms) & set(subset))
    if text_table is not None:
        text_table = np.asarray(text_table)
        if not np.isfinite(text_table).all():
            if text_arms_wanted:
                n_blank = int((~np.isfinite(text_table).all(axis=1)).sum())
                logger.warning(
                    f"model text block has no text for {n_blank} models — "
                    f"skipping the answer-text arms"
                )
            text_table = None
    elif text_arms_wanted:
        logger.warning("model text block unavailable — skipping the answer-text arms")
    if text_table is not None:
        enc["text_mean768"] = FixedEncoder(
            "text_mean768", dim=text_table.shape[1], seed=seed, table=text_table, rows=rows
        )
        if subset is None or "text_mean768_model_shuffled" in subset:
            derange = _derange(text_table, rows, seed, "text_mean768_model_shuffled")
            if derange is not None:
                enc["text_mean768_model_shuffled"] = FixedEncoder(
                    "text_mean768_model_shuffled",
                    dim=text_table.shape[1],
                    seed=seed,
                    table=text_table[derange],
                    rows=rows,
                )
        enc["pls_svd_fusion32"] = PLSSVDFusionEncoder(
            "pls_svd_fusion32", dim=dim, seed=seed, text_table=text_table, rows=rows
        )
        if subset is None or "pls_svd_fusion32_model_shuffled" in subset:
            derange = _derange(text_table, rows, seed, "pls_svd_fusion32_model_shuffled")
            if derange is not None:
                enc["pls_svd_fusion32_model_shuffled"] = PLSSVDFusionEncoder(
                    "pls_svd_fusion32_model_shuffled",
                    dim=dim,
                    seed=seed,
                    text_table=text_table[derange],
                    rows=rows,
                )
    if interview_text_table is not None:
        itext = np.asarray(interview_text_table)
        if not np.isfinite(itext).all():
            raise ValueError("interview text block carries NaN")
        enc["text_mean768__itext"] = FixedEncoder(
            "text_mean768__itext", dim=itext.shape[1], seed=seed, table=itext, rows=rows
        )
        enc["pls_svd_fusion32__itext"] = PLSSVDFusionEncoder(
            "pls_svd_fusion32__itext", dim=dim, seed=seed, text_table=itext, rows=rows
        )
    elif subset is None or set(INTERVIEW_TEXT_ARMS) & set(subset):
        logger.warning(f"interview text block unavailable — skipping {list(INTERVIEW_TEXT_ARMS)}")
    enc.update(
        _channel_group_arms(
            seed=seed,
            dim=dim,
            rows=rows,
            subset=subset,
            text_table=text_table,
            q_text_table=q_text_table,
            interview_text_table=interview_text_table,
        )
    )
    keep = ENCODER_ORDER if subset is None else [k for k in ENCODER_ORDER if k in subset]
    built = {k: enc[k] for k in keep if k in enc}
    if subset is not None:
        absent = [k for k in subset if k not in built]
        if absent:
            logger.warning(f"requested encoder(s) not built and NOT in the results: {absent}")
    return built
