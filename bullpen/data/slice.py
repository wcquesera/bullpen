"""The slice: correctness bits, model and question metadata, and the answer-embedding bank.

``slice.npz`` holds

    A            [M, Q] float32   correctness as float
    R            [M, Q] uint8     correctness bit of the latest response
    bench        [Q]    int32     benchmark index per question
    bench_names  [B]    str       benchmark name per index
    question_ids [Q]    str       question ids
    model_ids    [M]    str       model names
    vram_gb      [M]    float32   served VRAM for the knapsack task (NaN if unknown)
    release_date [M]    str       ISO date or ""
    split        [M]    str       "train" / "test" / ""

and, on a merged slice (fullset + extset models),

    covered      [M, Q] bool      True where the bit was collected
    in_fullset   [M]    bool      collected on the whole probe
    in_extset    [M]    bool      external-board model
    core         [Q]    bool      the 1,181-question core every model answered

A merged slice holds NaN where an extset model was never asked a question; it is read
through one of the finite :data:`VIEWS` (:meth:`Slice.view`). ``answers.npz`` holds

    Ae      [M, Q, Da] float16   sentence-encoder embedding of each answer
    Ae_mask [M, Q]     bool      True where that embedding is a real answer
    Qe      [Q, Dq]    float32   embedding of each question (optional)

or ships packed (:func:`save_packed_answers`, ``process.py pack``) as int8 parts
``answers.partNN.npz`` with a per-cell scale; :func:`load_slice` reads either form.
"""

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.lib.format import read_magic

from bullpen.config import CONFIG_DIR, REPO_ROOT

DATA_DIR = REPO_ROOT / "data"

DEFAULT_MANIFEST = CONFIG_DIR / "substrate.json"
DEFAULT_SLICE = DATA_DIR / "slice.npz"
DEFAULT_ANSWERS = DATA_DIR / "answers.npz"

#: Arrays ``slice.npz`` must carry.
REQUIRED_KEYS: tuple[str, ...] = ("A", "R", "bench", "bench_names", "model_ids", "question_ids")

#: On-disk dtype of the answer bank, and the dtype it is computed in.
ANSWER_STORAGE_DTYPE = np.float16
ANSWER_COMPUTE_DTYPE = np.float32

#: Size cap of one part of a packed bank, in MB of int8 ``Ae`` before compression
#: (GitHub refuses files over 100 MB).
PACKED_PART_MB = 90.0

#: In-memory dtype of the correctness matrices.
CORRECTNESS_DTYPE = np.float64

#: Coverage below which a model row is flagged as under-measured.
COVERAGE_FLAG_THRESHOLD = 0.8

#: Fraction of the question axis a model must have answered to enter the slice; its
#: remaining unanswered cells are written as incorrect.
COVERAGE_FLOOR = 0.90

#: Accuracy at or below which a model is treated as broken (refusal loop, template mismatch).
DEGENERATE_ACCURACY = 0.01

#: The finite views of a merged slice: ``core`` (every model, the core questions) and
#: ``full`` (fullset models, every question).
VIEWS: tuple[str, ...] = ("core", "full")


@dataclass
class Slice:
    """One loaded slice."""

    A: np.ndarray  # [M, Q] accuracies
    R: np.ndarray  # [M, Q] bits
    bench: np.ndarray  # [Q] benchmark index
    bench_names: list[str]
    model_ids: list[str]
    question_ids: list[str]
    vram_gb: np.ndarray | None = None  # [M], NaN where unknown
    release_date: list[str] = field(default_factory=list)  # [M] ISO date or ""
    #: per-model split label — "train" / "test", or "" where unassigned.
    split: list[str] = field(default_factory=list)
    Ae: np.ndarray | None = None  # [M, Q, Da] answer embeddings
    #: [M, Q] bool — True where ``Ae`` carries an embedded answer (``None``: unrecorded)
    Ae_mask: np.ndarray | None = None
    #: [M, Q] bool — True where the cell's bit was collected (uncollected cells are 0 in ``R``)
    covered: np.ndarray | None = None
    Qe: np.ndarray | None = None  # [Q, Dq] question embeddings
    #: [M] bool set membership, carried only by a merged slice
    in_fullset: np.ndarray | None = None
    in_extset: np.ndarray | None = None
    #: [Q] bool — the core question block every model of a merged slice answered.
    core: np.ndarray | None = None
    #: where this cut's label tables live (:func:`bullpen.data.labels.labels_root_for`)
    labels_root: Path | None = None
    #: [Q] int — on a view, each column's index on the parent axis (for the question blocks)
    columns: np.ndarray | None = None
    manifest: dict = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return self.A.shape[0]

    @property
    def n_questions(self) -> int:
        return self.A.shape[1]

    @property
    def n_benchmarks(self) -> int:
        return len(self.bench_names)

    @property
    def has_gaps(self) -> bool:
        """True where ``A`` carries NaN: a merged slice that must be read through a view."""
        return bool(np.isnan(self.A).any())

    @property
    def has_text(self) -> bool:
        """Both banks present — the condition the text arms actually need."""
        return self.Ae is not None and self.Qe is not None

    def _text_summary(self) -> str:
        """How much text this slice carries, for :meth:`summary`."""
        if not self.has_text:
            return "no"
        if self.Ae_mask is None:
            return "yes (coverage unrecorded)"
        return f"yes ({float(self.Ae_mask.mean()):.1%} of cells)"

    # ----------------------------------------------------------------- #
    # row / column views
    # ----------------------------------------------------------------- #
    def subset(
        self, models: np.ndarray | None = None, questions: np.ndarray | None = None
    ) -> Slice:
        """A view restricted to some models and/or questions (benchmark indices recompacted)."""
        m = np.arange(self.n_models) if models is None else np.asarray(models, dtype=int)
        q = np.arange(self.n_questions) if questions is None else np.asarray(questions, dtype=int)
        keep_b = sorted(set(self.bench[q].tolist()))
        remap = {b: i for i, b in enumerate(keep_b)}
        return Slice(
            A=self.A[np.ix_(m, q)],
            R=self.R[np.ix_(m, q)],
            bench=np.array([remap[b] for b in self.bench[q]], dtype=np.int32),
            bench_names=[self.bench_names[b] for b in keep_b],
            model_ids=[self.model_ids[i] for i in m],
            question_ids=[self.question_ids[i] for i in q],
            vram_gb=None if self.vram_gb is None else self.vram_gb[m],
            release_date=[self.release_date[i] for i in m] if self.release_date else [],
            split=[self.split[i] for i in m] if self.split else [],
            covered=None if self.covered is None else self.covered[np.ix_(m, q)],
            Ae=None if self.Ae is None else np.asarray(self.Ae)[np.ix_(m, q)],
            Ae_mask=None if self.Ae_mask is None else self.Ae_mask[np.ix_(m, q)],
            Qe=None if self.Qe is None else self.Qe[q],
            manifest=self.manifest,
            in_fullset=None if self.in_fullset is None else self.in_fullset[m],
            in_extset=None if self.in_extset is None else self.in_extset[m],
            core=None if self.core is None else self.core[q],
            labels_root=self.labels_root,
            columns=None if self.columns is None else self.columns[q],
        )

    def view(self, name: str) -> Slice:
        """One of the finite :data:`VIEWS` of a merged slice.

        ``core`` writes the few core cells an extset model never answered as incorrect.
        """
        if name not in VIEWS:
            raise ValueError(f"unknown view {name!r}; expected one of {VIEWS}")
        if self.in_fullset is None or self.core is None:
            raise ValueError(
                "view() needs a merged slice (in_fullset / core); this one is a single-set "
                "slice and is already finite"
            )
        if name == "full":
            rows = np.flatnonzero(self.in_fullset)
            if rows.size == 0 or not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
                raise ValueError("fullset rows are not one contiguous block; rebuild with merge")
            sl = self._rows(slice(int(rows[0]), int(rows[-1]) + 1))
        else:
            cols = np.flatnonzero(self.core)
            sl = self.subset(questions=cols)
            sl.columns = cols if self.columns is None else self.columns[cols]
            gap = np.isnan(sl.A)
            if gap.any():
                sl.A = np.where(gap, 0.0, sl.A)
                sl.R = np.where(gap, 0, sl.R).astype(sl.R.dtype)
        if sl.has_gaps:
            raise ValueError(f"view {name!r} still has uncollected cells — the merge gate failed")
        sl.manifest = {**self.manifest, "view": name}
        return sl

    def _rows(self, rows: slice) -> Slice:
        """A row block by basic slicing, so memory-mapped banks stay mapped."""
        return Slice(
            A=self.A[rows],
            R=self.R[rows],
            bench=self.bench,
            bench_names=self.bench_names,
            model_ids=self.model_ids[rows],
            question_ids=self.question_ids,
            vram_gb=None if self.vram_gb is None else self.vram_gb[rows],
            release_date=self.release_date[rows],
            split=self.split[rows],
            covered=None if self.covered is None else self.covered[rows],
            Ae=None if self.Ae is None else self.Ae[rows],
            Ae_mask=None if self.Ae_mask is None else self.Ae_mask[rows],
            Qe=self.Qe,
            manifest=self.manifest,
            in_fullset=self.in_fullset[rows],
            in_extset=None if self.in_extset is None else self.in_extset[rows],
            core=self.core,
            labels_root=self.labels_root,
        )

    def summary(self) -> str:
        cov = float(np.isfinite(self.A).mean()) if self.A.size else 0.0
        sets = (
            f"; fullset {int(self.in_fullset.sum())} / extset {int(self.in_extset.sum())}"
            f" / core {int(self.core.sum())} questions"
            if self.in_fullset is not None and self.in_extset is not None and self.core is not None
            else ""
        )
        split = (
            f"; split {sum(s == 'train' for s in self.split)}/"
            f"{sum(s == 'test' for s in self.split)}"
            if self.split
            else "; no frozen split"
        )
        return (
            f"{self.n_models} models x {self.n_questions} questions across "
            f"{self.n_benchmarks} benchmarks; "
            f"text={self._text_summary()}; "
            f"coverage {cov:.1%}{split}{sets}"
        )


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #
#: Bytes of a zip local file header before its two length fields.
_LOCAL_HEADER = 30


def _map_member(path: Path, name: str) -> np.ndarray | None:
    """One uncompressed ``.npz`` member as a memory map, or ``None`` if it is compressed.

    ``np.load(..., mmap_mode="r")`` does not map ``.npz`` members, so the member's
    ``.npy`` is mapped at its offset inside the archive.
    """
    with zipfile.ZipFile(path) as z:
        member = z.getinfo(name)
        if member.compress_type != zipfile.ZIP_STORED:
            return None
        with z.open(member) as fh:
            read_header = getattr(np.lib.format, "read_array_header_{}_{}".format(*read_magic(fh)))
            shape, fortran, dtype = read_header(fh)
            npy_header = fh.tell()
    with path.open("rb") as fh:
        fh.seek(member.header_offset)
        name_len, extra_len = struct.unpack("<HH", fh.read(_LOCAL_HEADER)[26:])
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        shape=shape,
        order="F" if fortran else "C",
        offset=member.header_offset + _LOCAL_HEADER + name_len + extra_len + npy_header,
    )


def packed_parts(answers_path: Path | str) -> list[Path]:
    """The parts of a packed bank (``<stem>.partNN.npz``) beside ``answers_path``, in order."""
    path = Path(answers_path)
    return sorted(path.parent.glob(f"{path.stem}.part[0-9][0-9].npz"))


def bank_exists(answers_path: Path | str) -> bool:
    """Whether an answer bank is present at ``answers_path``, whole or packed."""
    return Path(answers_path).is_file() or bool(packed_parts(answers_path))


def save_packed_answers(
    Ae: np.ndarray,
    Ae_mask: np.ndarray,
    Qe: np.ndarray | None,
    answers_path: Path | str,
    part_mb: float = PACKED_PART_MB,
) -> list[Path]:
    """Write a bank as int8 parts of at most ``part_mb`` MB of ``Ae`` each.

    Each cell's vector is scaled by its own absolute maximum over 127, so the
    rounding error is relative to the cell (cosine to the fp16 original > 0.999
    on BGE answers). Part 0 also carries ``Qe``, kept in float32.
    """
    path = Path(answers_path)
    for old in packed_parts(path):
        old.unlink()
    n_models, n_q, dim = Ae.shape
    per_part = max(1, int(part_mb * 1e6 // (n_q * dim)))
    written = []
    for k, start in enumerate(range(0, n_models, per_part)):
        block = np.asarray(Ae[start : start + per_part], dtype=np.float32)
        amax = np.abs(block).max(axis=-1)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        payload = {
            "Ae_q": np.rint(block / scale[..., None]).astype(np.int8),
            "Ae_scale": scale,
            "Ae_mask": np.asarray(Ae_mask[start : start + per_part], dtype=bool),
            "row_start": np.int64(start),
            "n_models": np.int64(n_models),
        }
        if k == 0 and Qe is not None:
            payload["Qe"] = np.asarray(Qe, dtype=ANSWER_COMPUTE_DTYPE)
        out = path.with_name(f"{path.stem}.part{k:02d}.npz")
        np.savez_compressed(out, **payload)
        written.append(out)
        logger.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return written


def _load_packed(parts: list[Path]) -> dict[str, np.ndarray]:
    """``{Ae (fp16), Ae_mask, Qe?}`` reassembled from the parts of a packed bank."""
    blocks, masks, Qe, rows = [], [], None, 0
    for part in parts:
        with np.load(part, allow_pickle=False) as z:
            if int(z["row_start"]) != rows:
                start = int(z["row_start"])
                raise ValueError(f"{part.name} starts at row {start}, expected {rows}")
            q = (z["Ae_q"] * z["Ae_scale"][..., None]).astype(ANSWER_STORAGE_DTYPE)
            blocks.append(q)
            masks.append(z["Ae_mask"])
            rows += q.shape[0]
            total = int(z["n_models"])
            if "Qe" in z.files:
                Qe = z["Qe"]
    if rows != total:
        raise ValueError(f"packed bank {parts[0].name}: {rows} of {total} model rows present")
    out = {"Ae": np.concatenate(blocks), "Ae_mask": np.concatenate(masks)}
    if Qe is not None:
        out["Qe"] = Qe
    return out


def _load_answers(
    answers_path: Path, shape: tuple[int, int], mmap: bool
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """``(Ae, Ae_mask, Qe)`` from ``answers.npz`` or its packed parts, checked against
    the slice's shape."""
    packed = not answers_path.is_file()
    if packed:
        za = _load_packed(packed_parts(answers_path))
        files = list(za)
    else:
        za = np.load(answers_path, allow_pickle=False)
        files = za.files
    if "Ae" not in files:
        raise ValueError(f"{answers_path.name} carries no 'Ae' — it is not an answer bank")
    Ae = _map_member(answers_path, "Ae.npy") if mmap and not packed else None
    if Ae is None:
        Ae = za["Ae"] if mmap else np.asarray(za["Ae"], dtype=ANSWER_COMPUTE_DTYPE)
    if tuple(Ae.shape[:2]) != shape:
        raise ValueError(
            f"answer bank {tuple(Ae.shape[:2])} does not match slice "
            f"{shape} — the two files are from different cuts"
        )
    # a bank without a mask is warned about, not read as fully covered
    Ae_mask = None
    if "Ae_mask" in files:
        Ae_mask = np.asarray(za["Ae_mask"], dtype=bool)
        if Ae_mask.shape != shape:
            raise ValueError(
                f"answer mask {Ae_mask.shape} does not match slice {shape} — "
                "the mask and the bank are from different cuts"
            )
    else:
        logger.warning(
            f"{answers_path.name} carries no 'Ae_mask'; how much of the bank is real is unrecorded"
        )

    # Qe is optional
    Qe = np.asarray(za["Qe"], dtype=ANSWER_COMPUTE_DTYPE) if "Qe" in files else None
    if Qe is not None and Qe.shape[0] != shape[1]:
        raise ValueError(
            f"question embeddings cover {Qe.shape[0]} questions and the slice "
            f"has {shape[1]} — the two files are from different cuts"
        )
    return Ae, Ae_mask, Qe


def core_answers_path(answers_path: Path | str) -> Path:
    """``answers.core.npz`` beside ``answers.npz``: the bank on the core columns only."""
    path = Path(answers_path)
    return path.with_name(f"{path.stem}.core{path.suffix}")


def load_slice(
    slice_path: Path | str = DEFAULT_SLICE,
    answers_path: Path | str | None = DEFAULT_ANSWERS,
    manifest_path: Path | str | None = DEFAULT_MANIFEST,
    require_text: bool = False,
    mmap: bool = True,
    view: str | None = None,
    allow_gaps: bool = False,
) -> Slice:
    """Load a slice and, when present, its answer bank.

    ``mmap=True`` leaves ``Ae`` a float16 memory map; ``mmap=False`` reads it as
    float32. ``view`` names one of :data:`VIEWS` and is required for a merged slice
    with uncollected cells unless ``allow_gaps=True``.
    """
    from bullpen.data.labels import labels_root_for  # labels imports this module

    slice_path = Path(slice_path)
    if not slice_path.exists():
        raise FileNotFoundError(f"no slice at {slice_path}")
    z = np.load(slice_path, allow_pickle=False)
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"{slice_path.name} is missing required array(s): {missing}")

    A = z["A"].astype(CORRECTNESS_DTYPE)
    R = z["R"].astype(CORRECTNESS_DTYPE)
    if R.shape != A.shape:
        raise ValueError(f"A is {A.shape} and R is {R.shape} — they must be the same cut")

    Ae = Ae_mask = Qe = None
    core_bank = None
    if view == "core" and answers_path is not None:
        core_bank = core_answers_path(answers_path)
        core_bank = core_bank if bank_exists(core_bank) else None
    if core_bank is None and answers_path is not None and bank_exists(answers_path):
        Ae, Ae_mask, Qe = _load_answers(Path(answers_path), A.shape, mmap)
    elif require_text:
        raise FileNotFoundError(f"answer embeddings required but not found at {answers_path}")

    manifest = {}
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())

    sl = Slice(
        A=A,
        R=R,
        bench=z["bench"].astype(np.int32),
        bench_names=[str(x) for x in z["bench_names"]],
        model_ids=[str(x) for x in z["model_ids"]],
        question_ids=[str(x) for x in z["question_ids"]],
        vram_gb=z["vram_gb"].astype(np.float64) if "vram_gb" in z.files else None,
        release_date=[str(x) for x in z["release_date"]] if "release_date" in z.files else [],
        split=[str(x) for x in z["split"]] if "split" in z.files else [],
        covered=z["covered"].astype(bool) if "covered" in z.files else None,
        Ae=Ae,
        Ae_mask=Ae_mask,
        Qe=Qe,
        manifest=manifest,
        in_fullset=z["in_fullset"].astype(bool) if "in_fullset" in z.files else None,
        in_extset=z["in_extset"].astype(bool) if "in_extset" in z.files else None,
        core=z["core"].astype(bool) if "core" in z.files else None,
        labels_root=labels_root_for(slice_path),
    )
    logger.info(f"loaded slice: {sl.summary()}")
    if view is not None:
        sl = sl.view(view)
        if core_bank is not None:
            sl.Ae, sl.Ae_mask, sl.Qe = _load_answers(core_bank, sl.A.shape, mmap)
        logger.info(f"view {view!r}: {sl.summary()}")
    elif sl.has_gaps and not allow_gaps:
        raise ValueError(
            f"{slice_path.name} is a merged slice with uncollected cells (NaN in A); load it "
            f"with view= one of {VIEWS} (train/evaluate: --view)"
        )
    return sl


def save_slice(
    sl: Slice,
    slice_path: Path | str = DEFAULT_SLICE,
    answers_path: Path | str | None = DEFAULT_ANSWERS,
) -> None:
    """Write a slice to disk in the layout :func:`load_slice` expects."""
    slice_path = Path(slice_path)
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "A": sl.A.astype(np.float32),
        "R": sl.R.astype(np.uint8),
        "bench": sl.bench.astype(np.int32),
        "bench_names": np.array(sl.bench_names, dtype=str),
        "model_ids": np.array(sl.model_ids, dtype=str),
        "question_ids": np.array(sl.question_ids, dtype=str),
    }
    if sl.vram_gb is not None:
        payload["vram_gb"] = sl.vram_gb.astype(np.float32)
    if sl.release_date:
        payload["release_date"] = np.array(sl.release_date, dtype=str)
    if sl.split:
        payload["split"] = np.array(sl.split, dtype=str)
    if sl.covered is not None:
        payload["covered"] = np.asarray(sl.covered, dtype=bool)
    for key in ("in_fullset", "in_extset", "core"):
        if getattr(sl, key) is not None:
            payload[key] = np.asarray(getattr(sl, key), dtype=bool)
    np.savez_compressed(slice_path, **payload)
    logger.info(f"wrote {slice_path} ({slice_path.stat().st_size / 1e6:.1f} MB)")

    if sl.Ae is not None and answers_path is not None:
        answers_path = Path(answers_path)
        banks = {"Ae": np.asarray(sl.Ae, dtype=ANSWER_STORAGE_DTYPE)}
        if sl.Ae_mask is not None:
            banks["Ae_mask"] = np.asarray(sl.Ae_mask, dtype=bool)
        if sl.Qe is not None:
            banks["Qe"] = np.asarray(sl.Qe, dtype=ANSWER_COMPUTE_DTYPE)
        np.savez(answers_path, **banks)
        logger.info(f"wrote {answers_path} ({answers_path.stat().st_size / 1e6:.1f} MB)")
