"""Ask each model every probe question and grade the answer (library for ``collect.py``).

One graded answer per (model, question): the latest single response, greedy for
standard models and sampled (``thinking_temperature``) for reasoning models, under
the generation budget of ``config/collect.yaml``. Grading is :mod:`bullpen.grading`,
keyed on (benchmark, model): which extractor reads the answer out of the response,
whether the post-``</think>`` region is read first, whether the gold is an option
letter or a literal string. Every answer lands in
``<raw>/responses/<model>/results.jsonl``, one JSON row per question in probe order::

    {"id", "axis", "correct", "n_tok", "model", "draw_idx", "prediction",
     "finish_reason", "config_hash"}

Two sources of responses: :class:`HFBackend` generates with ``transformers`` (needs
the ``collect`` extra and weights), and :func:`recorded_responses` reads
pre-recorded responses (``<dir>/<model>.jsonl`` rows ``{"id", "prediction"}``), which
is how the sample pipeline and the tests collect offline.
"""

from __future__ import annotations

import hashlib
import json
import lzma
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger

from bullpen.config import GenerationConfig
from bullpen.data.assemble import responses_path
from bullpen.grading import (
    DEFAULT_REASONING_MARKERS,
    DEFAULT_SANDBOX,
    Sandbox,
    grade_item,
    unrouted,
)


class CollectError(RuntimeError):
    """The collection cannot proceed: a missing probe, an ungradeable item."""


#: The probe names its grader as ``bench:<benchmark>``.
_GRADER_PREFIX = "bench:"


def benchmark_of(grader: str) -> str:
    """The benchmark a probe row's ``grader`` field names."""
    return grader.removeprefix(_GRADER_PREFIX)


@dataclass(frozen=True)
class ProbeItem:
    """One graded question: what the model is asked and what counts as right.

    ``format`` (``mcq`` / ``free``) and ``n_choices`` are the answer protocol the row
    was prepared under; ``n_choices`` bounds the option letters an MCQ extractor
    accepts.
    """

    id: str
    axis: str
    question: str
    gold: str
    grader: str
    format: str | None = None
    n_choices: int | None = None

    def as_item(self) -> dict[str, Any]:
        """The mapping :func:`bullpen.grading.grade_item` grades against."""
        item: dict[str, Any] = {"gold": self.gold}
        if self.format is not None:
            item["format"] = self.format
        if self.n_choices is not None:
            item["n_choices"] = self.n_choices
        return item


_PROBE_FIELDS = ("id", "axis", "question", "gold", "grader")


def load_probe(path: Path) -> list[ProbeItem]:
    """Read the probe set (JSONL, one question per line), refusing unroutable graders."""
    path = Path(path)
    if not path.is_file():
        raise CollectError(f"no probe set at {path}")
    items: list[ProbeItem] = []
    # split("\n") and not splitlines(): a JSON string may legally contain U+2028
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [f for f in _PROBE_FIELDS if row.get(f) is None]
        if missing:
            raise CollectError(f"{path}:{lineno} is missing probe field(s): {missing}")
        n_choices = row.get("n_choices")
        items.append(
            ProbeItem(
                **{f: str(row[f]) for f in _PROBE_FIELDS},
                format=None if row.get("format") is None else str(row["format"]),
                n_choices=None if n_choices is None else int(n_choices),
            )
        )
    ungradeable = unrouted(benchmark_of(i.grader) for i in items)
    if ungradeable:
        raise CollectError(f"{path} names grader(s) {ungradeable} that bullpen.grading lacks")
    logger.info(f"probe: {len(items)} questions across {len({i.axis for i in items})} benchmarks")
    return items


def grade_answer(
    item: ProbeItem,
    output: str,
    model: str | None = None,
    sandbox: Sandbox = DEFAULT_SANDBOX,
    markers: Sequence[str] = DEFAULT_REASONING_MARKERS,
) -> bool:
    """Grade one answer through :mod:`bullpen.grading`."""
    return grade_item(benchmark_of(item.grader), output, item.as_item(), model, sandbox, markers)


def config_hash(model_id: str, temperature: float, max_new_tokens: int, draws: int) -> str:
    """Identifies the generation settings a row was produced under."""
    payload = json.dumps(
        {
            "model": model_id,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "draws": draws,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_results(path: Path, order: Sequence[str], rows: Mapping[str, dict[str, Any]]) -> None:
    """Write one model's rows in probe order, via a temp file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for qid in order:
            row = rows.get(qid)
            if row is not None:
                fh.write(json.dumps({"id": qid, **{k: v for k, v in row.items() if k != "id"}}))
                fh.write("\n")
    tmp.rename(path)


class Generation(NamedTuple):
    """One decoded answer."""

    text: str
    n_tokens: int
    finish_reason: str


#: Used when a tokenizer carries no chat template of its own.
CHATML_TEMPLATE = (
    "{% for m in messages %}"
    r"{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    r"{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
CHATML_STOP = ("<|im_start|>", "<|im_end|>")


class HFBackend:
    """One loaded checkpoint (local directory or Hub repo id), generating in batches.

    torch and transformers are imported here, not at module scope, so the offline
    path (recorded responses) needs neither.
    """

    def __init__(self, model: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self.batch_size = batch_size
        logger.info(f"{model}: loading weights")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        self.template, stops = (
            (None, ())
            if getattr(self.tokenizer, "chat_template", None)
            else (CHATML_TEMPLATE, CHATML_STOP)
        )
        self.stop_ids = [
            ids[0] for s in stops if (ids := self.tokenizer.encode(s, add_special_tokens=False))
        ]

    def prompt(self, question: str, thinking: bool) -> str:
        """Wrap one question as a chat turn the model recognises."""
        messages = [{"role": "user", "content": question}]
        kwargs: dict[str, Any] = {}
        if self.template:
            kwargs["chat_template"] = self.template
        if thinking:
            try:
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                kwargs["enable_thinking"] = True
            except TypeError:
                pass  # this tokenizer has no thinking switch
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        temperature: float,
        gen: GenerationConfig,
    ) -> list[Generation]:
        import torch

        tok = self.tokenizer
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        sample = temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "pad_token_id": tok.pad_token_id,
        }
        if sample:
            kwargs.update(temperature=temperature, top_p=gen.top_p, top_k=gen.top_k)
        eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
        eos += [i for i in self.stop_ids if i not in eos]
        if eos:
            kwargs["eos_token_id"] = eos
        out: list[Generation] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = list(prompts[i : i + self.batch_size])
            enc = tok(batch, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                ids = self.model.generate(**enc, **kwargs)
            prompt_len = enc.input_ids.shape[1]
            for j in range(len(batch)):
                new = ids[j, prompt_len:]
                if tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id:
                    new = new[new != tok.pad_token_id]
                n = int(new.shape[0])
                out.append(
                    Generation(
                        text=tok.decode(new, skip_special_tokens=True),
                        n_tokens=n,
                        finish_reason="length" if n >= max_new_tokens else "stop",
                    )
                )
        return out


def recorded_responses(directory: Path, model_id: str) -> dict[str, str]:
    """``{question id: response text}`` from ``<directory>/<model_id>.jsonl`` (or ``.jsonl.xz``)."""
    path = Path(directory) / f"{model_id}.jsonl"
    if not path.is_file() and path.with_suffix(".jsonl.xz").is_file():
        path = path.with_suffix(".jsonl.xz")
    if not path.is_file():
        raise CollectError(f"no recorded responses for {model_id} at {path}")
    raw = lzma.decompress(path.read_bytes()) if path.suffix == ".xz" else path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]
    return {str(r["id"]): str(r["prediction"]) for r in rows}


def collect_model(
    model_id: str,
    probe: Sequence[ProbeItem],
    raw: Path,
    gen: GenerationConfig,
    recorded: Path | None = None,
    sandbox: Sandbox = DEFAULT_SANDBOX,
    markers: Sequence[str] = DEFAULT_REASONING_MARKERS,
) -> Path:
    """Generate (or read) and grade one model's answers; returns its results.jsonl."""
    max_new_tokens, temperature = gen.budget(model_id)
    thinking = gen.is_thinking(model_id)
    chash = config_hash(model_id, temperature, max_new_tokens, gen.draws)
    if recorded is not None:
        texts = recorded_responses(recorded, model_id)
        missing = [i.id for i in probe if i.id not in texts]
        if missing:
            logger.warning(f"{model_id}: {len(missing)} probe question(s) have no recorded answer")
        outs = {
            i.id: Generation(texts[i.id], len(texts[i.id].split()), "stop")
            for i in probe
            if i.id in texts
        }
    else:
        backend = HFBackend(model_id, gen.batch_size)
        prompts = [backend.prompt(i.question, thinking) for i in probe]
        gens = backend.generate(prompts, max_new_tokens, temperature, gen)
        outs = {i.id: g for i, g in zip(probe, gens, strict=True)}
    rows = {
        item.id: {
            "axis": item.axis,
            "correct": int(grade_answer(item, out.text, model_id, sandbox, markers)),
            "n_tok": out.n_tokens,
            "model": model_id,
            "draw_idx": 0,
            "prediction": out.text,
            "finish_reason": out.finish_reason,
            "config_hash": chash,
        }
        for item in probe
        if (out := outs.get(item.id)) is not None
    }
    path = responses_path(raw, model_id)
    write_results(path, [i.id for i in probe], rows)
    acc = sum(r["correct"] for r in rows.values()) / max(len(rows), 1)
    logger.info(f"{model_id}: {len(rows)}/{len(probe)} answered, accuracy {acc:.3f} -> {path}")
    return path
