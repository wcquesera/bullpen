"""Executing model-generated Python, and the code grader built on it.

The ``code`` benchmarks (humaneval, mbpp) run the model's function against the
reference tests, so this module executes untrusted code. Two layers of containment:

1. **bubblewrap**: a read-only root, a private ``/tmp`` and no network namespace.
2. **rlimits, ``python -I``, a scrubbed environment and** :data:`CODE_GUARD`, an
   in-process guard (after HumanEval's ``reliability_guard``) that blocks the
   destructive and network entry points.

:meth:`Sandbox.backend` reports which is in force. ``"subprocess"`` means bwrap is
missing and only layer 2 applies: the code can still read the filesystem and reach the
network. ``execute: false`` in ``config/grading.yaml`` disables execution, and code
cells then grade as incorrect.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from typing import Literal

from loguru import logger

from bullpen.grading.extract import extract_code

Backend = Literal["auto", "bwrap", "none"]

#: Wall-clock seconds one program may run for.
DEFAULT_TIMEOUT_SECONDS = 15
#: Address-space ceiling of the child.
DEFAULT_MEMORY_GB = 4
#: Write ceiling, so a runaway solution cannot fill the disk.
DEFAULT_MAX_FILE_MB = 32

_BYTES_PER_GB = 1024**3
_BYTES_PER_MB = 1024**2

# Prepended to every program. Deliberately narrow: `os.path`, file I/O and the
# rest of the stdlib stay usable, because legitimate solutions use them.
CODE_GUARD = """\
import builtins as _b, os as _os, sys as _sys
def _blocked(*_a, **_k):
    raise RuntimeError("blocked by the benchmark sandbox")
for _m, _names in (
        ("os", ("system", "popen", "execv", "execve", "execvp", "fork",
                "forkpty", "kill", "killpg", "remove", "unlink", "rmdir",
                "removedirs", "renames", "truncate", "setuid", "chown",
                "chmod", "chroot")),
        ("shutil", ("rmtree", "move", "chown")),
        ("subprocess", ("Popen", "run", "call", "check_call", "check_output")),
        ("socket", ("socket", "create_connection")),
        ("multiprocessing", ("Process", "Pool")),
        ("urllib.request", ("urlopen",)),
        ("http.client", ("HTTPConnection", "HTTPSConnection")),
        ("ftplib", ("FTP",)),
):
    try:
        _mod = __import__(_m, fromlist=["_"])
    except Exception:
        continue
    for _n in _names:
        if hasattr(_mod, _n):
            try:
                setattr(_mod, _n, _blocked)
            except Exception:
                pass
_b.exit = _b.quit = _blocked
_os.environ.clear()
_sys.setrecursionlimit(10000)
"""

#: How much of the child's stderr is kept for the failure log.
_STDERR_TAIL = 400


def _bwrap_argv(workdir: str) -> list[str]:
    """The bubblewrap wrapper: read-only root, private /tmp, no namespaces."""
    return [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        workdir,
        workdir,
        "--chdir",
        workdir,
    ]


#: Seconds allowed for the one-off "does bwrap work here?" probe.
_BWRAP_PROBE_TIMEOUT = 30


@cache
def bwrap_available() -> bool:
    """Whether bwrap is present and actually works here (probed once, cached)."""
    ok = False
    if shutil.which("bwrap"):
        try:
            probe = subprocess.run(
                [*_bwrap_argv("/tmp"), sys.executable, "-I", "-c", "pass"],
                capture_output=True,
                timeout=_BWRAP_PROBE_TIMEOUT,
                check=False,
            )
            ok = probe.returncode == 0
        except Exception:  # noqa: BLE001 — any probe failure means "not usable"
            ok = False
    state = "available" if ok else "unavailable — subprocess+rlimits only"
    logger.info(f"code-exec sandbox: bwrap {state}")
    return ok


@dataclass(frozen=True)
class Sandbox:
    """The containment one collection run grades its code cells under."""

    backend_choice: Backend = "auto"
    execute: bool = True
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    memory_gb: int = DEFAULT_MEMORY_GB
    max_file_mb: int = DEFAULT_MAX_FILE_MB

    def backend(self) -> str:
        """The backend :meth:`run` will actually use: ``'bwrap'`` or ``'subprocess'``."""
        if self.backend_choice == "none":
            return "subprocess"
        if self.backend_choice == "bwrap":
            return "bwrap"
        return "bwrap" if bwrap_available() else "subprocess"

    def _limits(self) -> object:
        """A ``preexec_fn`` applying the CPU, memory and file-size ceilings."""

        def apply() -> None:  # pragma: no cover — runs in the forked child
            import resource

            cpu = self.timeout_seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            mem = self.memory_gb * _BYTES_PER_GB
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            size = self.max_file_mb * _BYTES_PER_MB
            resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))

        return apply

    def run(self, program: str, guard: bool = True) -> tuple[bool, str]:
        """Execute ``program``. Returns ``(exited zero, stderr tail)``.

        Never raises: a broken child is a failed code cell, not a failed run.
        """
        if not self.execute:
            _warn_execution_disabled()
            return False, "code execution disabled"
        src = f"{CODE_GUARD}\n{program}" if guard else program
        with tempfile.TemporaryDirectory() as td:
            argv = [sys.executable, "-I", "-c", src]
            if self.backend() == "bwrap":
                argv = _bwrap_argv(td) + argv
            try:
                proc = subprocess.run(
                    argv,
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    preexec_fn=self._limits(),
                    env={"PATH": "/usr/bin:/bin", "HOME": td, "PYTHONDONTWRITEBYTECODE": "1"},
                    check=False,
                )
                return proc.returncode == 0, (proc.stderr or "")[-_STDERR_TAIL:]
            except subprocess.TimeoutExpired:
                return False, "timeout"
            except Exception as exc:  # noqa: BLE001 — a broken child must not end the run
                return False, f"{type(exc).__name__}: {exc}"


#: The sandbox used when no run has supplied one — the safe defaults.
DEFAULT_SANDBOX = Sandbox()


@cache
def _warn_execution_disabled() -> None:
    """Warn once that code cells are grading as incorrect without being run."""
    logger.warning(
        "code execution is disabled — code benchmarks grade as incorrect; set "
        "sandbox.execute in config/grading.yaml to enable execution grading"
    )


def code_gold(**fields: object) -> str:
    """Serialise the executable test payload into the string ``gold`` field."""
    return json.dumps(fields)


def build_code_program(completion: str, gold: str) -> str:
    """Assemble the runnable program: model code, then the harness tests."""
    spec = json.loads(gold)
    parts = [
        spec.get("preamble", ""),
        extract_code(completion),
        spec.get("setup", ""),
        spec.get("test", ""),
    ]
    if spec.get("entry_point"):
        parts.append(f"check({spec['entry_point']})")
    return "\n\n".join(p for p in parts if p)


def grade_code(output: str, gold: str, sandbox: Sandbox = DEFAULT_SANDBOX) -> bool:
    """Correct iff the assembled program exits zero under the sandbox."""
    ok, err = sandbox.run(build_code_program(output, gold))
    if not ok and err and err not in ("timeout", "code execution disabled"):
        logger.debug(f"code cell failed: {err.splitlines()[-1] if err else ''}")
    return ok
