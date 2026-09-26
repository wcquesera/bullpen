"""Console logging for the entry points.

:func:`configure_console` replaces loguru's pre-installed DEBUG stderr sink with one
at INFO (or ``BULLPEN_LOG_LEVEL``); the battery's per-readout DEBUG notes are too
verbose for a console.
"""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from typing import Any

from loguru import logger

#: overrides the console level for one process
CONSOLE_LEVEL_ENV = "BULLPEN_LOG_LEVEL"
DEFAULT_CONSOLE_LEVEL = "INFO"

#: loguru's pre-installed stderr sink is always handler 0
_PREINSTALLED_SINK_ID = 0

#: the handler :func:`configure_console` last added, so a second call replaces it
_own_sink: list[int] = []


def configure_console(level: str | None = None, stream: Any = None) -> str:
    """Replace the stderr sink with one at ``level`` (caller, env, then INFO); returns it.

    Removes only handler 0 and this function's own previous handler, so sinks added
    by a caller survive.
    """
    resolved = (level or os.environ.get(CONSOLE_LEVEL_ENV) or DEFAULT_CONSOLE_LEVEL).upper()
    doomed = _own_sink[0] if _own_sink else _PREINSTALLED_SINK_ID
    with suppress(ValueError):  # already removed, or never installed
        logger.remove(doomed)
    _own_sink[:] = [logger.add(sys.stderr if stream is None else stream, level=resolved)]
    return resolved


__all__ = ["CONSOLE_LEVEL_ENV", "DEFAULT_CONSOLE_LEVEL", "configure_console"]
