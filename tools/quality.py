"""Repository-only command runner for quality gates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from tools.quality_gate.commands import (
    baseline_main,
    fast_main,
    nightly_main,
    pr_main,
)

_COMMANDS: dict[str, Callable[[], None]] = {
    "baseline": baseline_main,
    "fast": fast_main,
    "nightly": nightly_main,
    "pr": pr_main,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one explicit repository quality command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(_COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if parsed.arguments and parsed.command != "nightly":
        parser.error(f"{parsed.command} does not accept additional arguments")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *parsed.arguments]
        _COMMANDS[parsed.command]()
    finally:
        sys.argv = original_argv
    return 0


__all__ = ["baseline_main", "fast_main", "main", "nightly_main", "pr_main"]


if __name__ == "__main__":
    raise SystemExit(main())
