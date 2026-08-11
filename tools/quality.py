"""Stable console entry points for repository quality gates."""

from tools.quality_gate.commands import (
    baseline_main,
    fast_main,
    nightly_main,
    pr_main,
)

__all__ = ["baseline_main", "fast_main", "nightly_main", "pr_main"]


if __name__ == "__main__":
    raise SystemExit("Use one of the quality-* console commands")
