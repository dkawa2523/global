"""Literal physical-series selectors with an explicit star wildcard."""

from __future__ import annotations

import re
from collections.abc import Iterable

from plasma_global.errors import CaseValidationError


def expand_series(patterns: Iterable[str], available: Iterable[str]) -> tuple[str, ...]:
    """Only '*' is special; brackets in physical series names stay literal."""
    catalog = tuple(available)
    selected: list[str] = []
    for pattern in patterns:
        regex = re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")
        matches = [name for name in catalog if regex.fullmatch(name)]
        if not matches:
            raise CaseValidationError(f"output series matches nothing: {pattern!r}")
        selected.extend(matches)
    return tuple(dict.fromkeys(selected))
