"""Identifiers embedded as components of public named physical series."""

from __future__ import annotations

import re

IDENTIFIER_PATTERN = r"^[^\s\[\],\x00\r\n](?:[^\[\],\x00\r\n]*[^\s\[\],\x00\r\n])?$"


def validate_identifier(value: str) -> str:
    """Keep Unicode and chemical punctuation while reserving series delimiters."""
    if not isinstance(value, str) or re.fullmatch(IDENTIFIER_PATTERN, value) is None:
        raise ValueError(
            "identifier must be nonempty, without edge whitespace, "
            "series delimiters [,], NUL or newlines"
        )
    return value
