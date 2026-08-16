"""Shared primitives for the schema-v2 migration translators.

This module deliberately contains only representation-agnostic helpers.  Domain
translation belongs in the corresponding ``_migrate_v2_*`` module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError


def first(values: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    """Return the first present, non-null legacy value from ``names``."""

    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return default


def float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def resolved_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def external_file(
    values: Mapping[str, Any],
    *,
    base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
) -> Path | None:
    """Resolve an inline legacy file or a named external input."""

    raw_file = values.get("file")
    if raw_file:
        return resolved_path(raw_file, base_dir)
    raw_key = values.get("file_key")
    if raw_key:
        key = str(raw_key)
        target = external_inputs.get(key)
        if not target:
            raise MigrationError(
                f"v2 external input {key!r} is referenced but has no resolved file"
            )
        used_external.add(key)
        return Path(target).resolve(strict=False)
    return None


def record_extra_keys(
    values: Mapping[str, Any], known: set[str], prefix: str, unused: set[str]
) -> None:
    """Record unconsumed legacy keys using their source path."""

    unused.update(f"{prefix}.{key}" for key in values if key not in known)
