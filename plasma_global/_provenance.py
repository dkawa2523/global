"""Deterministic input and software provenance shared by runs and audits."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from plasma_global._input_sources import InputSourceStore

_PACKAGE_NAME = "plasma-global-model"


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _source_fingerprint(package_directory: Path) -> dict[str, Any] | None:
    """Identify the package's source bytes, independently of Git or install path."""

    paths = sorted(
        path
        for path in package_directory.rglob("*.py")
        if "__pycache__" not in path.relative_to(package_directory).parts
    )
    if not paths:
        return None
    digest = sha256()
    for path in paths:
        content_hash, _ = _file_sha256(path)
        relative = path.relative_to(package_directory).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(content_hash))
    return {
        "algorithm": "sha256-python-source-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def software_provenance() -> Mapping[str, Any]:
    """Record source identity, the adjacent project lock, and runtime versions."""

    package_directory = Path(__file__).resolve().parent
    lockfile = package_directory.parent / "uv.lock"
    project_lock = None
    if lockfile.is_file():
        lock_hash, lock_size = _file_sha256(lockfile)
        project_lock = {
            "name": lockfile.name,
            "sha256": lock_hash,
            "size_bytes": lock_size,
        }
    runtime: dict[str, str | None] = {"python": platform.python_version()}
    runtime.update(
        {
            name: _installed_version(distribution)
            for name, distribution in (
                ("numpy", "numpy"),
                ("scipy", "scipy"),
                ("h5py", "h5py"),
                ("pydantic", "pydantic"),
                ("pyyaml", "PyYAML"),
                ("typing_extensions", "typing_extensions"),
            )
        }
    )
    return MappingProxyType(
        {
            "package": {
                "name": _PACKAGE_NAME,
                "version": _installed_version(_PACKAGE_NAME),
            },
            "source_tree": _source_fingerprint(package_directory),
            "project_lock": project_lock,
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "runtime": runtime,
        }
    )


def build_artifact_provenance(
    sources: InputSourceStore,
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Enrich domain provenance once, before simulation or audit output."""

    return {
        **dict(base),
        "input_files": sources.provenance(),
        "software": software_provenance(),
    }


__all__ = [
    "build_artifact_provenance",
    "software_provenance",
]
