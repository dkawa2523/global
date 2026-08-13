"""Deterministic input and software provenance shared by runs and audits."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np
import scipy

_PACKAGE_NAME = "plasma-global-model"


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _add_input_role(
    roles: dict[Path, set[str]], path: Path | str | None, role: str
) -> None:
    if path is not None:
        roles.setdefault(Path(path).resolve(), set()).add(role)


def _input_roles(case: Any, chemistry_data: Any) -> dict[Path, set[str]]:
    roles: dict[Path, set[str]] = {}
    _add_input_role(roles, case.source_path, "case")
    for path in case.included_files:
        _add_input_role(roles, path, "case_include")
    for path in chemistry_data.source_files:
        _add_input_role(roles, path, "chemistry_input")
    _add_input_role(
        roles,
        getattr(case.models.electrons, "file", None),
        "electron_kinetics_table",
    )
    _add_input_role(
        roles,
        getattr(case.models.electron_density, "file", None),
        "electron_density_profile",
    )
    for port in case.reactor.power_ports:
        _add_input_role(
            roles, getattr(port.model, "file", None), f"power_port:{port.port_id}"
        )
    for step in case.recipe.steps:
        for port_id, command in step.commands.power_ports.items():
            _add_input_role(
                roles,
                getattr(command, "file", None),
                f"recipe_power_override:{step.step_id}:{port_id}",
            )
    return roles


def collect_file_provenance(case: Any, chemistry_data: Any) -> Mapping[str, Any]:
    """Hash every file whose bytes have been consumed by case compilation."""

    from plasma_global.input.schema import CaseSpec

    if not isinstance(case, CaseSpec):
        raise TypeError("case must be a schema-v3 CaseSpec")
    files: dict[str, Any] = {}
    roles = _input_roles(case, chemistry_data)

    def normalized_path(path: Path) -> str:
        return str(path).casefold()

    for path in sorted(roles, key=normalized_path):
        digest, size = _file_sha256(path)
        files[str(path)] = {
            "sha256": digest,
            "size_bytes": size,
            "roles": sorted(roles[path]),
        }
    return MappingProxyType({"algorithm": "sha256", "files": MappingProxyType(files)})


def _commit_id(value: str | None) -> str | None:
    candidate = "" if value is None else value.strip()
    return candidate.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", candidate) else None


def _git_directory(root: Path) -> Path | None:
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    try:
        prefix, path = marker.read_text(encoding="utf-8").strip().split(":", 1)
    except (OSError, ValueError):
        return None
    if prefix != "gitdir":
        return None
    return (root / path.strip()).resolve()


def _packed_revision(git_directory: Path, reference: str) -> str | None:
    try:
        lines = (git_directory / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith(("#", "^")):
            commit, _, name = line.partition(" ")
            if name == reference:
                return _commit_id(commit)
    return None


def _repository_revision(root: Path) -> str | None:
    for name in ("GITHUB_SHA", "CI_COMMIT_SHA"):
        if revision := _commit_id(os.environ.get(name)):
            return revision
    git_directory = _git_directory(root)
    if git_directory is None:
        return None
    try:
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if revision := _commit_id(head):
        return revision
    prefix, separator, reference = head.partition(":")
    if prefix != "ref" or not separator:
        return None
    try:
        revision = _commit_id(
            (git_directory / reference.strip()).read_text(encoding="utf-8")
        )
    except OSError:
        revision = None
    return revision or _packed_revision(git_directory, reference.strip())


def software_provenance() -> Mapping[str, Any]:
    """Describe the package, source revision, and numerical runtime."""

    try:
        package_version = version(_PACKAGE_NAME)
    except PackageNotFoundError:
        package_version = "unknown"
    repository = Path(__file__).resolve().parents[1]
    revision = _repository_revision(repository)
    clean_ci_checkout = (
        os.environ.get("GITHUB_ACTIONS") == "true" and revision is not None
    )
    return MappingProxyType(
        {
            "package": {"name": _PACKAGE_NAME, "version": package_version},
            "git": {
                "revision": revision,
                "dirty": False if clean_ci_checkout else None,
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "h5py": h5py.__version__,
            },
        }
    )


def build_artifact_provenance(
    case: Any,
    chemistry_data: Any,
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Enrich domain provenance once, before simulation or audit output."""

    return {
        **dict(base),
        "input_files": collect_file_provenance(case, chemistry_data),
        "software": software_provenance(),
    }


__all__ = [
    "build_artifact_provenance",
    "collect_file_provenance",
    "software_provenance",
]
