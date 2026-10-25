"""Deterministic input and software provenance shared by runs and audits."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

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

    from plasma_global.input._resolved_case import ResolvedCase

    if not isinstance(case, ResolvedCase):
        raise TypeError("case must be a schema-v4 ResolvedCase")
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
