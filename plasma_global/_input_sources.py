"""Read-once input bytes and their provenance for one case preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(slots=True)
class InputSourceStore:
    """Keep the exact bytes consumed by readers, independently of later edits."""

    _content: dict[Path, bytes] = field(default_factory=dict)
    _roles: dict[Path, set[str]] = field(default_factory=dict)

    def import_record(self, path: str | Path, content: bytes, role: str) -> None:
        source = Path(path).resolve()
        if source in self._content and self._content[source] != content:
            raise ValueError(f"input changed during preparation: {source}")
        self._content[source] = content
        self._roles.setdefault(source, set()).add(role)

    def read_bytes(self, path: str | Path, role: str) -> bytes:
        source = Path(path).resolve()
        if source not in self._content:
            self._content[source] = source.read_bytes()
        self._roles.setdefault(source, set()).add(role)
        return self._content[source]

    def read_text(
        self, path: str | Path, role: str, encoding: str = "utf-8-sig"
    ) -> str:
        return self.read_bytes(path, role).decode(encoding)

    def provenance(self) -> Mapping[str, Any]:
        files = {
            str(path): MappingProxyType(
                {
                    "sha256": sha256(self._content[path]).hexdigest(),
                    "size_bytes": len(self._content[path]),
                    "roles": tuple(sorted(self._roles[path])),
                }
            )
            for path in sorted(self._content, key=str)
        }
        return MappingProxyType(
            {"algorithm": "sha256", "files": MappingProxyType(files)}
        )

    def contents(self) -> Mapping[Path, bytes]:
        """Return the captured dependency bytes for relocating an installed example."""

        return MappingProxyType(dict(self._content))
