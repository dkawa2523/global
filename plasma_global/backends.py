"""Small backend registry primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BackendSpec:
    builder: Callable[..., Any]
    description: str
    config_validator: Callable[..., None] | None = None
    maturity: str = 'reduced'
    intended_use: str = ''
    caveat: str = ''


@dataclass(frozen=True)
class BackendCatalog:
    entries: dict[str, BackendSpec]

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self.entries:
            raise KeyError(f"Unknown backend: {name}. Available: {self.names()}")
        return self.entries[name].builder(**kwargs)

    def names(self) -> list[str]:
        return sorted(self.entries)

    def details(self) -> dict[str, dict[str, str]]:
        return {
            name: {
                'description': spec.description,
                'maturity': spec.maturity,
                'intended_use': spec.intended_use,
                'caveat': spec.caveat,
            }
            for name, spec in sorted(self.entries.items())
        }


__all__ = ['BackendCatalog', 'BackendSpec']
