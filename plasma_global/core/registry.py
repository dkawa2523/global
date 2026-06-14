from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Registration:
    name: str
    builder: Callable[..., Any]
    description: str = ''


@dataclass
class Registry:
    """Lightweight plugin registry.

    The registry is intentionally tiny so that reviewers can understand plugin
    resolution in a single file.  Each registration may carry a short human
    readable description, which is used by developer tooling and documentation.
    """

    _entries: dict[str, Registration] = field(default_factory=dict)

    def register(
        self,
        name: str,
        builder: Callable[..., Any],
        description: str = '',
    ) -> None:
        self._entries[name] = Registration(
            name=name,
            builder=builder,
            description=description,
        )

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._entries:
            raise KeyError(f"Unknown registry entry: {name}. Available: {self.names()}")
        return self._entries[name].builder(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def describe(self) -> dict[str, str]:
        return {name: reg.description for name, reg in sorted(self._entries.items())}

    def details(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                'description': reg.description,
            }
            for name, reg in sorted(self._entries.items())
        }
