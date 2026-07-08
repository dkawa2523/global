"""Shared validation report primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str
    entity_id: str = ''


@dataclass
class ValidationReport:
    messages: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.level.upper() == 'ERROR' for issue in self.messages)

    def add(self, level: str, code: str, message: str, entity_id: str = '') -> None:
        self.messages.append(ValidationIssue(level=level, code=code, message=message, entity_id=entity_id))


__all__ = ['ValidationIssue', 'ValidationReport']
