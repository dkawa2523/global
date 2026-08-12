from __future__ import annotations

from pathlib import Path

import pytest

from tools.quality_gate.baselines import _pyrefly_fingerprints


def _line_number(source: str, text: str) -> int:
    return next(
        line_number
        for line_number, line in enumerate(source.splitlines(), start=1)
        if text in line
    )


def _pyrefly_payload(path: str, line: int, message: str) -> dict[str, object]:
    return {
        "errors": [
            {
                "path": path,
                "line": line,
                "name": "unnecessary-type-conversion",
                "severity": "warn",
                "concise_description": message,
            }
        ]
    }


def test_pyrefly_fingerprint_ignores_line_moves_within_same_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "plasma_global/example.py"
    base_source = """\
class Example:
    def convert(self):
        return str(self.value)
"""
    current_source = """\
class Example:
    def convert(self):
        value = self.value

        return str(value)
"""
    source_path = tmp_path / path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(current_source, encoding="utf-8")

    def fake_git(*arguments: str, allowed: set[int] | None = None) -> str:
        assert arguments == ("show", f"base-ref:{path}")
        assert allowed == {0, 128}
        return base_source

    monkeypatch.setattr("tools.quality_gate.context.ROOT", tmp_path)
    monkeypatch.setattr("tools.quality_gate.context._git", fake_git)
    previous = _pyrefly_payload(
        path,
        _line_number(base_source, "return str"),
        "Unnecessary  conversion\ninside value",
    )
    current = _pyrefly_payload(
        path,
        _line_number(current_source, "return str"),
        "Unnecessary conversion inside value",
    )

    previous_fingerprints = _pyrefly_fingerprints(previous, source_ref="base-ref")
    current_fingerprints = _pyrefly_fingerprints(current)

    assert previous_fingerprints == current_fingerprints
    assert next(iter(current_fingerprints)).startswith(
        "plasma_global/example.py|Example.convert|unnecessary-type-conversion|warn|"
    )


def test_pyrefly_fingerprint_detects_move_to_different_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "plasma_global/example.py"
    source = """\
class Example:
    def first(self):
        return str(self.first_value)

    def second(self):
        return str(self.second_value)
"""
    source_path = tmp_path / path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")

    def fake_git(*arguments: str, allowed: set[int] | None = None) -> str:
        assert arguments == ("show", f"base-ref:{path}")
        assert allowed == {0, 128}
        return source

    monkeypatch.setattr("tools.quality_gate.context.ROOT", tmp_path)
    monkeypatch.setattr("tools.quality_gate.context._git", fake_git)
    previous = _pyrefly_payload(
        path, _line_number(source, "return str(self.first_value)"), "same warning"
    )
    current = _pyrefly_payload(
        path, _line_number(source, "return str(self.second_value)"), "same warning"
    )

    previous_fingerprints = _pyrefly_fingerprints(previous, source_ref="base-ref")
    current_fingerprints = _pyrefly_fingerprints(current)

    assert previous_fingerprints != current_fingerprints
    assert "|Example.first|" in next(iter(previous_fingerprints))
    assert "|Example.second|" in next(iter(current_fingerprints))
