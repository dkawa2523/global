from __future__ import annotations

import sys

import pytest

from tools import quality


@pytest.mark.parametrize(
    ("command", "arguments"),
    [("fast", ()), ("nightly", ("--native",))],
)
def test_quality_runner_dispatches_and_forwards_command_arguments(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    arguments: tuple[str, ...],
) -> None:
    observed: list[tuple[str, ...]] = []
    original_argv = sys.argv
    monkeypatch.setitem(
        quality._COMMANDS,
        command,
        lambda: observed.append(tuple(sys.argv[1:])),
    )

    assert quality.main([command, *arguments]) == 0
    assert observed == [arguments]
    assert sys.argv is original_argv
