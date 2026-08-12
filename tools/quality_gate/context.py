"""Shared paths, process execution, and Git context for quality commands."""

from __future__ import annotations

import ast
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[2]
QUALITY_DIR = ROOT / "quality"
REPORT_DIR = ROOT / ".quality-reports"
BASELINE_PATH = QUALITY_DIR / "baseline.json"
PYREFLY_BASELINE_PATH = QUALITY_DIR / "pyrefly-baseline.json"
SECRETS_BASELINE_PATH = ROOT / ".secrets.baseline"
SOURCE_PATHS = ("plasma_global", "tools")
ALL_PYTHON_PATHS = (*SOURCE_PATHS, "tests")


class QualityFailure(RuntimeError):
    """A quality gate failed with an actionable message."""


@dataclass(frozen=True)
class Finding:
    """A line-stable fingerprint and its human-readable location."""

    key: str
    display: str


def _die(message: str) -> NoReturn:
    raise QualityFailure(message)


def _environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    if extra:
        environment.update(extra)
    return environment


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    allowed: set[int] | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"\n> {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    expected = {0} if allowed is None else allowed
    if result.returncode not in expected:
        _die(
            f"Command failed with exit code {result.returncode}: {shlex.join(command)}"
        )
    return result


def _capture(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    allowed: set[int] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    expected = {0} if allowed is None else allowed
    if result.returncode not in expected:
        details = "\n".join(part for part in (result.stdout, result.stderr) if part)
        _die(
            f"Command failed with exit code {result.returncode}: "
            f"{shlex.join(command)}\n{details}"
        )
    return result.stdout


def _git(*arguments: str, allowed: set[int] | None = None) -> str:
    return _capture(["git", *arguments], allowed=allowed)


def _base_ref() -> str | None:
    explicit = os.environ.get("QUALITY_BASE_REF")
    if explicit and set(explicit) != {"0"}:
        merge_base = _git("merge-base", "HEAD", explicit, allowed={0, 1, 128}).strip()
        return merge_base or explicit
    upstream = _git(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        allowed={0, 128},
    ).strip()
    if upstream:
        return _git("merge-base", "HEAD", upstream).strip() or None
    if _capture(["git", "rev-parse", "HEAD^"], allowed={0, 128}).strip():
        return "HEAD^"
    return None


def _changed_python_files() -> list[str]:
    files: set[str] = set()
    base = _base_ref()
    if base:
        files.update(
            line.strip()
            for line in _git(
                "diff", "--name-only", "--diff-filter=ACMR", base, "--", "*.py"
            ).splitlines()
            if line.strip()
        )
    files.update(
        line.strip()
        for line in _git(
            "diff", "--name-only", "--diff-filter=ACMR", "HEAD", "--", "*.py"
        ).splitlines()
        if line.strip()
    )
    files.update(
        line.strip()
        for line in _git(
            "ls-files", "--others", "--exclude-standard", "--", "*.py"
        ).splitlines()
        if line.strip()
    )
    return sorted(path for path in files if (ROOT / path).is_file())


def _source_for_path(
    relative_path: str, *, source_ref: str | None = None
) -> str | None:
    normalized_path = relative_path.replace("\\", "/")
    if source_ref is not None:
        return _git(
            "show",
            f"{source_ref}:{normalized_path}",
            allowed={0, 128},
        )
    try:
        return (ROOT / normalized_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _symbol_for_source(source: str | None, line_number: int) -> str:
    if source is None:
        return "<module>"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return "<module>"
    matches: list[tuple[int, str]] = []

    def visit(nodes: list[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno)
                qualified = ".".join((*parents, node.name))
                if node.lineno <= line_number <= end:
                    matches.append((len(parents), qualified))
                    visit(node.body, (*parents, node.name))

    visit(tree.body, ())
    return max(matches, default=(-1, "<module>"))[1]


def _symbol_for_line(relative_path: str, line_number: int) -> str:
    return _symbol_for_source(_source_for_path(relative_path), line_number)


def _relative_path(filename: str) -> str:
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _finding_key(
    tool: str,
    rule: str,
    filename: str,
    line_number: int,
    message: str,
) -> Finding:
    path = _relative_path(filename)
    symbol = _symbol_for_line(path, line_number)
    normalized = " ".join(message.split())
    key = "|".join((tool, rule, path, symbol, normalized))
    return Finding(key, f"{path}:{line_number}: {rule} {normalized} [{symbol}]")
