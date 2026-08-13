"""Stable facade for canonical result persistence, export, and inspection."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasma_global._result_common import (
    RESULT_CSV_NAME,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    RESULT_H5_NAME,
    SUMMARY_YAML_NAME,
)
from plasma_global._result_csv import read_result_csv as _read_result_csv
from plasma_global._result_csv import write_result_csv as _write_result_csv
from plasma_global._result_hdf5 import read_result_h5 as _read_result_h5
from plasma_global._result_hdf5 import write_result_h5 as _write_result_h5
from plasma_global._result_plot import plot_result_h5 as _plot_result_h5
from plasma_global._result_summary import (
    audit_from_metadata as _audit_from_metadata,
)
from plasma_global._result_summary import build_summary as _build_summary
from plasma_global._result_summary import write_summary_yaml as _write_summary_yaml
from plasma_global.audit import AuditReport, audit_result
from plasma_global.core.result import SimulationResult, SimulationStatus


@dataclass(frozen=True, slots=True)
class ResultPaths:
    """The canonical artifacts and their result-level audit outcome."""

    directory: Path
    result_h5: Path
    summary_yaml: Path
    audit_passed: bool = True


def write_result_h5(path: str | Path, result: SimulationResult) -> Path:
    """Write the fixed public HDF5 layout."""

    return _write_result_h5(path, result)


def read_result_h5(path: str | Path) -> SimulationResult:
    """Read the strict public HDF5 layout."""

    return _read_result_h5(path)


def write_result_csv(path: str | Path, result: SimulationResult) -> Path:
    """Export a canonical result to one reversible numeric CSV."""

    return _write_result_csv(path, result)


def read_result_csv(
    path: str | Path,
    *,
    status: SimulationStatus | None = None,
    solver_stats: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Read a CSV produced by :func:`write_result_csv`."""

    return _read_result_csv(
        path,
        status=status,
        solver_stats=solver_stats,
        metadata=metadata,
    )


def build_summary(
    result: SimulationResult, *, audit: AuditReport | None = None
) -> dict[str, Any]:
    """Build the fixed summary from result-declared quantities only."""

    return _build_summary(result, audit=audit)


def write_summary_yaml(
    path: str | Path,
    result: SimulationResult,
    *,
    audit: AuditReport | None = None,
) -> Path:
    """Write the stable human-readable result summary."""

    return _write_summary_yaml(path, result, audit=audit)


def write_result(result: SimulationResult, output_dir: str | Path) -> ResultPaths:
    """Atomically publish the canonical HDF5 and summary pair."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = _audit_from_metadata(result)
    result_h5 = directory / RESULT_H5_NAME
    summary_yaml = directory / SUMMARY_YAML_NAME
    with tempfile.TemporaryDirectory(prefix=".plasma-global-", dir=directory) as raw:
        stage = Path(raw)
        staged_h5 = write_result_h5(stage / RESULT_H5_NAME, result)
        staged_summary = write_summary_yaml(
            stage / SUMMARY_YAML_NAME, result, audit=report
        )
        targets = ((staged_h5, result_h5), (staged_summary, summary_yaml))
        backups: dict[Path, Path | None] = {}
        for _, target in targets:
            if target.exists():
                backup = stage / f"{target.name}.previous"
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        replaced: list[Path] = []
        try:
            for staged, target in targets:
                os.replace(staged, target)
                replaced.append(target)
        except OSError:
            for target in reversed(replaced):
                backup = backups[target]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise
    return ResultPaths(directory, result_h5, summary_yaml, audit_passed=report.passed)


def export_result_csv(result_h5: str | Path, output_csv: str | Path) -> Path:
    """Convert canonical HDF5 to the explicit CSV export format."""

    return write_result_csv(output_csv, read_result_h5(result_h5))


def audit_result_h5(
    result_h5: str | Path,
    *,
    conservation_observables: Iterable[str] = (),
    conservation_tolerances: Mapping[str, float] | None = None,
    default_conservation_tolerance: float | None = None,
    nonnegative_series: Iterable[str] = (),
    negative_tolerance: float = 0.0,
) -> AuditReport:
    """Audit one result with explicitly selected physical constraints."""

    result = read_result_h5(result_h5)
    residuals = {name: result.series(name) for name in conservation_observables}
    return audit_result(
        result,
        conservation_residuals=residuals,
        conservation_tolerances=conservation_tolerances,
        default_conservation_tolerance=default_conservation_tolerance,
        nonnegative_series=nonnegative_series,
        negative_tolerance=negative_tolerance,
    )


def plot_result_h5(
    result_h5: str | Path,
    output_dir: str | Path,
    *,
    series: Sequence[str] = (),
    image_format: str = "png",
    dpi: int = 150,
) -> tuple[Path, ...]:
    """Plot selected result series through the optional plotting adapter."""

    return _plot_result_h5(
        result_h5,
        output_dir,
        series=series,
        image_format=image_format,
        dpi=dpi,
        result_reader=read_result_h5,
    )


__all__ = [
    "RESULT_CSV_NAME",
    "RESULT_FORMAT",
    "RESULT_FORMAT_VERSION",
    "RESULT_H5_NAME",
    "SUMMARY_YAML_NAME",
    "ResultPaths",
    "audit_result_h5",
    "build_summary",
    "export_result_csv",
    "plot_result_h5",
    "read_result_csv",
    "read_result_h5",
    "write_result",
    "write_result_csv",
    "write_result_h5",
    "write_summary_yaml",
]
