"""Stable facade for canonical result persistence, export, and inspection."""

from __future__ import annotations

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
    new_output_directory,
)
from plasma_global._result_csv import read_result_csv as _read_result_csv
from plasma_global._result_csv import write_result_csv as _write_result_csv
from plasma_global._result_hdf5 import read_result_h5 as _read_result_h5
from plasma_global._result_hdf5 import write_result_h5 as _write_result_h5
from plasma_global._result_network import graphviz_module, write_networks
from plasma_global._result_plot import _load_pyplot, write_configured_plots
from plasma_global._result_plot import plot_result_h5 as _plot_result_h5
from plasma_global._result_summary import build_summary as _build_summary
from plasma_global._result_summary import write_series_yaml
from plasma_global._result_summary import write_summary_yaml as _write_summary_yaml
from plasma_global.audit import AuditReport, audit_result
from plasma_global.core.result import (
    SimulationResult,
    SimulationStatus,
    to_plain_mapping,
)
from plasma_global.input._schema_output import NetworkConfig, OutputConfig


@dataclass(frozen=True, slots=True)
class ResultPaths:
    """Numerical result and requested analysis artifacts in one immutable run."""

    directory: Path
    result_h5: Path
    summary_yaml: Path
    result_csv: Path | None = None
    analysis_files: tuple[Path, ...] = ()


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


def build_summary(result: SimulationResult) -> dict[str, Any]:
    """Build the fixed summary from result-declared quantities only."""

    return _build_summary(result)


def write_summary_yaml(
    path: str | Path,
    result: SimulationResult,
) -> Path:
    """Write the stable human-readable result summary."""

    return _write_summary_yaml(path, result)


def write_result(result: SimulationResult, output_dir: str | Path) -> ResultPaths:
    """Publish a complete new run with one directory rename; never overwrite a run."""

    directory = Path(output_dir).resolve()
    requests = OutputConfig.model_validate(
        to_plain_mapping(result.metadata.get("output_requests", {}))
    )
    preflight_output(requests)
    files: list[Path] = []
    with new_output_directory(directory) as stage:
        write_result_h5(stage / RESULT_H5_NAME, result)
        write_summary_yaml(stage / SUMMARY_YAML_NAME, result)
        if requests.csv:
            write_result_csv(stage / RESULT_CSV_NAME, result)
            files.append(write_series_yaml(stage / "series.yaml", result))
        if requests.plots:
            files.extend(
                write_configured_plots(result, stage / "plots", requests.plots)
            )
        if requests.networks:
            files.extend(write_networks(result, stage / "networks", requests.networks))
        relative = tuple(path.relative_to(stage) for path in files)
    return ResultPaths(
        directory,
        directory / RESULT_H5_NAME,
        directory / SUMMARY_YAML_NAME,
        directory / RESULT_CSV_NAME if requests.csv else None,
        tuple(directory / path for path in relative),
    )


def preflight_output(requests: OutputConfig) -> None:
    """Fail before integration when a requested renderer is unavailable."""
    if requests.plots:
        _load_pyplot()
    if requests.networks:
        graphviz_module(
            render=any(config.format != "dot" for config in requests.networks.values())
        )


def network_result_h5(
    result_h5: str | Path,
    output_dir: str | Path,
    *,
    config: NetworkConfig | None = None,
    image_format: str | None = None,
) -> tuple[Path, ...]:
    result = read_result_h5(result_h5)
    configured = OutputConfig.model_validate(
        to_plain_mapping(result.metadata.get("output_requests", {}))
    ).networks
    requests = (
        dict(configured)
        if config is None and configured
        else {"network": config or NetworkConfig()}
    )
    if image_format is not None:
        requests = {
            name: NetworkConfig.model_validate(
                {**item.model_dump(), "format": image_format}
            )
            for name, item in requests.items()
        }
    graphviz_module(render=any(item.format != "dot" for item in requests.values()))
    directory = Path(output_dir).resolve()
    with new_output_directory(directory) as stage:
        files = write_networks(result, stage / "diagrams", requests)
        relative = tuple(path.relative_to(stage) for path in files)
    return tuple(directory / path for path in relative)


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
    image_format: str | None = None,
    dpi: int | None = None,
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
    "network_result_h5",
    "plot_result_h5",
    "read_result_csv",
    "read_result_h5",
    "write_result",
    "write_result_csv",
    "write_result_h5",
    "write_summary_yaml",
]
