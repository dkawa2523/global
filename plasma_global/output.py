"""Stable facade for canonical result persistence, export, and inspection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

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
from plasma_global._result_plot import plot_result_h5 as _plot_result_h5
from plasma_global._result_plot import write_configured_plots
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
class ArtifactStatus:
    """Outcome of one requested export or rendering operation."""

    name: str
    kind: str
    status: str
    files: tuple[Path, ...] = ()
    message: str = ""

    def to_dict(self, directory: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "files": [str(path.relative_to(directory)) for path in self.files],
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ArtifactReport:
    """Artifact outcomes are independent of numerical simulation status."""

    requests: tuple[ArtifactStatus, ...] = ()
    manifest: Path | None = None

    @property
    def success(self) -> bool:
        return all(item.status == "completed" for item in self.requests)

    @property
    def status(self) -> str:
        if not self.requests:
            return "not_requested"
        return "completed" if self.success else "failed"


@dataclass(frozen=True, slots=True)
class ResultPaths:
    """Numerical result and requested analysis artifacts in one immutable run."""

    directory: Path
    result_h5: Path
    summary_yaml: Path
    result_csv: Path | None = None
    analysis_files: tuple[Path, ...] = ()
    artifacts: ArtifactReport = field(default_factory=ArtifactReport)


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


def _write_artifact(
    result: SimulationResult,
    directory: Path,
    kind: str,
    name: str,
    config: Any,
) -> ArtifactStatus:
    target = directory / kind / name if kind != "csv" else directory / "csv"
    try:
        with new_output_directory(target) as stage:
            if kind == "csv":
                files = (
                    write_result_csv(stage / RESULT_CSV_NAME, result),
                    write_series_yaml(stage / "series.yaml", result),
                )
            elif kind == "plots":
                files = write_configured_plots(
                    result, stage / "figures", {name: config}
                )
            else:
                files = write_networks(result, stage / "diagrams", {name: config})
            relative = tuple(path.relative_to(stage) for path in files)
        return ArtifactStatus(
            name, kind, "completed", tuple(target / p for p in relative)
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return ArtifactStatus(name, kind, "failed", message=str(exc))


def _write_artifacts(
    result: SimulationResult, directory: Path, requests: OutputConfig
) -> list[ArtifactStatus]:
    outcomes: list[ArtifactStatus] = []
    if requests.csv:
        outcomes.append(_write_artifact(result, directory, "csv", "csv", None))
    for name, config in requests.plots.items():
        outcomes.append(_write_artifact(result, directory, "plots", name, config))
    for name, config in requests.networks.items():
        outcomes.append(_write_artifact(result, directory, "networks", name, config))
    return outcomes


def _publish_artifact_manifest(
    directory: Path,
    outcomes: tuple[ArtifactStatus, ...],
) -> ArtifactReport:
    manifest = directory / "artifacts.yaml"
    report = ArtifactReport(outcomes)
    payload = {
        "status": report.status,
        "requests": [item.to_dict(directory) for item in outcomes],
    }
    try:
        with TemporaryDirectory(prefix=".artifacts-", dir=directory) as raw:
            staged = Path(raw) / manifest.name
            with staged.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
            staged.rename(manifest)
    except OSError as exc:
        failure = ArtifactStatus("manifest", "manifest", "failed", message=str(exc))
        return ArtifactReport((*outcomes, failure))
    return ArtifactReport(outcomes, manifest)


def write_result(result: SimulationResult, output_dir: str | Path) -> ResultPaths:
    """Publish numerical data first, then report each requested artifact separately."""
    directory = Path(output_dir).resolve()
    requests = OutputConfig.model_validate(
        to_plain_mapping(result.metadata.get("output_requests", {}))
    )
    with new_output_directory(directory) as stage:
        write_result_h5(stage / RESULT_H5_NAME, result)
        write_summary_yaml(stage / SUMMARY_YAML_NAME, result)
    outcomes = _write_artifacts(result, directory, requests)
    report = _publish_artifact_manifest(directory, tuple(outcomes))
    files = tuple(path for item in outcomes for path in item.files)
    csv = next(
        (
            item.files[0]
            for item in outcomes
            if item.kind == "csv" and item.status == "completed"
        ),
        None,
    )
    return ResultPaths(
        directory,
        directory / RESULT_H5_NAME,
        directory / SUMMARY_YAML_NAME,
        csv,
        tuple(path for path in files if path != csv),
        report,
    )


def _selected_network(
    configured: Mapping[str, NetworkConfig],
    config: NetworkConfig | None,
    name: str | None,
) -> dict[str, NetworkConfig]:
    if name is not None and config is not None:
        raise ValueError("choose a saved network name or an explicit config")
    if name is not None:
        if name not in configured:
            raise ValueError(
                f"unknown saved network {name!r}; available: {sorted(configured)}"
            )
        requests = {name: configured[name]}
    elif config is not None:
        requests = {"network": config}
    elif len(configured) > 1:
        raise ValueError(
            f"multiple saved networks require --name; available: {sorted(configured)}"
        )
    else:
        requests = dict(configured) or {"network": NetworkConfig()}
    return requests


def _network_requests(
    configured: Mapping[str, NetworkConfig],
    config: NetworkConfig | None,
    name: str | None,
    overrides: Mapping[str, Any] | None,
    image_format: str | None,
) -> dict[str, NetworkConfig]:
    requests = _selected_network(configured, config, name)
    changes = dict(overrides or {})
    if image_format is not None:
        changes["format"] = image_format
    requests = {
        identity: NetworkConfig.model_validate({**item.model_dump(), **changes})
        for identity, item in requests.items()
    }
    return requests


def network_result_h5(
    result_h5: str | Path,
    output_dir: str | Path,
    *,
    config: NetworkConfig | None = None,
    image_format: str | None = None,
    name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    result = read_result_h5(result_h5)
    configured = OutputConfig.model_validate(
        to_plain_mapping(result.metadata.get("output_requests", {}))
    ).networks
    requests = _network_requests(configured, config, name, overrides, image_format)
    graphviz_module(render=any(item.format != "dot" for item in requests.values()))
    directory = Path(output_dir).resolve()
    with new_output_directory(directory) as stage:
        files = write_networks(result, stage / "diagrams", requests)
        relative = tuple(path.relative_to(stage) for path in files)
    return tuple(directory / path for path in relative)


def export_result_csv(result_h5: str | Path, output_csv: str | Path) -> Path:
    """Convert canonical HDF5 to the explicit CSV export format."""

    return write_result_csv(output_csv, read_result_h5(result_h5))


def inspect_result_h5(result_h5: str | Path) -> dict[str, Any]:
    """Describe one saved result without loading case files or integrating."""
    result = read_result_h5(result_h5)
    metadata = to_plain_mapping(result.metadata)
    return {
        **build_summary(result),
        "series_catalog": metadata["series_catalog"],
        "provenance": metadata.get("provenance", {}),
        "model_assumptions": metadata.get(
            "model_assumptions", metadata.get("assessment", {}).get("assumptions", [])
        ),
        "default_output_omissions": metadata.get("default_output_omissions", []),
        "failure": {
            key: metadata[key]
            for key in (
                "failure_time_s",
                "requested_end_s",
                "reached_end_s",
                "failure_segment_id",
            )
            if key in metadata
        },
    }


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
    "ArtifactReport",
    "ArtifactStatus",
    "NetworkConfig",
    "ResultPaths",
    "audit_result_h5",
    "build_summary",
    "export_result_csv",
    "inspect_result_h5",
    "network_result_h5",
    "plot_result_h5",
    "read_result_csv",
    "read_result_h5",
    "write_result",
    "write_result_csv",
    "write_result_h5",
    "write_summary_yaml",
]
