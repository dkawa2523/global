"""Canonical result persistence plus explicit export and inspection helpers."""

from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from plasma_global.audit import AuditReport, audit_result
from plasma_global.core.result import (
    SimulationResult,
    SimulationStatus,
    to_plain_mapping,
)

RESULT_FORMAT = "plasma_global_result"
RESULT_FORMAT_VERSION = 1
RESULT_H5_NAME = "result.h5"
SUMMARY_YAML_NAME = "summary.yaml"
RESULT_CSV_NAME = "result.csv"

_CSV_TIME = "time_s"
_CSV_STATE = "state:"
_CSV_OBSERVABLE = "observable:"
_CSV_SCALAR_OBSERVABLE = "observable_scalar:"

_METADATA_DATASETS = ("effective_case_yaml", "model_ids", "provenance")
_SOLVER_DATASETS = (
    "success",
    "status_code",
    "status_message",
    "statistics_yaml",
)


@dataclass(frozen=True, slots=True)
class ResultPaths:
    """The two canonical artifacts written for every simulation."""

    directory: Path
    result_h5: Path
    summary_yaml: Path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _dataset_options(values: np.ndarray) -> dict[str, Any]:
    return {"compression": "gzip", "shuffle": True} if values.size > 1 else {}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _yaml_text(value: Any) -> str:
    return yaml.safe_dump(_plain(value), sort_keys=True, allow_unicode=True)


def _text_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(
        name,
        data=value,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _required_group(h5: h5py.File, name: str) -> h5py.Group:
    value = h5.get(name)
    if value is None:
        raise ValueError(f"Result HDF5 requires group /{name}")
    if not isinstance(value, h5py.Group):
        raise TypeError(f"Result HDF5 /{name} must be a group")
    return value


def _required_dataset(
    owner: h5py.File | h5py.Group, name: str, full_name: str
) -> h5py.Dataset:
    value = owner.get(name)
    if value is None:
        raise ValueError(f"Result HDF5 requires dataset {full_name}")
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"Result HDF5 {full_name} must be a dataset")
    return value


def _require_exact_children(
    owner: h5py.File | h5py.Group, expected: set[str], path: str
) -> None:
    actual = set(owner)
    if actual != expected:
        raise ValueError(
            f"Result HDF5 {path} layout mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _read_text(group: h5py.Group, name: str, full_name: str) -> str:
    return str(_required_dataset(group, name, full_name).asstr()[()])


def _read_yaml(group: h5py.Group, name: str, full_name: str) -> Any:
    return yaml.safe_load(_read_text(group, name, full_name))


def _metadata_values(result: SimulationResult) -> tuple[str, Any, Any]:
    metadata = to_plain_mapping(result.metadata)
    effective_case_yaml = metadata.get("effective_case_yaml", "")
    if not isinstance(effective_case_yaml, str):
        raise TypeError("result metadata effective_case_yaml must be a string")
    model_ids = metadata.get("model_ids", {})
    provenance = metadata.get("provenance", {})
    if not isinstance(model_ids, dict) or not isinstance(provenance, dict):
        raise TypeError("result metadata model_ids and provenance must be mappings")
    return effective_case_yaml, model_ids, provenance


def write_result_h5(path: str | Path, result: SimulationResult) -> Path:
    """Write the fixed public HDF5 layout."""

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    target = Path(path)
    _ensure_parent(target)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    effective_case_yaml, model_ids, provenance = _metadata_values(result)

    with h5py.File(target, "w") as h5:
        h5.attrs["format"] = RESULT_FORMAT
        h5.attrs["format_version"] = RESULT_FORMAT_VERSION
        h5.attrs["state_layout"] = "time-major"

        time_dataset = h5.create_dataset(
            "time_s", data=result.time_s, **_dataset_options(result.time_s)
        )
        time_dataset.attrs["unit"] = "s"

        state_group = h5.create_group("state")
        state_group.create_dataset(
            "values", data=result.state, **_dataset_options(result.state)
        )
        state_group.create_dataset(
            "labels",
            data=np.asarray(result.state_labels, dtype=object),
            dtype=string_dtype,
        )

        observables_group = h5.create_group("observables")
        for name, values in result.observables.items():
            observables_group.create_dataset(
                name, data=values, **_dataset_options(values)
            )

        metadata_group = h5.create_group("metadata")
        _text_dataset(metadata_group, "effective_case_yaml", effective_case_yaml)
        _text_dataset(metadata_group, "model_ids", _yaml_text(model_ids))
        _text_dataset(metadata_group, "provenance", _yaml_text(provenance))

        solver_group = h5.create_group("solver")
        solver_group.create_dataset("success", data=result.status.success)
        _text_dataset(solver_group, "status_code", result.status.code)
        _text_dataset(solver_group, "status_message", result.status.message)
        _text_dataset(
            solver_group,
            "statistics_yaml",
            _yaml_text(to_plain_mapping(result.solver_stats)),
        )
    return target


def read_result_h5(path: str | Path) -> SimulationResult:
    """Read and validate a canonical result artifact."""

    source = Path(path)
    with h5py.File(source, "r") as h5:
        file_format = h5.attrs.get("format", "")
        if isinstance(file_format, bytes):
            file_format = file_format.decode("utf-8")
        if str(file_format) != RESULT_FORMAT:
            raise ValueError(f"{source} is not a {RESULT_FORMAT} HDF5 file")
        version = int(h5.attrs.get("format_version", -1))
        if version != RESULT_FORMAT_VERSION:
            raise ValueError(f"Unsupported result HDF5 format version {version}")
        if str(h5.attrs.get("state_layout", "")) != "time-major":
            raise ValueError("Result HDF5 state layout must be time-major")
        _require_exact_children(
            h5,
            {"time_s", "state", "observables", "metadata", "solver"},
            "/",
        )

        time_s = np.asarray(
            _required_dataset(h5, "time_s", "/time_s")[...], dtype=float
        )
        state_group = _required_group(h5, "state")
        _require_exact_children(state_group, {"values", "labels"}, "/state")
        state = np.asarray(
            _required_dataset(state_group, "values", "/state/values")[...],
            dtype=float,
        )
        labels = _required_dataset(state_group, "labels", "/state/labels").asstr()[...]
        state_labels = tuple(str(value) for value in labels.tolist())

        observables_group = _required_group(h5, "observables")
        observables: dict[str, np.ndarray] = {}
        for name, dataset in observables_group.items():
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"Result HDF5 /observables/{name} must be a dataset")
            observables[str(name)] = np.asarray(dataset[()], dtype=float)

        metadata_group = _required_group(h5, "metadata")
        _require_exact_children(metadata_group, set(_METADATA_DATASETS), "/metadata")
        for name in _METADATA_DATASETS:
            _required_dataset(metadata_group, name, f"/metadata/{name}")
        model_ids = _read_yaml(metadata_group, "model_ids", "/metadata/model_ids") or {}
        provenance = (
            _read_yaml(metadata_group, "provenance", "/metadata/provenance") or {}
        )
        if not isinstance(model_ids, dict) or not isinstance(provenance, dict):
            raise TypeError("result metadata model_ids and provenance must be mappings")
        metadata = {
            "effective_case_yaml": _read_text(
                metadata_group,
                "effective_case_yaml",
                "/metadata/effective_case_yaml",
            ),
            "model_ids": model_ids,
            "provenance": provenance,
        }

        solver_group = _required_group(h5, "solver")
        _require_exact_children(solver_group, set(_SOLVER_DATASETS), "/solver")
        for name in _SOLVER_DATASETS:
            _required_dataset(solver_group, name, f"/solver/{name}")
        status = SimulationStatus(
            success=bool(solver_group["success"][()]),
            code=_read_text(solver_group, "status_code", "/solver/status_code"),
            message=_read_text(
                solver_group, "status_message", "/solver/status_message"
            ),
        )
        solver_stats = _read_yaml(
            solver_group, "statistics_yaml", "/solver/statistics_yaml"
        )
        if not isinstance(solver_stats, dict):
            raise TypeError("/solver/statistics_yaml must contain a mapping")

    return SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=state_labels,
        observables=observables,
        status=status,
        solver_stats=solver_stats,
        metadata=metadata,
    )


def _string_sequence(metadata: Mapping[str, Any], name: str) -> tuple[str, ...]:
    raw = metadata.get(name, ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError(f"result metadata {name} must be a sequence of names")
    return tuple(dict.fromkeys(str(value) for value in raw))


def _audit_from_metadata(result: SimulationResult) -> AuditReport:
    metadata = to_plain_mapping(result.metadata)
    conservation_names = _string_sequence(metadata, "conservation_observables")
    residuals = {name: result.series(name) for name in conservation_names}
    provenance = metadata.get("provenance", {})
    if provenance is not None and not isinstance(provenance, dict):
        raise TypeError("result metadata provenance must be a mapping")
    runtime = dict(provenance or {}).get("runtime_diagnostics", {})
    if runtime is not None and not isinstance(runtime, dict):
        raise TypeError("result provenance runtime_diagnostics must be a mapping")
    stored_maxima = dict(runtime or {}).get("conservation_max_abs_residual", {})
    if stored_maxima is not None and not isinstance(stored_maxima, dict):
        raise TypeError(
            "runtime diagnostic conservation_max_abs_residual must be a mapping"
        )
    for name, value in dict(stored_maxima or {}).items():
        residuals.setdefault(str(name), np.asarray(float(value)))
    tolerances = metadata.get("conservation_tolerances")
    if tolerances is not None and not isinstance(tolerances, dict):
        raise TypeError("result metadata conservation_tolerances must be a mapping")
    default_tolerance = metadata.get("default_conservation_tolerance")
    return audit_result(
        result,
        conservation_residuals=residuals,
        conservation_tolerances=tolerances,
        default_conservation_tolerance=default_tolerance,
    )


def build_summary(
    result: SimulationResult, *, audit: AuditReport | None = None
) -> dict[str, Any]:
    """Build the fixed summary from result-declared quantities only."""

    metadata = to_plain_mapping(result.metadata)
    final_names = _string_sequence(metadata, "summary_series")
    report = audit or _audit_from_metadata(result)
    return {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "status": {
            "success": result.status.success,
            "code": result.status.code,
            "message": result.status.message,
        },
        "time": {
            "count": result.n_times,
            "start_s": float(result.time_s[0]) if result.n_times else None,
            "end_s": float(result.time_s[-1]) if result.n_times else None,
        },
        "model_ids": _plain(metadata.get("model_ids", {})),
        "final": {name: result.final_value(name) for name in final_names},
        "conservation": {
            "max_abs_residual": report.max_conservation_residual,
            "max_abs_residual_by_quantity": dict(report.conservation_max_abs_residual),
        },
        "audit": {
            "passed": report.passed,
            "issue_count": len(report.issues),
        },
        "solver": to_plain_mapping(result.solver_stats),
    }


def write_summary_yaml(
    path: str | Path,
    result: SimulationResult,
    *,
    audit: AuditReport | None = None,
) -> Path:
    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            build_summary(result, audit=audit),
            stream,
            sort_keys=False,
            allow_unicode=True,
        )
    return target


def write_result(result: SimulationResult, output_dir: str | Path) -> ResultPaths:
    """Stage both canonical artifacts before replacing either public file."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = _audit_from_metadata(result)
    result_h5 = directory / RESULT_H5_NAME
    summary_yaml = directory / SUMMARY_YAML_NAME
    # Serializing into the destination filesystem first prevents a failed
    # summary (or HDF5) writer from leaving a new half-bundle beside an old
    # counterpart.  ``os.replace`` is atomic for each same-filesystem target.
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
            # A pair cannot be swapped by one filesystem syscall. Restore the
            # previous pair if the second same-filesystem replacement fails.
            for target in reversed(replaced):
                backup = backups[target]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            raise
    return ResultPaths(directory, result_h5, summary_yaml)


def _result_csv_columns(
    result: SimulationResult,
) -> list[tuple[str, np.ndarray, bool]]:
    columns = [
        (f"{_CSV_STATE}{label}", result.state[:, index], False)
        for index, label in enumerate(result.state_labels)
    ]
    for name, values in result.observables.items():
        prefix = _CSV_SCALAR_OBSERVABLE if values.ndim == 0 else _CSV_OBSERVABLE
        columns.append((f"{prefix}{name}", values, values.ndim == 0))
    return columns


def write_result_csv(path: str | Path, result: SimulationResult) -> Path:
    """Export a canonical result to one reversible numeric CSV."""

    if result.n_times == 0 and any(
        values.ndim == 0 for values in result.observables.values()
    ):
        raise ValueError("CSV cannot preserve scalar observables without time rows")
    target = Path(path)
    _ensure_parent(target)
    columns = _result_csv_columns(result)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([_CSV_TIME, *(name for name, _values, _scalar in columns)])
        for row_index, time_s in enumerate(result.time_s):
            row: list[float] = [float(time_s)]
            for _name, values, scalar in columns:
                row.append(float(values) if scalar else float(values[row_index]))
            writer.writerow(row)
    return target


def read_result_csv(
    path: str | Path,
    *,
    status: SimulationStatus | None = None,
    solver_stats: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Read a CSV produced by :func:`write_result_csv`."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Result CSV is empty: {source}") from exc
        rows = list(reader)
    if not header or header[0] != _CSV_TIME:
        raise ValueError(f"Result CSV first column must be {_CSV_TIME!r}")
    if len(set(header)) != len(header):
        raise ValueError("Result CSV column names must be unique")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Result CSV rows do not match its header")
    try:
        values = np.asarray(
            [[float(cell) for cell in row] for row in rows], dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Result CSV contains a non-numeric cell: {source}") from exc
    if not rows:
        values = np.empty((0, len(header)), dtype=float)

    time_s = values[:, 0]
    labels: list[str] = []
    state_columns: list[np.ndarray] = []
    observables: dict[str, np.ndarray] = {}
    for index, column in enumerate(header[1:], start=1):
        data = values[:, index]
        if column.startswith(_CSV_STATE):
            labels.append(column.removeprefix(_CSV_STATE))
            state_columns.append(data)
        elif column.startswith(_CSV_SCALAR_OBSERVABLE):
            name = column.removeprefix(_CSV_SCALAR_OBSERVABLE)
            if data.size == 0 or not np.allclose(data, data[0], equal_nan=True):
                raise ValueError(f"Invalid scalar observable column {name!r}")
            observables[name] = np.asarray(data[0], dtype=float)
        elif column.startswith(_CSV_OBSERVABLE):
            observables[column.removeprefix(_CSV_OBSERVABLE)] = data
        else:
            raise ValueError(f"Unsupported result CSV column {column!r}")
    state = (
        np.column_stack(state_columns)
        if state_columns
        else np.empty((time_s.size, 0), dtype=float)
    )
    return SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=tuple(labels),
        observables=observables,
        status=status
        or SimulationStatus(
            success=True,
            code="imported_csv",
            message=f"Loaded from {source.name}",
        ),
        solver_stats=solver_stats or {},
        metadata=metadata or {},
    )


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
    """Plot selected series; matplotlib is imported only by this operation."""

    result = read_result_h5(result_h5)
    names = tuple(series) or tuple(result.observables) or result.state_labels
    if not names:
        raise ValueError("the result contains no series to plot")
    if result.n_times == 0:
        raise ValueError("an empty result cannot be plotted")
    if not image_format.isalnum() or dpi <= 0:
        raise ValueError("plot format and dpi are invalid")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Plotting requires the optional dependency: pip install .[plot]"
        ) from exc

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, name in enumerate(names):
        values = result.series(name)
        if values.ndim == 0:
            values = np.full(result.n_times, float(values))
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.")
        filename = filename or f"series_{index}"
        path = directory / f"{filename}.{image_format}"
        figure, axes = plt.subplots()
        axes.plot(result.time_s, values)
        axes.set_xlabel("time [s]")
        axes.set_ylabel(name)
        figure.tight_layout()
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


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
