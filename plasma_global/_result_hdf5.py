"""Canonical HDF5 codec for simulation results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from plasma_global._result_common import (
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    ensure_parent,
    series_unit,
    yaml_text,
)
from plasma_global.core.result import (
    SimulationResult,
    SimulationStatus,
    to_plain_mapping,
)

_METADATA_DATASETS = ("effective_case_yaml", "model_ids", "provenance")
_SOLVER_DATASETS = (
    "success",
    "status_code",
    "status_message",
    "statistics_yaml",
)


def _dataset_options(values: np.ndarray) -> dict[str, Any]:
    return {"compression": "gzip", "shuffle": True} if values.size > 1 else {}


def _text_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=value, dtype=h5py.string_dtype(encoding="utf-8"))


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
    ensure_parent(target)
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
        state_values = state_group.create_dataset(
            "values", data=result.state, **_dataset_options(result.state)
        )
        state_values.attrs.create(
            "column_units",
            np.asarray(
                [series_unit(label) for label in result.state_labels], dtype=object
            ),
            dtype=string_dtype,
        )
        state_group.create_dataset(
            "labels",
            data=np.asarray(result.state_labels, dtype=object),
            dtype=string_dtype,
        )
        observables_group = h5.create_group("observables")
        for name, values in result.observables.items():
            dataset = observables_group.create_dataset(
                name, data=values, **_dataset_options(values)
            )
            dataset.attrs["unit"] = series_unit(name)
        metadata_group = h5.create_group("metadata")
        _text_dataset(metadata_group, "effective_case_yaml", effective_case_yaml)
        _text_dataset(metadata_group, "model_ids", yaml_text(model_ids))
        _text_dataset(metadata_group, "provenance", yaml_text(provenance))
        solver_group = h5.create_group("solver")
        solver_group.create_dataset("success", data=result.status.success)
        _text_dataset(solver_group, "status_code", result.status.code)
        _text_dataset(solver_group, "status_message", result.status.message)
        _text_dataset(
            solver_group,
            "statistics_yaml",
            yaml_text(to_plain_mapping(result.solver_stats)),
        )
    return target


def _validate_h5_header(h5: h5py.File, source: Path) -> None:
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
        h5, {"time_s", "state", "observables", "metadata", "solver"}, "/"
    )


def _read_h5_state(
    h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    time_s = np.asarray(_required_dataset(h5, "time_s", "/time_s")[...], dtype=float)
    state_group = _required_group(h5, "state")
    _require_exact_children(state_group, {"values", "labels"}, "/state")
    state = np.asarray(
        _required_dataset(state_group, "values", "/state/values")[...], dtype=float
    )
    labels = _required_dataset(state_group, "labels", "/state/labels").asstr()[...]
    return time_s, state, tuple(str(value) for value in labels.tolist())


def _read_h5_observables(h5: h5py.File) -> dict[str, np.ndarray]:
    group = _required_group(h5, "observables")
    observables: dict[str, np.ndarray] = {}
    for name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Result HDF5 /observables/{name} must be a dataset")
        observables[str(name)] = np.asarray(dataset[()], dtype=float)
    return observables


def _read_h5_metadata(h5: h5py.File) -> dict[str, Any]:
    group = _required_group(h5, "metadata")
    _require_exact_children(group, set(_METADATA_DATASETS), "/metadata")
    for name in _METADATA_DATASETS:
        _required_dataset(group, name, f"/metadata/{name}")
    model_ids = _read_yaml(group, "model_ids", "/metadata/model_ids") or {}
    provenance = _read_yaml(group, "provenance", "/metadata/provenance") or {}
    if not isinstance(model_ids, dict) or not isinstance(provenance, dict):
        raise TypeError("result metadata model_ids and provenance must be mappings")
    return {
        "effective_case_yaml": _read_text(
            group, "effective_case_yaml", "/metadata/effective_case_yaml"
        ),
        "model_ids": model_ids,
        "provenance": provenance,
    }


def _read_h5_solver(
    h5: h5py.File,
) -> tuple[SimulationStatus, Mapping[str, Any]]:
    group = _required_group(h5, "solver")
    _require_exact_children(group, set(_SOLVER_DATASETS), "/solver")
    for name in _SOLVER_DATASETS:
        _required_dataset(group, name, f"/solver/{name}")
    success = bool(_required_dataset(group, "success", "/solver/success")[()])
    status = SimulationStatus(
        success=success,
        code=_read_text(group, "status_code", "/solver/status_code"),
        message=_read_text(group, "status_message", "/solver/status_message"),
    )
    stats = _read_yaml(group, "statistics_yaml", "/solver/statistics_yaml") or {}
    if not isinstance(stats, dict):
        raise TypeError("/solver/statistics_yaml must contain a mapping")
    return status, stats


def read_result_h5(path: str | Path) -> SimulationResult:
    """Read the strict public HDF5 layout."""

    source = Path(path)
    with h5py.File(source, "r") as h5:
        _validate_h5_header(h5, source)
        time_s, state, state_labels = _read_h5_state(h5)
        observables = _read_h5_observables(h5)
        metadata = _read_h5_metadata(h5)
        status, solver_stats = _read_h5_solver(h5)
    return SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=state_labels,
        observables=observables,
        status=status,
        solver_stats=solver_stats,
        metadata=metadata,
    )


__all__ = ["read_result_h5", "write_result_h5"]
