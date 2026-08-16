"""Read and validate immutable electron-kinetics table arrays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from plasma_global.errors import CaseValidationError


@dataclass(frozen=True, slots=True)
class ElectronTableArrays:
    mean_energy_eV: np.ndarray
    mobility_m2_V_s: np.ndarray
    effective_field_Td: np.ndarray
    rate_tables: dict[str, np.ndarray]


def _read_rate_tables(
    group: Any,
    *,
    source: Path,
    required_rate_ids: tuple[str, ...],
    optional_rate_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    missing = sorted(set(required_rate_ids) - set(group))
    if missing:
        raise CaseValidationError(
            f"electron table {source} is missing rate coefficients: "
            f"{', '.join(missing)}"
        )
    selected = tuple(
        dict.fromkeys(
            (
                *required_rate_ids,
                *(name for name in optional_rate_ids if name in group),
            )
        )
    ) or tuple(str(name) for name in group)
    return {name: np.asarray(group[name][:]) for name in selected}


def read_electron_table(
    source: Path,
    *,
    required_rate_ids: tuple[str, ...],
    optional_rate_ids: tuple[str, ...],
) -> ElectronTableArrays:
    """Read datasets from one HDF5 table without applying model semantics."""

    with h5py.File(source, "r") as handle:
        required = {"mean_energy_eV", "mobility_m2_V_s", "effective_field_Td"}
        unknown = sorted(set(handle) - {*required, "rate_coefficients"})
        if unknown:
            raise CaseValidationError(
                f"electron table {source} has unknown datasets: {', '.join(unknown)}"
            )
        missing = sorted(name for name in required if name not in handle)
        if missing:
            raise CaseValidationError(
                f"electron table {source} is missing: {', '.join(missing)}"
            )
        if "rate_coefficients" not in handle:
            raise CaseValidationError(
                f"electron table {source} is missing rate_coefficients"
            )
        return ElectronTableArrays(
            mean_energy_eV=np.asarray(handle["mean_energy_eV"][:]),
            mobility_m2_V_s=np.asarray(handle["mobility_m2_V_s"][:]),
            effective_field_Td=np.asarray(handle["effective_field_Td"][:]),
            rate_tables=_read_rate_tables(
                handle["rate_coefficients"],
                source=source,
                required_rate_ids=required_rate_ids,
                optional_rate_ids=optional_rate_ids,
            ),
        )


def validate_table_modes(lookup: object, bounds: object) -> None:
    if not isinstance(lookup, str) or lookup not in {"mean_energy", "local_field"}:
        raise CaseValidationError("table lookup must be mean_energy or local_field")
    if not isinstance(bounds, str) or bounds not in {"error", "clip"}:
        raise CaseValidationError("table bounds must be error or clip")


def _copied_array(source: Path, name: str, values: object) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CaseValidationError(
            f"electron table {source} dataset {name} must be numeric"
        ) from exc
    invalid_kind = raw.dtype.kind in {"b", "c", "S", "U", "V"}
    invalid_objects = raw.dtype.kind == "O" and any(
        isinstance(value, bool) or not isinstance(value, Real) for value in raw.flat
    )
    if invalid_kind or invalid_objects:
        raise CaseValidationError(
            f"electron table {source} dataset {name} must be numeric"
        )
    try:
        return np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CaseValidationError(
            f"electron table {source} dataset {name} must be numeric"
        ) from exc


def _table_arrays(data: ElectronTableArrays, axis: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "axis": axis,
        "mean_energy_eV": data.mean_energy_eV,
        "mobility_m2_V_s": data.mobility_m2_V_s,
        "effective_field_Td": data.effective_field_Td,
        **{f"rate:{name}": values for name, values in data.rate_tables.items()},
    }


def _validate_shapes(
    source: Path, arrays: Mapping[str, np.ndarray], axis: np.ndarray
) -> None:
    size = axis.size
    if size < 2 or any(
        values.ndim != 1 or values.size != size for values in arrays.values()
    ):
        raise CaseValidationError(
            f"electron table {source} datasets must be equal 1-D arrays with at "
            "least two points"
        )
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise CaseValidationError(f"electron table {source} contains non-finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise CaseValidationError(
            f"electron table {source} lookup axis must be strictly increasing "
            "and unique"
        )


def _validate_values(source: Path, data: ElectronTableArrays) -> None:
    if (
        np.any(data.mean_energy_eV < 0.0)
        or np.any(data.mobility_m2_V_s <= 0.0)
        or np.any(data.effective_field_Td < 0.0)
    ):
        raise CaseValidationError(
            f"electron table {source} contains invalid transport values"
        )
    if any(np.any(values < 0.0) for values in data.rate_tables.values()):
        raise CaseValidationError(f"electron table {source} contains negative rates")


def validated_table_arrays(
    source: Path,
    lookup: str,
    axis_values: object,
    mean_energy_eV: object,
    mobility_m2_V_s: object,
    effective_field_Td: object,
    rate_tables: object,
) -> tuple[np.ndarray, ElectronTableArrays]:
    """Copy, validate, and freeze arrays supplied by either public input path."""

    if not isinstance(rate_tables, Mapping):
        raise CaseValidationError(
            f"electron table {source} rate coefficients must be a mapping"
        )
    if any(not isinstance(name, str) or not name for name in rate_tables):
        raise CaseValidationError(
            f"electron table {source} rate coefficient IDs must be non-empty strings"
        )
    data = ElectronTableArrays(
        mean_energy_eV=_copied_array(source, "mean_energy_eV", mean_energy_eV),
        mobility_m2_V_s=_copied_array(source, "mobility_m2_V_s", mobility_m2_V_s),
        effective_field_Td=_copied_array(
            source, "effective_field_Td", effective_field_Td
        ),
        rate_tables={
            name: _copied_array(source, f"rate:{name}", values)
            for name, values in rate_tables.items()
        },
    )
    axis = _copied_array(source, "axis", axis_values)
    arrays = _table_arrays(data, axis)
    _validate_shapes(source, arrays, axis)
    expected_axis = (
        data.mean_energy_eV if lookup == "mean_energy" else data.effective_field_Td
    )
    if not np.array_equal(axis, expected_axis):
        raise CaseValidationError(
            f"electron table {source} axis must equal its {lookup} coordinate"
        )
    _validate_values(source, data)
    for values in arrays.values():
        values.setflags(write=False)
    return axis, data
