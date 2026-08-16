"""Strict CSV decoding for canonical chemistry inputs.

This module translates already-resolved CSV files into the public chemistry
data objects.  Manifest traversal and YAML validation live in
``_data_manifest``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from plasma_global.chemistry._contracts import cross_section_data_error
from plasma_global.chemistry.data import ReactionData, SpeciesData, parse_equation
from plasma_global.errors import ChemistryError


def _split(value: str | None, delimiter: str = "|") -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(delimiter) if item.strip())


def _number_map(value: str | None, where: str) -> Mapping[str, float]:
    if value is None or not value.strip():
        return MappingProxyType({})
    result: dict[str, float] = {}
    for token in value.split(";"):
        if not token.strip() or ":" not in token:
            raise ChemistryError(
                f"{where} must use name:value entries separated by ';'"
            )
        name, raw_number = token.split(":", 1)
        name = name.strip()
        if not name or name in result:
            raise ChemistryError(
                f"{where} contains an empty or duplicate name: {name!r}"
            )
        number = float(raw_number)
        if not np.isfinite(number) or number <= 0.0:
            raise ChemistryError(f"{where}.{name} must be finite and positive")
        result[name] = number
    return MappingProxyType(result)


def _csv_rows(
    path: Path, *, required: set[str], optional: set[str]
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if len(set(fieldnames)) != len(fieldnames):
            raise ChemistryError(f"{path} contains duplicate CSV columns")
        columns = set(fieldnames)
        missing = sorted(required - columns)
        unknown = sorted(columns - required - optional)
        if missing:
            raise ChemistryError(f"{path} is missing columns: {', '.join(missing)}")
        if unknown:
            raise ChemistryError(
                f"{path} contains unknown columns: {', '.join(unknown)}"
            )
        rows: list[dict[str, str]] = []
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ChemistryError(f"{path}:{line} contains extra CSV fields")
            rows.append({str(key): (value or "").strip() for key, value in row.items()})
        return rows


def _metadata(row: Mapping[str, str]) -> Mapping[str, Any]:
    result = {
        name: row[name]
        for name in ("display_name", "notes", "provenance")
        if row.get(name)
    }
    return MappingProxyType(result)


_ELECTRON_ENERGY_FIELDS = (
    "energy_loss_eV",
    "electron_energy_transfer_eV",
)


def _electron_energy_field(
    values: Mapping[str, Any],
    where: str,
    *,
    required: bool,
) -> str | None:
    present = tuple(
        name for name in _ELECTRON_ENERGY_FIELDS if values.get(name) not in {None, ""}
    )
    if len(present) > 1 or (required and len(present) != 1):
        raise ChemistryError(
            f"{where} must define exactly one of energy_loss_eV or "
            "electron_energy_transfer_eV"
        )
    return present[0] if present else None


def _finite_electron_energy_value(
    values: Mapping[str, Any], where: str, name: str
) -> float:
    try:
        value = float(values[name])
    except (TypeError, ValueError) as exc:
        raise ChemistryError(f"{where}.{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ChemistryError(f"{where}.{name} must be a finite number")
    return value


def electron_energy_values(
    values: Mapping[str, Any],
    where: str,
    *,
    required: bool,
) -> tuple[float | None, float | None]:
    """Parse either the legacy loss or canonical signed energy transfer."""

    name = _electron_energy_field(values, where, required=required)
    if name is None:
        return None, None
    value = _finite_electron_energy_value(values, where, name)
    if name == "energy_loss_eV":
        if value < 0.0:
            raise ChemistryError(
                f"{where}.energy_loss_eV must be finite and nonnegative"
            )
        return value, None
    return None, value


def load_species(path: Path) -> tuple[SpeciesData, ...]:
    rows = _csv_rows(
        path,
        required={"id", "phase", "charge", "mass_amu", "elements"},
        optional={
            "state_tags",
            "surfaces",
            "cv_over_kb",
            "display_name",
            "notes",
            "provenance",
        },
    )
    result: list[SpeciesData] = []
    for line, row in enumerate(rows, start=2):
        species_id = row["id"]
        if not species_id:
            raise ChemistryError(f"{path}:{line}: species id is empty")
        phase = row["phase"]
        if phase not in {"gas", "surface"}:
            raise ChemistryError(f"{path}:{line}: phase must be gas or surface")
        charge, mass = int(row["charge"]), float(row["mass_amu"])
        if not np.isfinite(mass) or mass < 0.0:
            raise ChemistryError(
                f"{path}:{line}: mass_amu must be finite and nonnegative"
            )
        cv_over_kb = float(row["cv_over_kb"]) if row.get("cv_over_kb") else None
        if cv_over_kb is not None and (
            not np.isfinite(cv_over_kb) or cv_over_kb <= 0.0
        ):
            raise ChemistryError(
                f"{path}:{line}: cv_over_kb must be finite and positive"
            )
        result.append(
            SpeciesData(
                id=species_id,
                phase=phase,
                charge=charge,
                mass_amu=mass,
                elements=_number_map(row.get("elements"), f"{path}:{line}:elements"),
                state_tags=frozenset(_split(row.get("state_tags"))),
                surfaces=_split(row.get("surfaces")),
                cv_over_kb=cv_over_kb,
                metadata=_metadata(row),
            )
        )
    return tuple(result)


def _validate_reaction_energy_fields(
    *,
    family: Literal["gas", "boundary", "surface"],
    where: str,
    legacy_loss_eV: float | None,
    signed_transfer_eV: float | None,
    gas_heating_eV: float,
) -> None:
    nonzero_electron_transfer = any(
        value is not None and value != 0.0
        for value in (legacy_loss_eV, signed_transfer_eV)
    )
    if family != "gas" and nonzero_electron_transfer:
        raise ChemistryError(
            f"{where}: {family} reactions do not support electron-energy fields"
        )
    if family == "boundary" and gas_heating_eV != 0.0:
        raise ChemistryError(
            f"{where}: boundary reactions do not support gas_heating_eV"
        )


def load_reactions(
    path: Path | None, *, family: Literal["gas", "boundary", "surface"] = "gas"
) -> tuple[ReactionData, ...]:
    if path is None:
        return ()
    boundary = family == "boundary"
    required = {"id", "equation"} if boundary else {"id", "equation", "rate_model"}
    rows = _csv_rows(
        path,
        required=required,
        optional={
            "energy_loss_eV",
            "electron_energy_transfer_eV",
            "gas_heating_eV",
            "zones",
            "surfaces",
            "notes",
            "provenance",
        },
    )
    result: list[ReactionData] = []
    for line, row in enumerate(rows, start=2):
        reactants, products = parse_equation(row["equation"])
        loss, transfer = electron_energy_values(row, f"{path}:{line}", required=False)
        gas_heating = float(row["gas_heating_eV"]) if row.get("gas_heating_eV") else 0.0
        if not np.isfinite(gas_heating) or gas_heating < 0.0:
            raise ChemistryError(
                f"{path}:{line}: gas_heating_eV must be finite and nonnegative"
            )
        _validate_reaction_energy_fields(
            family=family,
            where=f"{path}:{line}",
            legacy_loss_eV=loss,
            signed_transfer_eV=transfer,
            gas_heating_eV=gas_heating,
        )
        result.append(
            ReactionData(
                id=row["id"],
                reactants=reactants,
                products=products,
                rate_model=None if boundary else row["rate_model"],
                energy_loss_eV=loss,
                gas_heating_eV=gas_heating,
                zones=_split(row.get("zones")),
                surfaces=_split(row.get("surfaces")),
                metadata=_metadata(row),
                electron_energy_transfer_eV=transfer,
            )
        )
    return tuple(result)


def _cross_section_rows(path: Path) -> list[tuple[float, float]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header != ["energy_eV", "sigma_m2"]:
                raise ChemistryError(
                    f"{path} must use the exact header energy_eV,sigma_m2"
                )
            rows: list[tuple[float, float]] = []
            for line, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ChemistryError(
                        f"{path}:{line} must contain exactly two numeric fields"
                    )
                rows.append((float(row[0]), float(row[1])))
    except (OSError, ValueError) as exc:
        raise ChemistryError(
            f"cannot read two-column SI cross section {path}: {exc}"
        ) from exc
    return rows


def load_cross_section_curve(
    path: Path,
    *,
    cross_section_id: object,
    kind: object,
    target: object,
    threshold_eV: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one exact two-column SI cross-section curve."""

    values = np.asarray(_cross_section_rows(path), dtype=float).reshape((-1, 2))
    energy, sigma = np.asarray(values[:, 0], float), np.asarray(values[:, 1], float)
    data_error = cross_section_data_error(
        cross_section_id=cross_section_id,
        kind=kind,
        target=target,
        threshold_eV=threshold_eV,
        energy_eV=energy,
        sigma_m2=sigma,
        where=f"{path} cross section {cross_section_id!r}",
    )
    if data_error is not None:
        raise ChemistryError(data_error)
    energy.setflags(write=False)
    sigma.setflags(write=False)
    return energy, sigma
