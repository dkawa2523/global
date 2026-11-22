"""Strict CSV decoding for canonical chemistry inputs.

This module translates already-resolved CSV files into the public chemistry
data objects.  Manifest traversal and YAML validation live in
``_data_manifest``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from plasma_global._input_sources import InputSourceStore
from plasma_global.chemistry._contracts import (
    cross_section_data_error,
    validate_reaction_energy_fields,
)
from plasma_global.chemistry.data import ReactionData, SpeciesData, parse_equation
from plasma_global.errors import ChemistryError


def _split(value: str | None, delimiter: str = "|") -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(delimiter) if item.strip())


def _selectors(value: str | None) -> tuple[str, ...]:
    return () if value is None or value == "" else tuple(value.split("|"))


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
    path: Path,
    *,
    required: set[str],
    optional: set[str],
    sources: InputSourceStore | None = None,
    role: str,
) -> list[dict[str, str]]:
    source_store = sources if sources is not None else InputSourceStore()
    with io.StringIO(source_store.read_text(path, role), newline="") as stream:
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
            rows.append(
                {str(key): _csv_cell(str(key), value) for key, value in row.items()}
            )
        return rows


def _csv_cell(key: str, value: str | None) -> str:
    text = value or ""
    if key in {"id", "rate_model", "zones", "surfaces"}:
        return text
    return text.strip()


def _metadata(row: Mapping[str, str]) -> Mapping[str, Any]:
    result = {
        name: row[name]
        for name in ("display_name", "notes", "provenance")
        if row.get(name)
    }
    return MappingProxyType(result)


def electron_energy_transfer(
    values: Mapping[str, Any],
    where: str,
    *,
    required: bool,
) -> float | None:
    """Read the signed electron energy deposited per reaction event."""
    name = "electron_energy_transfer_eV"
    if values.get(name) in {None, ""}:
        if required:
            raise ChemistryError(f"{where} must define {name}")
        return None
    try:
        value = float(values[name])
    except (TypeError, ValueError) as exc:
        raise ChemistryError(f"{where}.{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ChemistryError(f"{where}.{name} must be a finite number")
    return value


def load_species(
    path: Path, *, sources: InputSourceStore | None = None
) -> tuple[SpeciesData, ...]:
    rows = _csv_rows(
        path,
        required={"id", "phase", "charge", "mass_amu", "elements"},
        sources=sources,
        role="chemistry.species",
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
        phase = row["phase"]
        charge, mass = int(row["charge"]), float(row["mass_amu"])
        cv_over_kb = float(row["cv_over_kb"]) if row.get("cv_over_kb") else None
        result.append(
            SpeciesData(
                id=species_id,
                phase=phase,
                charge=charge,
                mass_amu=mass,
                elements=_number_map(row.get("elements"), f"{path}:{line}:elements"),
                state_tags=frozenset(_split(row.get("state_tags"))),
                surfaces=_selectors(row.get("surfaces")),
                cv_over_kb=cv_over_kb,
                metadata=_metadata(row),
            )
        )
    return tuple(result)


def load_reactions(
    path: Path | None,
    *,
    family: Literal["gas", "boundary", "surface"] = "gas",
    sources: InputSourceStore | None = None,
) -> tuple[ReactionData, ...]:
    if path is None:
        return ()
    boundary = family == "boundary"
    required = {"id", "equation"} if boundary else {"id", "equation", "rate_model"}
    rows = _csv_rows(
        path,
        required=required,
        sources=sources,
        role=f"chemistry.{family}_reactions",
        optional={
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
        reactants, products = parse_equation(
            row["equation"], allow_wall_absorption=boundary
        )
        transfer = electron_energy_transfer(row, f"{path}:{line}", required=False)
        gas_heating = float(row["gas_heating_eV"]) if row.get("gas_heating_eV") else 0.0
        validate_reaction_energy_fields(
            family=family,
            where=f"{path}:{line}",
            signed_transfer_eV=transfer,
            gas_heating_eV=gas_heating,
        )
        result.append(
            ReactionData(
                id=row["id"],
                reactants=reactants,
                products=products,
                rate_model=None if boundary else row["rate_model"],
                gas_heating_eV=gas_heating,
                zones=_selectors(row.get("zones")),
                surfaces=_selectors(row.get("surfaces")),
                metadata=_metadata(row),
                electron_energy_transfer_eV=transfer,
            )
        )
    return tuple(result)


def _cross_section_rows(
    path: Path, channel_id: object, sources: InputSourceStore | None
) -> list[tuple[float, float]]:
    try:
        store = sources if sources is not None else InputSourceStore()
        with io.StringIO(
            store.read_text(path, "chemistry.cross_section"), newline=""
        ) as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            grouped = header == ["channel_id", "energy_eV", "sigma_m2"]
            if header != ["energy_eV", "sigma_m2"] and not grouped:
                raise ChemistryError(
                    f"{path} must use the exact header energy_eV,sigma_m2"
                )
            rows: list[tuple[float, float]] = []
            for line, row in enumerate(reader, start=2):
                if len(row) != (3 if grouped else 2):
                    raise ChemistryError(
                        f"{path}:{line} must contain exactly two numeric fields"
                    )
                if not grouped or row[0] == channel_id:
                    rows.append((float(row[-2]), float(row[-1])))
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
    sources: InputSourceStore | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one exact two-column SI cross-section curve."""

    values = np.asarray(
        _cross_section_rows(path, cross_section_id, sources), dtype=float
    ).reshape((-1, 2))
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
