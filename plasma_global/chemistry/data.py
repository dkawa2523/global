"""Canonical, strict chemistry input for schema version 3.

Raw third-party chemistry formats belong in :mod:`tools.importers`.  The
runtime deliberately accepts one small SI representation and never repairs
or guesses malformed scientific data.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global._yaml import YamlLoadError, load_unique_yaml
from plasma_global.chemistry._contracts import cross_section_support_error
from plasma_global.errors import ChemistryError

AMU_TO_KG = 1.66053906660e-27


def _empty_metadata() -> Mapping[str, Any]:
    return MappingProxyType({})


_TERM = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s+)?([^\s].*?)\s*$")


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = load_unique_yaml(stream)
    except (OSError, YamlLoadError) as exc:
        raise ChemistryError(
            f"cannot read canonical chemistry YAML {path}: {exc}"
        ) from exc
    return _mapping(document or {}, str(path))


def _resolved_reaction_electron_energy_transfer(
    reaction: ReactionData,
) -> float | None:
    """Return signed electron gain, migrating legacy positive loss to a sink."""

    if reaction.electron_energy_transfer_eV is not None:
        return reaction.electron_energy_transfer_eV
    if reaction.energy_loss_eV is None:
        return None
    return -reaction.energy_loss_eV


def _resolved_cross_section_electron_energy_transfer(
    cross_section: CrossSectionData,
) -> float:
    """Return signed electron gain, migrating legacy positive loss to a sink."""

    if cross_section.electron_energy_transfer_eV is not None:
        return cross_section.electron_energy_transfer_eV
    if cross_section.energy_loss_eV is None:
        raise ChemistryError(
            f"cross section {cross_section.id!r} has no electron-energy transfer"
        )
    return -cross_section.energy_loss_eV


@dataclass(frozen=True, slots=True)
class SpeciesData:
    id: str
    phase: str
    charge: int
    mass_amu: float
    elements: Mapping[str, float]
    state_tags: frozenset[str] = frozenset()
    surfaces: tuple[str, ...] = ()
    cv_over_kb: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU_TO_KG


@dataclass(frozen=True, slots=True)
class ReactionData:
    id: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_model: str | None
    energy_loss_eV: float | None
    gas_heating_eV: float = 0.0
    zones: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    electron_energy_transfer_eV: float | None = None

    resolved_electron_energy_transfer_eV = property(
        _resolved_reaction_electron_energy_transfer
    )


@dataclass(frozen=True, slots=True)
class RateModelData:
    id: str
    kind: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CrossSectionData:
    id: str
    kind: str
    target: str
    threshold_eV: float
    energy_loss_eV: float | None
    energy_eV: np.ndarray
    sigma_m2: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    electron_energy_transfer_eV: float | None = None

    resolved_electron_energy_transfer_eV = property(
        _resolved_cross_section_electron_energy_transfer
    )


@dataclass(frozen=True, slots=True)
class ChemistryData:
    source: Path
    species: tuple[SpeciesData, ...]
    gas_reactions: tuple[ReactionData, ...]
    boundary_reactions: tuple[ReactionData, ...]
    surface_reactions: tuple[ReactionData, ...]
    rate_models: Mapping[str, RateModelData]
    cross_sections: Mapping[str, CrossSectionData]
    experimental: Mapping[str, Any] = field(default_factory=_empty_metadata)
    provenance: Mapping[str, Any] = field(default_factory=_empty_metadata)
    source_files: tuple[Path, ...] = ()


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChemistryError(f"{where} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _only_keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ChemistryError(f"{where} contains unknown keys: {', '.join(unknown)}")


def _required_path(base: Path, value: object, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ChemistryError(f"{where} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise ChemistryError(f"{where} does not exist: {path}")
    return path


def _optional_path(base: Path, value: object, where: str) -> Path | None:
    if value is None:
        return None
    return _required_path(base, value, where)


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


def parse_equation(equation: str) -> tuple[Mapping[str, float], Mapping[str, float]]:
    """Parse one irreversible equation without aliases or implicit species."""

    if equation.count("->") != 1:
        raise ChemistryError(f"reaction equation must contain one '->': {equation!r}")

    def side(text: str) -> Mapping[str, float]:
        values: dict[str, float] = {}
        for token in (part.strip() for part in text.split("+")):
            if not token:
                continue
            match = _TERM.match(token)
            if match is None:
                raise ChemistryError(f"invalid reaction term: {token!r}")
            coefficient_text, species_id = match.groups()
            coefficient = float(coefficient_text) if coefficient_text else 1.0
            species_id = species_id.strip()
            if coefficient <= 0.0 or not np.isfinite(coefficient) or not species_id:
                raise ChemistryError(f"invalid reaction term: {token!r}")
            values[species_id] = values.get(species_id, 0.0) + coefficient
        return MappingProxyType(values)

    lhs, rhs = equation.split("->")
    reactants, products = side(lhs), side(rhs)
    if not reactants:
        raise ChemistryError(f"reaction has no reactants: {equation!r}")
    return reactants, products


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
    """Select the only populated legacy or signed energy field."""

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
    """Parse one selected energy field without changing its sign convention."""

    try:
        value = float(values[name])
    except (TypeError, ValueError) as exc:
        raise ChemistryError(f"{where}.{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ChemistryError(f"{where}.{name} must be a finite number")
    return value


def _electron_energy_values(
    values: Mapping[str, Any],
    where: str,
    *,
    required: bool,
) -> tuple[float | None, float | None]:
    """Parse the legacy loss or canonical signed electron-energy transfer."""

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


def _load_species(path: Path) -> tuple[SpeciesData, ...]:
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


def _load_reactions(
    path: Path | None, *, boundary: bool = False
) -> tuple[ReactionData, ...]:
    if path is None:
        return ()
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
        loss, transfer = _electron_energy_values(row, f"{path}:{line}", required=False)
        gas_heating = float(row["gas_heating_eV"]) if row.get("gas_heating_eV") else 0.0
        if not np.isfinite(gas_heating) or gas_heating < 0.0:
            raise ChemistryError(
                f"{path}:{line}: gas_heating_eV must be finite and nonnegative"
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


_RATE_KEYS: dict[str, set[str]] = {
    "electron_impact": {"cross_section", "branching_yield"},
    "arrhenius": {"A", "beta", "activation_eV"},
    "constant": {"value"},
    "first_order": {"rate_s_inv"},
    "experimental.electron_temperature_power_law": {
        "A",
        "reference_temperature_K",
        "exponent",
    },
    "tabulated_1d": {"axis", "file", "bounds"},
    "sticking": {"value", "coverage"},
    "ion_assisted": {
        "yield",
        "threshold_eV",
        "reference_energy_eV",
        "exponent",
        "coverage",
    },
    "desorption": {"frequency_s_inv", "activation_eV"},
    "langmuir_hinshelwood": {"A_m2_s_inv", "activation_eV"},
}

_RATE_REQUIRED: dict[str, set[str]] = {
    "electron_impact": {"cross_section", "branching_yield"},
    "arrhenius": {"A", "beta", "activation_eV"},
    "constant": {"value"},
    "first_order": {"rate_s_inv"},
    "experimental.electron_temperature_power_law": {
        "A",
        "reference_temperature_K",
        "exponent",
    },
    "tabulated_1d": {"axis", "file", "bounds"},
    "sticking": {"value"},
    "ion_assisted": {
        "yield",
        "threshold_eV",
        "reference_energy_eV",
        "exponent",
    },
    "desorption": {"frequency_s_inv", "activation_eV"},
    "langmuir_hinshelwood": {"A_m2_s_inv", "activation_eV"},
}


def _finite_number(
    model: Mapping[str, Any], name: str, where: str, *, minimum: float | None = None
) -> float:
    value = model[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChemistryError(f"{where}.{name} must be a number")
    number = float(value)
    if not np.isfinite(number) or (minimum is not None and number < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum:g}"
        raise ChemistryError(f"{where}.{name} must be {suffix}")
    return number


def _validate_coverage(value: object, where: str) -> None:
    if value is None:
        return
    config = _mapping(value, where)
    kind = config.get("kind")
    if kind == "constant":
        _only_keys(config, {"kind"}, where)
        return
    if kind not in {"site_blocking", "species_power"}:
        raise ChemistryError(f"{where}.kind is unsupported: {kind!r}")
    species_key = "site_species" if kind == "site_blocking" else "species"
    _only_keys(config, {"kind", species_key, "exponent"}, where)
    missing = [name for name in (species_key, "exponent") if name not in config]
    if missing:
        raise ChemistryError(f"{where} is missing: {', '.join(missing)}")
    if not isinstance(config[species_key], str) or not config[species_key].strip():
        raise ChemistryError(f"{where}.{species_key} must be a non-empty species ID")
    _finite_number(config, "exponent", where, minimum=0.0)


def _validate_tabulated_rate_parameters(model: Mapping[str, Any], where: str) -> None:
    if model["axis"] not in {
        "gas_temperature_K",
        "mean_energy_eV",
        "electron_temperature_eV",
        "reduced_field_Td",
        "pressure_Pa",
    }:
        raise ChemistryError(f"{where}.axis is unsupported: {model['axis']!r}")
    if model["bounds"] not in {"error", "clip"}:
        raise ChemistryError(f"{where}.bounds must be error or clip")


def _validate_gas_rate_parameters(
    kind: str, model: Mapping[str, Any], where: str
) -> None:
    if kind == "electron_impact":
        if (
            not isinstance(model["cross_section"], str)
            or not model["cross_section"].strip()
        ):
            raise ChemistryError(f"{where}.cross_section must be a non-empty ID")
        _finite_number(model, "branching_yield", where, minimum=0.0)
    elif kind == "arrhenius":
        _finite_number(model, "A", where, minimum=0.0)
        _finite_number(model, "beta", where)
        _finite_number(model, "activation_eV", where, minimum=0.0)
    elif kind == "constant":
        _finite_number(model, "value", where, minimum=0.0)
    elif kind == "first_order":
        _finite_number(model, "rate_s_inv", where, minimum=0.0)
    elif kind == "experimental.electron_temperature_power_law":
        _finite_number(model, "A", where, minimum=0.0)
        reference = _finite_number(model, "reference_temperature_K", where, minimum=0.0)
        if reference == 0.0:
            raise ChemistryError(f"{where}.reference_temperature_K must be positive")
        _finite_number(model, "exponent", where)
    elif kind == "tabulated_1d":
        _validate_tabulated_rate_parameters(model, where)


def _validate_surface_rate_parameters(
    kind: str, model: Mapping[str, Any], where: str
) -> None:
    if kind == "sticking":
        value = _finite_number(model, "value", where, minimum=0.0)
        if value > 1.0:
            raise ChemistryError(f"{where}.value must not exceed one")
        _validate_coverage(model.get("coverage"), f"{where}.coverage")
    elif kind == "ion_assisted":
        _finite_number(model, "yield", where, minimum=0.0)
        threshold = _finite_number(model, "threshold_eV", where, minimum=0.0)
        reference = _finite_number(model, "reference_energy_eV", where, minimum=0.0)
        if reference <= threshold:
            raise ChemistryError(
                f"{where}.reference_energy_eV must exceed threshold_eV"
            )
        _finite_number(model, "exponent", where, minimum=0.0)
        _validate_coverage(model.get("coverage"), f"{where}.coverage")
    elif kind in {"desorption", "langmuir_hinshelwood"}:
        amplitude = "frequency_s_inv" if kind == "desorption" else "A_m2_s_inv"
        _finite_number(model, amplitude, where, minimum=0.0)
        _finite_number(model, "activation_eV", where, minimum=0.0)


def _validate_rate_parameters(kind: str, model: Mapping[str, Any], where: str) -> None:
    missing = sorted(_RATE_REQUIRED[kind] - set(model))
    if missing:
        raise ChemistryError(f"{where} is missing: {', '.join(missing)}")
    if kind in {
        "electron_impact",
        "arrhenius",
        "constant",
        "first_order",
        "experimental.electron_temperature_power_law",
        "tabulated_1d",
    }:
        _validate_gas_rate_parameters(kind, model, where)
    else:
        _validate_surface_rate_parameters(kind, model, where)


def _load_rate_models(
    path: Path, source_files: list[Path]
) -> Mapping[str, RateModelData]:
    raw = _yaml_mapping(path)
    _only_keys(raw, {"rate_models"}, str(path))
    models = _mapping(raw.get("rate_models"), f"{path}.rate_models")
    result: dict[str, RateModelData] = {}
    for model_id, value in models.items():
        model = _mapping(value, f"{path}.rate_models.{model_id}")
        kind = model.pop("kind", None)
        if kind not in _RATE_KEYS:
            raise ChemistryError(
                f"{path}: rate model {model_id!r} has unsupported kind {kind!r}"
            )
        _only_keys(model, _RATE_KEYS[kind], f"{path}.rate_models.{model_id}")
        _validate_rate_parameters(kind, model, f"{path}.rate_models.{model_id}")
        if "file" in model:
            model["file"] = _required_path(
                path.parent, model["file"], f"rate model {model_id}.file"
            )
            source_files.append(model["file"])
        result[model_id] = RateModelData(model_id, kind, MappingProxyType(model))
    return MappingProxyType(result)


def _strict_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ChemistryError(
            f"{path} must contain at least two rows and exactly two numeric columns"
        )
    energy, sigma = np.asarray(values[:, 0], float), np.asarray(values[:, 1], float)
    if not np.all(np.isfinite(values)):
        raise ChemistryError(f"{path} contains non-finite values")
    if np.any(energy < 0.0) or np.any(np.diff(energy) <= 0.0):
        raise ChemistryError(
            f"{path} energy must be nonnegative, strictly increasing, and unique"
        )
    if np.any(sigma < 0.0):
        raise ChemistryError(f"{path} contains negative cross sections")
    energy.setflags(write=False)
    sigma.setflags(write=False)
    return energy, sigma


_CROSS_SECTION_KINDS = frozenset(
    {
        "momentum_transfer",
        "attachment",
        "dissociation",
        "deexcitation",
        "excitation",
        "ionization",
    }
)


def _cross_section_kind(entry: Mapping[str, Any], cross_section_id: str) -> str:
    kind = str(entry["kind"])
    if kind not in _CROSS_SECTION_KINDS:
        raise ChemistryError(
            f"cross section {cross_section_id} has unsupported kind {kind!r}"
        )
    return kind


def _cross_section_threshold(entry: Mapping[str, Any], cross_section_id: str) -> float:
    threshold = float(entry["threshold_eV"])
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ChemistryError(f"cross section {cross_section_id} has invalid threshold")
    return threshold


def _load_cross_sections(
    path: Path | None, source_files: list[Path]
) -> Mapping[str, CrossSectionData]:
    if path is None:
        return MappingProxyType({})
    raw = _yaml_mapping(path)
    _only_keys(raw, {"cross_sections"}, str(path))
    entries = raw.get("cross_sections")
    if not isinstance(entries, list):
        raise ChemistryError(f"{path}.cross_sections must be a list")
    result: dict[str, CrossSectionData] = {}
    allowed = {
        "id",
        "kind",
        "target",
        "threshold_eV",
        "energy_loss_eV",
        "electron_energy_transfer_eV",
        "file",
        "metadata",
    }
    for index, value in enumerate(entries):
        entry = _mapping(value, f"{path}.cross_sections[{index}]")
        _only_keys(entry, allowed, f"{path}.cross_sections[{index}]")
        missing = [
            key
            for key in (
                "id",
                "kind",
                "target",
                "threshold_eV",
                "file",
            )
            if key not in entry or entry[key] in {None, ""}
        ]
        if missing:
            raise ChemistryError(
                f"{path}.cross_sections[{index}] missing: {', '.join(missing)}"
            )
        cross_section_id = str(entry["id"])
        if cross_section_id in result:
            raise ChemistryError(f"duplicate cross-section id: {cross_section_id}")
        curve_path = _required_path(
            path.parent, entry["file"], f"cross section {cross_section_id}.file"
        )
        source_files.append(curve_path)
        energy, sigma = _strict_curve(curve_path)
        kind = _cross_section_kind(entry, cross_section_id)
        threshold = _cross_section_threshold(entry, cross_section_id)
        loss, transfer = _electron_energy_values(
            entry, f"cross section {cross_section_id}", required=True
        )
        metadata = _mapping(
            entry.get("metadata", {}), f"cross section {cross_section_id}.metadata"
        )
        cross_section = CrossSectionData(
            id=cross_section_id,
            kind=kind,
            target=str(entry["target"]),
            threshold_eV=threshold,
            energy_loss_eV=loss,
            energy_eV=energy,
            sigma_m2=sigma,
            metadata=MappingProxyType(metadata),
            electron_energy_transfer_eV=transfer,
        )
        support_error = cross_section_support_error(cross_section)
        if support_error is not None:
            raise ChemistryError(support_error)
        result[cross_section_id] = cross_section
    return MappingProxyType(result)


def load_chemistry(path: str | Path) -> ChemistryData:
    """Load a canonical v3 chemistry manifest and all referenced SI data."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ChemistryError(f"chemistry manifest does not exist: {source}")
    raw = _yaml_mapping(source)
    allowed = {
        "schema_version",
        "species",
        "gas_reactions",
        "boundary_reactions",
        "surface_reactions",
        "rate_models",
        "cross_sections",
        "experimental",
        "provenance",
    }
    _only_keys(raw, allowed, str(source))
    if raw.get("schema_version") != 3:
        raise ChemistryError(f"{source}: chemistry schema_version must be 3")
    base = source.parent
    species_path = _required_path(base, raw.get("species"), "chemistry.species")
    gas_path = _required_path(base, raw.get("gas_reactions"), "chemistry.gas_reactions")
    rate_path = _required_path(base, raw.get("rate_models"), "chemistry.rate_models")
    boundary_path = _optional_path(
        base, raw.get("boundary_reactions"), "chemistry.boundary_reactions"
    )
    surface_path = _optional_path(
        base, raw.get("surface_reactions"), "chemistry.surface_reactions"
    )
    cross_section_path = _optional_path(
        base, raw.get("cross_sections"), "chemistry.cross_sections"
    )
    experimental = _mapping(raw.get("experimental", {}), "chemistry.experimental")
    provenance = _mapping(raw.get("provenance", {}), "chemistry.provenance")
    source_files = [source, species_path, gas_path, rate_path]
    source_files.extend(
        path
        for path in (boundary_path, surface_path, cross_section_path)
        if path is not None
    )
    return ChemistryData(
        source=source,
        species=_load_species(species_path),
        gas_reactions=_load_reactions(gas_path),
        boundary_reactions=_load_reactions(boundary_path, boundary=True),
        surface_reactions=_load_reactions(surface_path),
        rate_models=_load_rate_models(rate_path, source_files),
        cross_sections=_load_cross_sections(cross_section_path, source_files),
        experimental=MappingProxyType(experimental),
        provenance=MappingProxyType(provenance),
        source_files=tuple(dict.fromkeys(source_files)),
    )


__all__ = [
    "AMU_TO_KG",
    "ChemistryData",
    "CrossSectionData",
    "RateModelData",
    "ReactionData",
    "SpeciesData",
    "load_chemistry",
    "parse_equation",
]
