"""Translate legacy chemistry species, rates, and reactions to schema v3."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from plasma_global.errors import MigrationError


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a legacy YAML mapping without changing its insertion order."""

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise MigrationError(f"{path} must contain a mapping")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read and trim a legacy CSV while preserving row and column order."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            {str(key): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def write_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write canonical CSV bytes with the migration format's fixed dialect."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write canonical YAML without sorting caller-defined keys."""

    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def convert_species_rows(
    species_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Convert species rows and report explicit legacy heat-capacity inference."""

    converted: list[dict[str, Any]] = []
    inferred_cv: dict[str, float] = {}
    for row in species_rows:
        cv_over_kb, inferred_value = _legacy_cv_over_kb(row)
        if inferred_value is not None:
            inferred_cv[row["canonical_id"]] = inferred_value
        converted.append(
            {
                "id": row["canonical_id"],
                "phase": row["phase"],
                "charge": row.get("charge", "0"),
                "mass_amu": row.get("mass_amu", "0"),
                "elements": row.get("elements", ""),
                "state_tags": row.get("state_tags", ""),
                "surfaces": row.get("surfaces", ""),
                "display_name": row.get("display_name", ""),
                "cv_over_kb": cv_over_kb,
            }
        )
    if not inferred_cv:
        return converted, None
    rendered = ", ".join(
        f"{species_id}={value:g}" for species_id, value in inferred_cv.items()
    )
    warning = (
        "legacy chemistry had no cv_over_kb; the migration boundary made "
        f"the former ideal-gas heat-capacity assumption explicit ({rendered})"
    )
    return converted, warning


def convert_rate_models(
    manifest: Mapping[str, Any], base: Path, target: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Translate rate and event-energy model registries in declaration order."""

    rate_models: dict[str, dict[str, Any]] = {}
    energy_models: dict[str, float] = {}
    for category, filename in (manifest.get("model_files") or {}).items():
        models = read_yaml(base / str(filename)).get("rate_models") or {}
        if category == "energy_loss":
            for model_id, model in models.items():
                if str(model.get("backend")) != "constant_event_loss":
                    raise MigrationError(
                        f"energy model {model_id} is not constant_event_loss"
                    )
                energy_models[str(model_id)] = float(model.get("energy_loss_eV", 0.0))
            continue
        for model_id, model in models.items():
            if model_id in rate_models:
                raise MigrationError(f"duplicate legacy rate model {model_id}")
            rate_models[str(model_id)] = _convert_rate_model(
                str(model_id), dict(model), base, target
            )
    return rate_models, energy_models


def convert_reaction_rows(
    source: Path, energy_models: Mapping[str, float], *, surface: bool
) -> list[dict[str, Any]]:
    """Translate enabled gas or surface reaction rows without reordering them."""

    converted: list[dict[str, Any]] = []
    for row in read_rows(source):
        if not _enabled(row.get("enabled", "true")):
            continue
        energy_key = row.get("energy_model_key", "")
        if energy_key and energy_key not in energy_models:
            raise MigrationError(
                f"reaction {row['reaction_id']} references unknown energy model "
                f"{energy_key}"
            )
        converted.append(
            {
                "id": row["reaction_id"],
                "equation": row["equation"],
                "rate_model": row["rate_model_key"],
                "energy_loss_eV": energy_models.get(energy_key, ""),
                "zones": row.get("zone_filter", ""),
                "surfaces": row.get("surface_filter", "") if surface else "",
                "notes": row.get("notes", ""),
            }
        )
    return converted


def convert_boundary_rows(
    species_rows: list[dict[str, str]],
    boundary_products: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    """Create explicit ion-neutralization boundaries or reject ambiguity."""

    neutral_by_signature = _neutral_species_by_signature(species_rows)
    explicit_boundary = {
        ion: equation.strip() for ion, equation in (boundary_products or {}).items()
    }
    converted: list[dict[str, Any]] = []
    for row in species_rows:
        if row["phase"] != "gas" or int(float(row.get("charge") or 0)) <= 0:
            continue
        signature = _element_signature(row.get("elements", ""))
        candidates = neutral_by_signature.get(signature, [])
        if row["canonical_id"] in explicit_boundary:
            product_side = explicit_boundary[row["canonical_id"]]
        elif len(candidates) == 1:
            product_side = candidates[0]
        else:
            raise MigrationError(
                f"cannot infer one neutral wall product for {row['canonical_id']}; "
                "add boundary_reactions.csv explicitly"
            )
        converted.append(
            {
                "id": f"wall_{row['canonical_id']}_neutralization",
                "equation": f"{row['canonical_id']} -> {product_side}",
                "zones": "",
                "surfaces": "",
            }
        )
    return converted


def _convert_surface_rate_model(backend: str, raw: dict[str, Any]) -> dict[str, Any]:
    if backend == "sticking":
        return {
            "kind": "sticking",
            "value": float(raw.get("sticking_value", 0.0)),
            "coverage": raw.get("coverage_factor", {"kind": "constant"}),
        }
    if backend == "ion_assisted":
        return {
            "kind": "ion_assisted",
            "yield": float(raw.get("yield_value", 0.0)),
            "threshold_eV": float(raw.get("threshold_eV", 0.0)),
            "reference_energy_eV": float(raw.get("reference_energy_eV", 1.0)),
            "exponent": float(raw.get("energy_exponent", 1.0)),
            "coverage": raw.get("coverage_factor", {"kind": "constant"}),
        }
    if backend == "desorption":
        return {
            "kind": "desorption",
            "frequency_s_inv": float(raw.get("nu0_s_inv", 0.0)),
            "activation_eV": float(raw.get("Ea_eV", 0.0)),
        }
    return {
        "kind": "langmuir_hinshelwood",
        "A_m2_s_inv": float(raw.get("A_m2_s_inv", 0.0)),
        "activation_eV": float(raw.get("Ea_eV", 0.0)),
    }


def _convert_tabulated_rate_model(
    model_id: str, raw: dict[str, Any], base: Path, target: Path
) -> dict[str, Any]:
    source = (base / str(raw["file"])).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    axis_column, value_column = str(raw["x_column"]), str(raw["value_column"])
    table_name = f"{model_id}.csv"
    write_rows(
        target / "tables" / table_name,
        ["x", "value"],
        [{"x": row[axis_column], "value": row[value_column]} for row in source_rows],
    )
    return {
        "kind": "tabulated_1d",
        "axis": str(raw["x"]),
        "file": f"tables/{table_name}",
        "bounds": str(raw.get("bounds_policy", "error")),
    }


def _convert_rate_model(
    model_id: str, raw: dict[str, Any], base: Path, target: Path
) -> dict[str, Any]:
    backend = str(raw.get("backend", ""))
    if backend == "electron_impact_xsec":
        return {
            "kind": "electron_impact",
            "cross_section": raw["cross_section_id"],
            **(
                {"branching_yield": float(raw["branching_yield"])}
                if "branching_yield" in raw
                else {}
            ),
        }
    if backend == "arrhenius":
        return {
            "kind": "arrhenius",
            "A": _number(raw.get("A")),
            "beta": _number(raw.get("beta")),
            "activation_eV": _number(raw.get("Ea_eV")),
        }
    if backend == "constant":
        return {"kind": "constant", "value": _number(raw.get("value"))}
    if backend == "first_order_loss":
        return {
            "kind": "first_order",
            "rate_s_inv": _number(raw.get("rate_s_inv", raw.get("value"))),
        }
    if backend in {"te_power_law", "electron_temperature_power_law"}:
        factor = _number(raw.get("electron_temperature_factor"), 2.0 / 3.0)
        if abs(factor - 2.0 / 3.0) > 1.0e-12:
            raise MigrationError(
                f"rate model {model_id} uses a noncanonical electron temperature factor"
            )
        return {
            "kind": "experimental.electron_temperature_power_law",
            "A": _number(raw.get("A", raw.get("value"))),
            "reference_temperature_K": _number(
                raw.get("Tref_K", raw.get("reference_temperature_K")), 1.0
            ),
            "exponent": _number(raw.get("alpha", raw.get("exponent"))),
        }
    if backend in {
        "sticking",
        "ion_assisted",
        "desorption",
        "langmuir_hinshelwood",
    }:
        return _convert_surface_rate_model(backend, raw)
    if backend == "tabulated_1d":
        return _convert_tabulated_rate_model(model_id, raw, base, target)
    raise MigrationError(f"rate model {model_id} uses unsupported backend {backend!r}")


def _legacy_atom_count(row: Mapping[str, str]) -> float:
    atom_count = 0.0
    for token in row.get("elements", "").split(";"):
        token = token.strip()
        if not token:
            continue
        element, separator, count = token.partition(":")
        if not separator or element == "site":
            raise MigrationError(
                f"species {row['canonical_id']} has an unusable legacy element formula"
            )
        atom_count += float(count)
    return atom_count


def _legacy_cv_over_kb(row: Mapping[str, str]) -> tuple[str | float, float | None]:
    explicit = row.get("cv_over_kb", "").strip()
    if explicit:
        return explicit, None
    tags = {tag.strip() for tag in row.get("state_tags", "").split("|")}
    if row.get("phase") != "gas" or "electron" in tags:
        return "", None
    atom_count = _legacy_atom_count(row)
    if atom_count <= 0.0:
        raise MigrationError(
            f"species {row['canonical_id']} needs an explicit cv_over_kb for v3"
        )
    # Legacy v2 did not carry heat capacities. The one-way migrator makes its
    # former ideal-gas assumption explicit; runtime v3 never performs inference.
    value = 1.5 if atom_count == 1.0 else 2.5 if atom_count == 2.0 else 3.0
    return value, value


def _neutral_species_by_signature(
    species_rows: list[dict[str, str]],
) -> dict[tuple[tuple[str, str], ...], list[str]]:
    neutral_by_signature: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for row in species_rows:
        if (
            row["phase"] == "gas"
            and int(float(row.get("charge") or 0)) == 0
            and row.get("elements")
        ):
            signature = _element_signature(row["elements"])
            neutral_by_signature.setdefault(signature, []).append(row["canonical_id"])
    return neutral_by_signature


def _element_signature(value: str) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for part in value.split(";"):
        if not part:
            continue
        pieces = part.split(":", 1)
        if len(pieces) != 2:
            raise MigrationError(f"invalid element entry {part!r}")
        entries.append((pieces[0], pieces[1]))
    return tuple(sorted(entries))


def _enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise MigrationError(f"legacy enabled value is not a strict boolean: {value!r}")
    return normalized in {"true", "1", "yes"}


def _number(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)
