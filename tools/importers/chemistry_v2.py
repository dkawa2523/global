"""One-way v2 chemistry conversion to the canonical schema-v3 format."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from plasma_global.errors import MigrationError


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise MigrationError(f"{path} must contain a mapping")
    return value


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            {str(key): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _write_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


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


def _write_canonical_cross_section(
    source: Path,
    destination: Path,
    *,
    segment_index: int | None = None,
) -> None:
    """Convert a legacy SI curve to the runtime's strict two-column form.

    A repeated energy node is removed only when its cross section is exactly
    identical.  Conflicting duplicates, descending axes, negative data, and
    non-finite values are rejected rather than repaired.
    """

    rows = _rows(source)
    if not rows or set(rows[0]) != {"energy_eV", "sigma_m2"}:
        raise MigrationError(
            f"cross section {source} must have exactly energy_eV,sigma_m2 columns"
        )
    segments: list[list[tuple[float, float]]] = [[]]
    previous_energy: float | None = None
    for index, row in enumerate(rows, start=2):
        try:
            energy = float(row["energy_eV"])
            sigma = float(row["sigma_m2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MigrationError(
                f"cross section {source}:{index} must contain numeric SI values"
            ) from exc
        if not math.isfinite(energy) or not math.isfinite(sigma):
            raise MigrationError(f"cross section {source}:{index} is non-finite")
        if energy < 0.0 or sigma < 0.0:
            raise MigrationError(f"cross section {source}:{index} is negative")
        if previous_energy is not None and energy < previous_energy:
            segments.append([])
        segments[-1].append((energy, sigma))
        previous_energy = energy
    if len(segments) > 1 and segment_index is None:
        raise MigrationError(
            f"cross section {source} contains {len(segments)} concatenated axes; "
            "select one explicitly with --cross-section-segments"
        )
    selected_index = 0 if segment_index is None else int(segment_index)
    if selected_index < 0 or selected_index >= len(segments):
        raise MigrationError(
            f"cross section {source} segment {selected_index} is outside "
            f"0..{len(segments) - 1}"
        )
    converted: list[dict[str, float]] = []
    previous_energy = None
    previous_sigma: float | None = None
    for energy, sigma in segments[selected_index]:
        if previous_energy is not None and energy == previous_energy:
            if sigma != previous_sigma:
                raise MigrationError(
                    f"cross section {source} has conflicting values at {energy:g} eV"
                )
            continue
        converted.append({"energy_eV": energy, "sigma_m2": sigma})
        previous_energy, previous_sigma = energy, sigma
    if len(converted) < 2:
        raise MigrationError(f"cross section {source} needs at least two unique nodes")
    _write_rows(destination, ["energy_eV", "sigma_m2"], converted)


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
    if backend == "langmuir_hinshelwood":
        return {
            "kind": "langmuir_hinshelwood",
            "A_m2_s_inv": float(raw.get("A_m2_s_inv", 0.0)),
            "activation_eV": float(raw.get("Ea_eV", 0.0)),
        }
    if backend == "tabulated_1d":
        source = (base / str(raw["file"])).resolve()
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        axis_column, value_column = str(raw["x_column"]), str(raw["value_column"])
        table_name = f"{model_id}.csv"
        _write_rows(
            target / "tables" / table_name,
            ["x", "value"],
            [
                {"x": row[axis_column], "value": row[value_column]}
                for row in source_rows
            ],
        )
        return {
            "kind": "tabulated_1d",
            "axis": str(raw["x"]),
            "file": f"tables/{table_name}",
            "bounds": str(raw.get("bounds_policy", "error")),
        }
    raise MigrationError(f"rate model {model_id} uses unsupported backend {backend!r}")


def convert_v2_chemistry(
    source_manifest: str | Path,
    target_directory: str | Path,
    *,
    boundary_products: Mapping[str, str] | None = None,
    cross_section_segments: Mapping[str, int] | None = None,
    migration_warnings: list[str] | None = None,
) -> Path:
    """Convert one legacy manifest without adding legacy behavior to runtime."""

    source = Path(source_manifest).resolve()
    target = Path(target_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest = _yaml(source)
    base = source.parent

    species_rows = _rows(base / str(manifest["species_file"]))
    inferred_cv: dict[str, float] = {}

    def legacy_cv_over_kb(row: Mapping[str, str]) -> str | float:
        explicit = row.get("cv_over_kb", "").strip()
        if explicit:
            return explicit
        tags = {tag.strip() for tag in row.get("state_tags", "").split("|")}
        if row.get("phase") != "gas" or "electron" in tags:
            return ""
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
        if atom_count <= 0.0:
            raise MigrationError(
                f"species {row['canonical_id']} needs an explicit cv_over_kb for v3"
            )
        # Legacy v2 did not carry heat capacities.  The one-way migrator makes
        # its former ideal-gas assumption explicit: monatomic=3/2, diatomic=5/2,
        # nonlinear polyatomic=3.  Runtime v3 never performs this inference.
        value = 1.5 if atom_count == 1.0 else 2.5 if atom_count == 2.0 else 3.0
        inferred_cv[str(row["canonical_id"])] = value
        return value

    species_out = []
    for row in species_rows:
        species_out.append(
            {
                "id": row["canonical_id"],
                "phase": row["phase"],
                "charge": row.get("charge", "0"),
                "mass_amu": row.get("mass_amu", "0"),
                "elements": row.get("elements", ""),
                "state_tags": row.get("state_tags", ""),
                "surfaces": row.get("surfaces", ""),
                "display_name": row.get("display_name", ""),
                "cv_over_kb": legacy_cv_over_kb(row),
            }
        )
    _write_rows(
        target / "species.csv",
        [
            "id",
            "phase",
            "charge",
            "mass_amu",
            "elements",
            "state_tags",
            "surfaces",
            "display_name",
            "cv_over_kb",
        ],
        species_out,
    )
    if inferred_cv and migration_warnings is not None:
        rendered = ", ".join(
            f"{species_id}={value:g}" for species_id, value in inferred_cv.items()
        )
        migration_warnings.append(
            "legacy chemistry had no cv_over_kb; the migration boundary made "
            f"the former ideal-gas heat-capacity assumption explicit ({rendered})"
        )

    all_models: dict[str, dict[str, Any]] = {}
    energy_models: dict[str, float] = {}
    for category, filename in (manifest.get("model_files") or {}).items():
        models = _yaml(base / str(filename)).get("rate_models") or {}
        if category == "energy_loss":
            for model_id, model in models.items():
                if str(model.get("backend")) != "constant_event_loss":
                    raise MigrationError(
                        f"energy model {model_id} is not constant_event_loss"
                    )
                energy_models[str(model_id)] = float(model.get("energy_loss_eV", 0.0))
            continue
        for model_id, model in models.items():
            if model_id in all_models:
                raise MigrationError(f"duplicate legacy rate model {model_id}")
            all_models[str(model_id)] = _convert_rate_model(
                str(model_id), dict(model), base, target
            )
    (target / "rate_models.yaml").write_text(
        yaml.safe_dump(
            {"rate_models": all_models}, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )

    def convert_reactions(filename: str, output_name: str, surface: bool) -> None:
        result: list[dict[str, Any]] = []
        for row in _rows(base / filename):
            if not _enabled(row.get("enabled", "true")):
                continue
            energy_key = row.get("energy_model_key", "")
            if energy_key and energy_key not in energy_models:
                raise MigrationError(
                    f"reaction {row['reaction_id']} references unknown energy model {energy_key}"
                )
            result.append(
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
        _write_rows(
            target / output_name,
            [
                "id",
                "equation",
                "rate_model",
                "energy_loss_eV",
                "zones",
                "surfaces",
                "notes",
            ],
            result,
        )

    convert_reactions(str(manifest["gas_reactions_file"]), "gas_reactions.csv", False)
    surface_file = manifest.get("surface_reactions_file")
    if surface_file:
        convert_reactions(str(surface_file), "surface_reactions.csv", True)

    neutral_by_signature: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for row in species_rows:
        if (
            row["phase"] == "gas"
            and int(float(row.get("charge") or 0)) == 0
            and row.get("elements")
        ):
            signature = _element_signature(row["elements"])
            neutral_by_signature.setdefault(signature, []).append(row["canonical_id"])
    explicit_boundary = {
        str(ion): str(equation).strip()
        for ion, equation in (boundary_products or {}).items()
    }
    boundary_rows: list[dict[str, Any]] = []
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
                f"cannot infer one neutral wall product for {row['canonical_id']}; add boundary_reactions.csv explicitly"
            )
        boundary_rows.append(
            {
                "id": f"wall_{row['canonical_id']}_neutralization",
                "equation": f"{row['canonical_id']} -> {product_side}",
                "zones": "",
                "surfaces": "",
            }
        )
    _write_rows(
        target / "boundary_reactions.csv",
        ["id", "equation", "zones", "surfaces"],
        boundary_rows,
    )

    cross_section_output = None
    if cross_section_manifest := manifest.get("cross_sections_manifest"):
        cross_source = (base / str(cross_section_manifest)).resolve()
        entries = _yaml(cross_source).get("cross_sections") or []
        converted: list[dict[str, Any]] = []
        for entry in entries:
            if not entry.get("file"):
                raise MigrationError(
                    f"cross section {entry.get('cross_section_id')} has no canonical source file"
                )
            original = (cross_source.parent / str(entry["file"])).resolve()
            destination = target / "cross_sections" / original.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            cross_section_id = str(entry["cross_section_id"])
            selected_segment = (cross_section_segments or {}).get(cross_section_id)
            _write_canonical_cross_section(
                original,
                destination,
                segment_index=selected_segment,
            )
            converted.append(
                {
                    "id": entry["cross_section_id"],
                    "kind": entry.get("kind", "inelastic"),
                    "target": entry.get("target_species", ""),
                    "threshold_eV": float(entry.get("threshold_eV", 0.0)),
                    "energy_loss_eV": float(
                        entry.get("energy_loss_eV", entry.get("threshold_eV", 0.0))
                    ),
                    "file": f"cross_sections/{original.name}",
                    "metadata": entry.get("metadata", {}),
                }
            )
        cross_section_output = "cross_sections.yaml"
        (target / cross_section_output).write_text(
            yaml.safe_dump(
                {"cross_sections": converted}, sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )

    output_manifest = {
        "schema_version": 3,
        "species": "species.csv",
        "gas_reactions": "gas_reactions.csv",
        "boundary_reactions": "boundary_reactions.csv",
        **({"surface_reactions": "surface_reactions.csv"} if surface_file else {}),
        "rate_models": "rate_models.yaml",
        **({"cross_sections": cross_section_output} if cross_section_output else {}),
        "provenance": {"migrated_from": str(source)},
    }
    output = target / "chemistry.yaml"
    output.write_text(
        yaml.safe_dump(output_manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--boundary-products",
        type=Path,
        help="YAML mapping from incident ion ID to an explicit neutral product side",
    )
    parser.add_argument(
        "--cross-section-segments",
        type=Path,
        help="YAML mapping from cross-section ID to a zero-based concatenated segment",
    )
    arguments = parser.parse_args(argv)
    products: dict[str, str] | None = None
    if arguments.boundary_products is not None:
        raw_products = _yaml(arguments.boundary_products)
        products = {
            str(ion): str(product_side) for ion, product_side in raw_products.items()
        }
    segments: dict[str, int] | None = None
    if arguments.cross_section_segments is not None:
        raw_segments = _yaml(arguments.cross_section_segments)
        segments = {
            str(cross_section_id): int(segment)
            for cross_section_id, segment in raw_segments.items()
        }
    print(
        convert_v2_chemistry(
            arguments.manifest,
            arguments.output,
            boundary_products=products,
            cross_section_segments=segments,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["convert_v2_chemistry", "main"]
