"""One-way v3 to v4 migration; legacy parsing never runs during simulation."""

from __future__ import annotations

import csv
import shutil
from collections.abc import Mapping
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from plasma_global._result_common import new_output_directory
from plasma_global._yaml import load_unique_yaml
from plasma_global.chemistry._data_yaml import load_definition
from plasma_global.chemistry.compile import compile_chemistry
from plasma_global.chemistry.data import ChemistryData, RateModelData
from plasma_global.input._serialize import case_document
from plasma_global.input._v3_document import read_v3_case
from plasma_global.input.compile_reactor import compile_initial_densities


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def record(value: Any) -> dict[str, Any]:
    return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}


def _reactions_document(values: Any) -> list[dict[str, Any]]:
    result = []
    for reaction in values:
        item = record(reaction)
        item.pop("reactants")
        item.pop("products")

        def side(terms: Mapping[str, float]) -> str:
            return " + ".join(
                f"{amount:g} {name}" if amount != 1.0 else name
                for name, amount in terms.items()
            )

        item["equation"] = f"{side(reaction.reactants)} -> {side(reaction.products)}"
        rate_id = item.pop("rate_model")
        if rate_id is not None:
            item["rate"] = rate_id
        result.append({key: value for key, value in item.items() if value is not None})
    return result


def _curves_document(data: ChemistryData, directory: Path) -> list[dict[str, Any]]:
    curves = []
    if data.cross_sections:
        with (directory / "cross_sections.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("channel_id", "energy_eV", "sigma_m2"))
            for curve in data.cross_sections.values():
                writer.writerows(
                    (curve.id, float(energy), float(sigma))
                    for energy, sigma in zip(
                        curve.energy_eV, curve.sigma_m2, strict=True
                    )
                )
                item = record(curve)
                item.pop("energy_eV")
                item.pop("sigma_m2")
                item["file"] = "cross_sections.csv"
                curves.append(
                    {key: value for key, value in item.items() if value is not None}
                )
    return curves


def chemistry_document(data: ChemistryData, directory: Path) -> dict[str, Any]:
    """Write external curves once and keep small definitions together in YAML."""
    species = [record(item) for item in data.species]
    for item in species:
        item["state_tags"] = sorted(item["state_tags"])
    models: dict[str, Any] = {}
    for key, model in data.rate_models.items():
        values = {"kind": model.kind, **_plain(model.parameters)}
        if "file" in values:
            target = f"rate-{len(models):04d}.csv"
            shutil.copyfile(values["file"], directory / target)
            values["file"] = target
        models[key] = values

    curves = _curves_document(data, directory)
    return {
        "schema_version": 4,
        "species": species,
        "gas_reactions": _reactions_document(data.gas_reactions),
        "boundary_reactions": _reactions_document(data.boundary_reactions),
        "surface_reactions": _reactions_document(data.surface_reactions),
        "rate_models": models,
        "cross_sections": curves,
        "experimental": _plain(data.experimental),
        "provenance": _plain(data.provenance),
    }


def _validate_legacy_arrays(arrays: dict[str, np.ndarray]) -> None:
    length = len(arrays["mean_energy_eV"])
    if any(
        values.shape != (length,)
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        for values in arrays.values()
    ):
        raise ValueError(
            "legacy electron arrays must be finite 1D numeric arrays sharing one grid"
        )


def convert_electron_hdf5(
    source: Path, target: Path, momentum: dict[str, str] | None = None
) -> dict[str, Any]:
    """Preserve stored float64 values; no sorting, interpolation or rescaling."""
    with h5py.File(source) as handle:
        attributes = {key: _plain(value) for key, value in handle.attrs.items()}
        if missing := set(momentum or {}) - set(handle["rate_coefficients"]):
            raise ValueError(
                f"legacy table is missing momentum rates {sorted(missing)}"
            )
        arrays = {
            name: np.asarray(handle[name][:])
            for name in ("mean_energy_eV", "effective_field_Td", "mobility_m2_V_s")
        }
        arrays.update(
            {
                (momentum or {}).get(name, f"k:{name}"): np.asarray(values[:])
                for name, values in handle["rate_coefficients"].items()
            }
        )
    _validate_legacy_arrays(arrays)
    arrays["reduced_field_Td"] = arrays.pop("effective_field_Td")
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(arrays)
        writer.writerows(zip(*arrays.values(), strict=True))
    return {
        "converted_from_hdf5": {
            "name": source.name,
            "sha256": sha256(source.read_bytes()).hexdigest(),
            "attributes": attributes,
        }
    }


def _legacy_reference(case: Any, chemistry: ChemistryData, notes: list[str]) -> float:
    electrons = case.models.electrons
    reference = electrons.mobility_reference_neutral_density_m3
    if reference is None:
        densities = compile_initial_densities(case, compile_chemistry(chemistry))
        neutrals = {
            item.id
            for item in chemistry.species
            if item.phase == "gas" and item.charge == 0
        }
        values = [
            sum(state.get(name, 0.0) for name in neutrals)
            for state in densities.values()
        ]
        if (
            not values
            or values[0] <= 0.0
            or not np.allclose(values, values[0], rtol=1e-12, atol=0.0)
        ):
            raise ValueError(
                "legacy mobility requires an explicit "
                "table-generation reference density"
            )
        reference = values[0]
        notes.append(
            "Mobility reference expanded from the old initial density; "
            "table generation conditions remain unverified."
        )
    return reference


def table_chemistry(data: ChemistryData) -> ChemistryData:
    """Separate supplied coefficients and explicit reaction energy from curves."""
    models = dict(data.rate_models)
    reactions = []
    energy_sources = {}
    for reaction in data.gas_reactions:
        model = data.rate_models[str(reaction.rate_model)]
        if model.kind == "electron_impact":
            curve = data.cross_sections[str(model.parameters["cross_section"])]
            transfer = reaction.resolved_electron_energy_transfer_eV
            if transfer is None:
                transfer = curve.resolved_electron_energy_transfer_eV
            reaction = replace(
                reaction, energy_loss_eV=None, electron_energy_transfer_eV=transfer
            )
            models[model.id] = RateModelData(
                model.id,
                "electron_table",
                {
                    "rate_id": curve.id,
                    "branching_yield": model.parameters["branching_yield"],
                },
            )
            energy_sources[reaction.id] = {
                "cross_section_id": curve.id,
                "source": dict(curve.metadata),
            }
        reactions.append(reaction)
    provenance = {**data.provenance, "electron_energy_sources": energy_sources}
    return replace(
        data,
        rate_models=models,
        gas_reactions=tuple(reactions),
        cross_sections={},
        provenance=provenance,
    )


def _migrate_electrons(
    case: Any,
    chemistry: ChemistryData,
    document: dict[str, Any],
    stage: Path,
    notes: list[str],
) -> ChemistryData:
    config = case.models.electrons
    if config.kind != "table":
        return chemistry
    momentum = {}
    for curve in chemistry.cross_sections.values():
        if curve.kind == "momentum_transfer":
            name = f"momentum:{curve.target}"
            if name in momentum.values():
                raise ValueError(
                    "multiple momentum curves per target require explicit review"
                )
            momentum[curve.id] = name
    provenance = convert_electron_hdf5(config.file, stage / "electrons.csv", momentum)
    reference = _legacy_reference(case, chemistry, notes)
    document["models"]["electrons"]["coefficients"].update(
        file="electrons.csv",
        mobility_reference_neutral_density_m3=reference,
        provenance=provenance,
    )
    for port_id, port in document["reactor"]["power_ports"].items():
        model = port["model"]
        if (
            model["kind"] == "dc_series"
            and model.pop("electron_mobility_m2_V_s", None) is not None
        ):
            notes.append(
                f"Removed unused configured mobility on {port_id}: "
                "the legacy table mobility took precedence."
            )
    return table_chemistry(chemistry)


def migrate_v3(source: str | Path, output: str | Path) -> Path:
    """Publish a self-contained new bundle, retaining explicit physical values."""
    case = read_v3_case(source)
    if case.chemistry.manifest is None:
        raise ValueError("v3 chemistry requires a manifest")
    with case.chemistry.manifest.open(encoding="utf-8") as stream:
        raw = load_unique_yaml(stream)
    raw["schema_version"] = 4
    chemistry = load_definition(
        raw, case.chemistry.manifest, case.chemistry.manifest.parent
    )
    notes = []
    directory = Path(output).resolve()
    with new_output_directory(directory) as stage:
        chemistry_dir = stage / "chemistry"
        chemistry_dir.mkdir()
        document = case_document(case)
        chemistry = _migrate_electrons(case, chemistry, document, stage, notes)
        chemistry_input = chemistry_document(chemistry, chemistry_dir)
        (chemistry_dir / "chemistry.yaml").write_text(
            yaml.safe_dump(chemistry_input, sort_keys=False), encoding="utf-8"
        )
        document["chemistry"] = {"file": "chemistry/chemistry.yaml"}
        for index, port in enumerate(document["reactor"]["power_ports"].values()):
            model = port["model"]
            if "file" in model:
                target = f"power-{index:04d}.csv"
                shutil.copyfile(model["file"], stage / target)
                model["file"] = target
        density = document["models"]["electron_density"]
        if "file" in density:
            shutil.copyfile(density["file"], stage / "electron-density.csv")
            density["file"] = "electron-density.csv"
        for index, step in enumerate(document["run"].get("steps", [])):
            for port_id, command in step["commands"].get("power_ports", {}).items():
                if "file" in command:
                    name = f"power-step-{index:04d}-{len(notes):04d}.csv"
                    shutil.copyfile(command["file"], stage / name)
                    command["file"] = name
                    notes.append(f"Copied step source for {port_id}")
        target = stage / "case.yaml"
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        (stage / "migration.yaml").write_text(
            yaml.safe_dump(
                {"source_schema": 3, "target_schema": 4, "notes": notes},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        from plasma_global.build import compile_case
        from plasma_global.input.load import load_case

        compile_case(load_case(target))
    return directory / "case.yaml"
