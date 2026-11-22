from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from plasma_global import (
    case_from_mapping,
    compile_case,
    load_case,
    simulate,
    write_result,
)
from plasma_global.errors import ChemistryError
from plasma_global.output import read_result_h5

EXAMPLE = (
    Path(__file__).parents[1] / "plasma_global/examples/cases/surface_inventory.yaml"
)


def _ion_assisted_document() -> dict[str, Any]:
    data = load_case(EXAMPLE).to_dict()
    data.pop("experimental")
    chemistry = data["chemistry"]
    chemistry["species"] = [s for s in chemistry["species"] if s["id"] != "Ar_film"]
    for species in chemistry["species"]:
        species.pop("surfaces", None)
    chemistry["boundary_reactions"] = [
        {"id": "neutralize", "equation": "Ar_plus -> Ar"},
    ]
    chemistry["surface_reactions"] = [
        {
            "id": "release",
            "equation": "Ar_plus + Ar_ads -> Ar_plus + Ar + site",
            "rate": {
                "kind": "ion_assisted",
                "yield": 1.0,
                "threshold_eV": 0.0,
                "reference_energy_eV": 10.0,
                "exponent": 1.0,
            },
        }
    ]
    zone = data["reactor"]["zones"]["plasma"]
    zone["initial"] = {
        "densities_m3": {"Ar": 1e15, "Ar_plus": 1e15},
        "gas_temperature_K": 300.0,
        "mean_energy_eV": 3.0,
        "pressure_Pa": 2e15 * 1.380649e-23 * 300.0,
    }
    wall = data["reactor"]["surfaces"]["wall"]
    wall.update(
        initial_coverages={"Ar_ads": 0.4},
        ion_impact_energy_eV=10.0,
        wall_transport={"kind": "prescribed_frequency", "frequency_s_inv": 0.5},
    )
    data["run"] = {"duration_s": 1e-3}
    data["solver"] = {"rtol": 1e-9, "atol": 1e-14, "sample_interval_s": 5e-5}
    data["output"] = {"profile": "minimal", "csv": True}
    return data


def _csv_chemistry(directory: Path, chemistry: dict[str, Any]) -> dict[str, str]:
    directory.mkdir()
    species_path = directory / "species.csv"
    fields = ["id", "phase", "charge", "mass_amu", "elements", "state_tags"]
    with species_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for species in chemistry["species"]:
            row = {key: species[key] for key in fields if key in species}
            row["phase"] = species.get("phase", "gas")
            row["charge"] = species.get("charge", 0)
            row["elements"] = ";".join(
                f"{k}:{v}" for k, v in species.get("elements", {}).items()
            )
            row["state_tags"] = ";".join(species.get("state_tags", ()))
            writer.writerow(row)
    (directory / "gas.csv").write_text("id,equation,rate_model\n", encoding="utf-8")
    (directory / "boundary.csv").write_text(
        "id,equation\nneutralize,Ar_plus -> Ar\n", encoding="utf-8"
    )
    reaction = chemistry["surface_reactions"][0]
    with (directory / "surface.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "equation", "rate_model"])
        writer.writerow([reaction["id"], reaction["equation"], "assist"])
    (directory / "rates.yaml").write_text(
        yaml.safe_dump({"rate_models": {"assist": reaction["rate"]}}), encoding="utf-8"
    )
    manifest = directory / "chemistry.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 4,
                "species": "species.csv",
                "gas_reactions": "gas.csv",
                "boundary_reactions": "boundary.csv",
                "surface_reactions": "surface.csv",
                "rate_models": "rates.yaml",
            }
        ),
        encoding="utf-8",
    )
    return {"file": str(manifest)}


@pytest.mark.parametrize("chemistry_format", ["inline", "csv"])
def test_public_case_rejects_balanced_noncatalytic_ion_capture_before_solving(
    tmp_path: Path,
    chemistry_format: str,
) -> None:
    data = _ion_assisted_document()
    # This capture conserves both atoms and charge on paper; it is unsupported
    # because the transport wall already owns the incident ion and its return.
    for species in data["chemistry"]["species"]:
        if species["id"] == "Ar_ads":
            species["charge"] = 1
    data["chemistry"]["surface_reactions"][0]["equation"] = "Ar_plus + site -> Ar_ads"
    if chemistry_format == "csv":
        data["chemistry"] = _csv_chemistry(tmp_path / "chemistry", data["chemistry"])
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ChemistryError, match="must preserve the incident ion"):
        compile_case(load_case(path))


def test_public_ion_assisted_catalyst_conserves_atoms_with_wall_return_and_hdf(
    tmp_path: Path,
) -> None:
    data = _ion_assisted_document()
    result = simulate(case_from_mapping(data, base_directory=tmp_path))
    assert result.status.success
    paths = write_result(result, tmp_path / "catalyst")
    assert paths.artifacts.success
    saved = read_result_h5(paths.result_h5)
    np.testing.assert_array_equal(saved.state, result.state)
    gas_atoms = (
        saved.series("n[plasma,Ar]") + saved.series("n[plasma,Ar_plus]")
    ) * 0.01
    adsorbed_atoms = saved.series("coverage[wall,Ar_ads]") * 1e19 * 0.01
    np.testing.assert_allclose(
        gas_atoms + adsorbed_atoms, gas_atoms[0] + adsorbed_atoms[0], rtol=1e-12
    )
    np.testing.assert_allclose(
        saved.series("n[plasma,Ar_plus]"), 1e15 * np.exp(-0.5 * saved.time_s), rtol=1e-8
    )
    assert saved.series("coverage[wall,Ar_ads]")[-1] < 0.4


def test_public_surface_events_keep_at_ids_and_joining_collisions_distinct(
    tmp_path: Path,
) -> None:
    source = load_case(EXAMPLE)
    data = source.to_dict()
    pairs = (("deposit@a", "wall@a", 0.0001), ("deposit", "a@wall@a", 0.0003))
    # Both pairs used to become the same key: deposit@a@wall@a.
    surfaces = {}
    for _, surface, _ in pairs:
        surfaces[surface] = deepcopy(data["reactor"]["surfaces"]["wall"])
    data["reactor"]["surfaces"] = surfaces
    for species in data["chemistry"]["species"]:
        species.pop("surfaces", None)
    deposit = data["chemistry"]["surface_reactions"][-1]
    reactions = []
    for reaction, surface, sticking in pairs:
        record = deepcopy(deposit)
        record.update(id=reaction, surfaces=[surface])
        record["rate"]["value"] = sticking
        reactions.append(record)
    data["chemistry"]["surface_reactions"] = reactions
    inventory = data["experimental"]["wall_inventory"]
    inventory["initial_by_surface"] = {s: {"deposited_Ar": 0.0} for _, s, _ in pairs}
    inventory["events"] = [
        {
            "reaction_id": r,
            "surface_id": s,
            "inventory_particles_per_event": {"deposited_Ar": 1.0},
        }
        for r, s, _ in pairs
    ]
    data["output"] = {
        "profile": "minimal",
        "csv": True,
        "observables": [
            f"{quantity}[{surface},{reaction}]"
            for reaction, surface, _ in pairs
            for quantity in (
                "surface_reaction_rate_m2_s",
                "surface_event_rate_s_inv",
                "surface_event_count",
            )
        ],
    }
    path = tmp_path / "at-ids.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    prepared = compile_case(load_case(path))
    result = simulate(prepared)
    assert result.status.success
    paths = write_result(result, tmp_path / "colliding-pairs")
    assert paths.artifacts.success
    saved = read_result_h5(paths.result_h5)
    np.testing.assert_array_equal(saved.state, result.state)
    thermal_speed = np.sqrt(
        8.0 * 1.380649e-23 * 300.0 / (np.pi * 39.948 * 1.66053906660e-27)
    )
    deposited = []
    for reaction, surface, sticking in pairs:
        rate = f"surface_reaction_rate_m2_s[{surface},{reaction}]"
        frequency = f"surface_event_rate_s_inv[{surface},{reaction}]"
        count = f"surface_event_count[{surface},{reaction}]"
        np.testing.assert_allclose(
            saved.series(rate),
            0.25 * sticking * thermal_speed * saved.series("n[plasma,Ar]"),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            saved.series(frequency), saved.series(rate) * 0.01, rtol=1e-12
        )
        particles = saved.series(f"inventory[{surface},deposited_Ar]")
        np.testing.assert_allclose(saved.series(count), particles, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(
            saved.series(f"film[{surface}]") / 3e-10 * 1e19 * 0.01,
            particles,
            rtol=1e-12,
        )
        assert saved.metadata["series_units"][rate] == "m^-2 s^-1"
        assert saved.metadata["series_units"][frequency] == "s^-1"
        assert saved.metadata["series_units"][count] == "1"
        deposited.append(particles)
    np.testing.assert_allclose(deposited[1], 3.0 * deposited[0], rtol=1e-12)
    gas = (saved.series("n[plasma,Ar]") + saved.series("n[plasma,Ar_plus]")) * 0.01
    total = gas + deposited[0] + deposited[1]
    np.testing.assert_allclose(total, total[0], rtol=1e-12)
