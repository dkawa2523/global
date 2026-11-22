from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from plasma_global import (
    case_from_mapping,
    compile_case,
    load_case,
    simulate,
    write_result,
)
from plasma_global._result_network import network_data
from plasma_global.examples import list_examples
from plasma_global.input._schema_output import NetworkConfig
from plasma_global.output import inspect_result_h5, read_result_h5
from tools.generate_schema_docs import minimal_case_document

HEAVY_ENERGY_TERMS = (
    "gas_power_W_m3",
    "gas_reaction_heating_W_m3",
    "surface_reaction_heating_W_m3",
    "wall_heavy_power_W_m3",
    "elastic_heating_W_m3",
    "wall_species_power_W_m3",
    "surface_species_power_W_m3",
    "transport_heavy_power_W_m3",
)


def _particle_exchange_case(exchange: str, directory: Path):
    document: dict[str, Any] = minimal_case_document()
    species = document["chemistry"]["species"]
    species[1]["cv_over_kb"] = 1.5
    species[2].update(
        id="Ar2_plus", mass_amu=79.896, elements={"Ar": 2.0}, cv_over_kb=2.5
    )
    document["reactor"]["zones"]["plasma"]["initial"] = {
        "densities_m3": {"Ar": 1e15, "Ar2_plus": 1e15},
        "gas_temperature_K": 300.0,
        "mean_energy_eV": 3.0,
        "pressure_Pa": 2e15 * 1.380649e-23 * 300.0,
    }
    wall = {
        "zone_id": "plasma",
        "area_m2": 0.1,
        "temperature_K": 300.0,
        "wall_transport": {"kind": "prescribed_frequency", "frequency_s_inv": 10.0},
    }
    document["reactor"]["surfaces"] = {"wall": wall}
    document["models"]["gas_energy"] = {"kind": "evolved"}
    if exchange == "wall":
        document["chemistry"]["boundary_reactions"] = [
            {"id": "dissociate", "equation": "Ar2_plus -> 2 Ar"}
        ]
    else:
        wall.update(site_density_m2=1e19, wall_transport={"kind": "off"})
        species.extend(
            [
                {
                    "id": "site",
                    "phase": "surface",
                    "mass_amu": 0.0,
                    "elements": {"site": 1.0},
                    "state_tags": ["site"],
                },
                {
                    "id": "Ar_ads",
                    "phase": "surface",
                    "mass_amu": 39.948,
                    "elements": {"Ar": 1.0, "site": 1.0},
                },
            ]
        )
        document["models"]["surface_kinetics"] = {"kind": "mass_action"}
        document["chemistry"]["surface_reactions"] = [
            {
                "id": "stick",
                "equation": "Ar + site -> Ar_ads",
                "rate": {"kind": "sticking", "value": 0.2},
            }
        ]
    document["run"] = {"duration_s": 1e-3}
    document["solver"] = {
        "rtol": 1e-10,
        "atol": 1e-18,
        "save_at_s": np.linspace(0.0, 1e-3, 101).tolist(),
    }
    document["output"] = {
        "profile": "minimal",
        "observables": [f"{quantity}[plasma]" for quantity in HEAVY_ENERGY_TERMS],
    }
    return case_from_mapping(document, base_directory=directory)


@pytest.mark.parametrize(("exchange", "sign"), [("wall", 1), ("surface", -1)])
def test_saved_heavy_energy_terms_reconstruct_rhs_and_internal_energy(
    tmp_path: Path, exchange: str, sign: int
) -> None:
    prepared = compile_case(_particle_exchange_case(exchange, tmp_path))
    paths = write_result(simulate(prepared), tmp_path / "run")
    saved = read_result_h5(paths.result_h5)
    catalog = inspect_result_h5(paths.result_h5)["series_catalog"]
    summed = np.sum(
        [saved.series(f"{quantity}[plasma]") for quantity in HEAVY_ENERGY_TERMS],
        axis=0,
    )
    energy_index = prepared.model.layout.heavy_energy_indices["plasma"]
    rhs = [
        prepared.model.evaluate(float(time), state, prepared.segments[0]).derivative[
            energy_index
        ]
        for time, state in zip(saved.time_s, saved.state, strict=True)
    ]
    np.testing.assert_allclose(summed, rhs, rtol=1e-13, atol=1e-18)
    assert np.all(sign * saved.series(f"{exchange}_species_power_W_m3[plasma]") > 0)
    energy = saved.series("gas_internal_energy[plasma]")
    np.testing.assert_allclose(
        cumulative_trapezoid(summed, saved.time_s, initial=0),
        energy - energy[0],
        rtol=2e-6,
        atol=1e-16,
    )
    heat_capacity = 1.380649e-23 * (
        1.5 * saved.series("n[plasma,Ar]") + 2.5 * saved.series("n[plasma,Ar2_plus]")
    )
    np.testing.assert_allclose(energy / heat_capacity, 300.0, rtol=2e-9)
    for quantity in HEAVY_ENERGY_TERMS:
        row = catalog[f"{quantity}[plasma]"]
        assert (row["region_kind"], row["region"], row["unit"]) == (
            "zone",
            "plasma",
            "W m^-3",
        )
        assert row["stored"] is True


@pytest.mark.parametrize("example", list_examples())
def test_bundled_catalog_regions_match_declared_geometry(example: str) -> None:
    root = Path(str(files("plasma_global.examples")))
    prepared = compile_case(load_case(root / "cases" / f"{example}.yaml"))
    geometry = prepared.output_plan.metadata["model_geometry"]
    regions = {
        "zone": geometry["zones"],
        "surface": geometry["surfaces"],
        "inventory": geometry["surfaces"],
        "power_port": prepared.case.reactor.power_ports,
    }
    catalog = prepared.output_plan.catalog
    for name, row in catalog.items():
        assert row["region"] in regions[row["region_kind"]], (name, row)
        if row["quantity"] == "inventory":
            assert row["region_kind"] == "inventory"
    for event in prepared.output_plan.metadata["reaction_network"]:
        expected = "zone" if event["domain"] == "gas" else "surface"
        for field in ("rate_series", "frequency_series", "count_series"):
            assert catalog[event[field]]["region_kind"] == expected


def test_saved_network_event_ids_distinguish_colons_in_regions_and_reactions(
    tmp_path: Path,
) -> None:
    document: dict[str, Any] = minimal_case_document()
    document["chemistry"]["species"].append(
        {"id": "Ar_star", "mass_amu": 39.948, "elements": {"Ar": 1.0}}
    )
    zone = document["reactor"]["zones"]["plasma"]
    zone["initial"] = {
        "densities_m3": {"Ar": 1e20, "Ar_star": 1e18, "Ar_plus": 1e12},
        "gas_temperature_K": 300.0,
        "mean_energy_eV": 3.0,
        "pressure_Pa": (1.01e20 + 1e12) * 1.380649e-23 * 300.0,
    }
    document["reactor"]["zones"] = {"a:b": zone, "a": deepcopy(zone)}
    document["chemistry"]["gas_reactions"] = [
        {
            "id": reaction,
            "equation": "Ar_star -> Ar",
            "zones": [region],
            "rate": {"kind": "first_order", "rate_s_inv": rate},
        }
        for reaction, region, rate in (("c", "a:b", 1.0), ("b:c", "a", 10.0))
    ]
    document["run"] = {"duration_s": 1e-6}
    document["output"] = {"profile": "minimal", "reaction_rates": True}
    case = case_from_mapping(document, base_directory=tmp_path)
    paths = write_result(simulate(case), tmp_path / "run")
    saved = read_result_h5(paths.result_h5)
    events = saved.metadata["reaction_network"]
    assert len({event["id"] for event in events}) == len(events) == 2
    data = network_data(
        saved,
        NetworkConfig(mode="events", measure="rate", time_s=0.0, min_fraction=0.5),
    )
    assert [event["rate_series"] for event in data["events"]] == [
        "gas_reaction_rate_m3_s[a,b:c]"
    ]
    assert data["omitted_event_ids"] == ["gas_reaction_rate_m3_s[a:b,c]"]
