from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plasma_global import compile_case, simulate, write_result
from plasma_global.chemistry.compile import CompiledChemistry, compile_chemistry
from plasma_global.chemistry.data import SpeciesData, load_chemistry
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.power import CCPPowerPort
from plasma_global.input._compile_power import compile_power_coordinator
from plasma_global.input.load import case_from_mapping, load_case
from plasma_global.input.schema import CaseSpec
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.output import read_result_h5

SMOKE = (
    Path(__file__).parents[1] / "plasma_global" / "examples" / "cases" / "smoke.yaml"
)


@pytest.fixture(scope="module")
def smoke_inputs() -> tuple[CaseSpec, CompiledChemistry]:
    case = load_case(SMOKE)
    chemistry = compile_chemistry(
        load_chemistry(case.base_directory / case.chemistry["file"])
    )
    return case, chemistry


def _compiled_ccp(case: CaseSpec, chemistry: CompiledChemistry) -> CCPPowerPort:
    coordinator = compile_power_coordinator(case, chemistry, ExternalTableStore())
    assert coordinator is not None
    port = next(port for port in coordinator.ports if port.port_id == "wafer_bias")
    assert isinstance(port, CCPPowerPort)
    return port


def test_ccp_explicit_geometry_is_independent_of_reactor_surface_inference(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
) -> None:
    case, chemistry = smoke_inputs
    data = case.to_dict()
    geometry = {
        "electrode_gap_m": 0.043,
        "powered_area_m2": 0.017,
        "grounded_area_m2": 0.091,
    }
    data["reactor"]["power_ports"]["wafer_bias"]["model"].update(geometry)
    data["reactor"]["surfaces"]["wafer"]["area_m2"] *= 9.0
    public = case_from_mapping(data, base_directory=case.base_directory)
    port = _compiled_ccp(public, chemistry)
    assert port.electrode_gap_m == geometry["electrode_gap_m"]
    assert port.powered_area_m2 == geometry["powered_area_m2"]
    assert port.grounded_area_m2 == geometry["grounded_area_m2"]
    assert port.zone_volume_m3 == case.reactor.zones["process"].volume_m3


@pytest.mark.parametrize(
    "field", ["electrode_gap_m", "powered_area_m2", "grounded_area_m2", "dominant_ion"]
)
def test_ccp_effective_inputs_are_required_in_public_configuration(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
    field: str,
) -> None:
    case, _ = smoke_inputs
    data = case.to_dict()
    del data["reactor"]["power_ports"]["wafer_bias"]["model"][field]
    with pytest.raises(CaseValidationError, match=field):
        case_from_mapping(data, base_directory=case.base_directory)


@pytest.mark.parametrize("ion", ["Ar", "missing"])
def test_ccp_rejects_dominant_species_without_positive_gas_charge(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
    ion: str,
) -> None:
    case, chemistry = smoke_inputs
    data = case.to_dict()
    data["reactor"]["power_ports"]["wafer_bias"]["model"]["dominant_ion"] = ion
    public = case_from_mapping(data, base_directory=case.base_directory)
    with pytest.raises(CaseValidationError, match="positive gas ion"):
        _compiled_ccp(public, chemistry)


def test_ccp_uses_the_selected_positive_ion_mass(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
) -> None:
    case, chemistry = smoke_inputs
    data = case.to_dict()
    model = data["reactor"]["power_ports"]["wafer_bias"]["model"]
    source = load_chemistry(case.base_directory / case.chemistry["file"])
    ion = SpeciesData("Ar2_plus", "gas", 1, 79.896, {"Ar": 2.0})
    chemistry = compile_chemistry(replace(source, species=(*source.species, ion)))
    model["dominant_ion"] = ion.id
    port = _compiled_ccp(
        case_from_mapping(data, base_directory=case.base_directory), chemistry
    )
    assert port.dominant_ion_mass_kg == ion.mass_kg


def test_removed_surface_target_is_rejected_as_an_unknown_public_field(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
) -> None:
    case, _ = smoke_inputs
    data = deepcopy(case.to_dict())
    data["reactor"]["power_ports"]["wafer_bias"]["coupling_target"] = "wafer"
    with pytest.raises(CaseValidationError, match="coupling_target"):
        case_from_mapping(data, base_directory=case.base_directory)


_RF_PLASMA_DIAGNOSTICS = {
    "port_powered_sheath_voltage_V[wafer_bias]": "V",
    "port_grounded_sheath_voltage_V[wafer_bias]": "V",
    "port_ion_flux_m2_s[wafer_bias]": "m^-2 s^-1",
    "port_estimated_mean_ion_energy_eV[wafer_bias]": "eV",
    "port_pressure_Pa[wafer_bias]": "Pa",
    "port_pressure_Pa[source_rf]": "Pa",
    "port_mode_index[source_rf]": "1",
    "port_mode_H[source_rf]": "1",
}


def test_all_on_public_rf_diagnostics_survive_hdf5_with_units(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
    tmp_path: Path,
) -> None:
    case, _ = smoke_inputs
    data = case.to_dict()
    data["output"] = {"profile": "minimal", "observables": list(_RF_PLASMA_DIAGNOSTICS)}
    prepared = compile_case(case_from_mapping(data, base_directory=case.base_directory))
    for name, unit in _RF_PLASMA_DIAGNOSTICS.items():
        assert prepared.output_plan.catalog[name]["available"]
        assert prepared.output_plan.catalog[name]["unit"] == unit
    result = simulate(prepared)
    saved = read_result_h5(write_result(result, tmp_path / "rf").result_h5)
    for name, unit in _RF_PLASMA_DIAGNOSTICS.items():
        assert saved.metadata["series_units"][name] == unit
        assert np.isfinite(saved.series(name)).all()
        if name == "port_mode_H[source_rf]":
            assert ((saved.series(name) >= 0.0) & (saved.series(name) <= 1.0)).all()
        else:
            assert (saved.series(name) > 0.0).all()
        assert (saved.series(name) == result.series(name)).all()


@pytest.mark.parametrize("series", _RF_PLASMA_DIAGNOSTICS)
def test_plasma_only_rf_diagnostics_report_the_off_interval_before_solving(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
    series: str,
) -> None:
    case, _ = smoke_inputs
    data = case.to_dict()
    step = deepcopy(data["run"]["steps"][0])
    port = "source_rf" if "source_rf" in series else "wafer_bias"
    setpoint = "delivered_power_W" if port == "source_rf" else "absorbed_power_W"
    step["commands"]["power_ports"][port] = {setpoint: 0.0}
    data["run"]["steps"].append(step)
    data["output"] = {"profile": "minimal"}
    prepared = compile_case(case_from_mapping(data, base_directory=case.base_directory))
    catalog = prepared.output_plan.catalog[series]
    assert not catalog["available"]
    assert catalog["unavailable_intervals"][0]["segment_id"] == "step_2"
    assert catalog["unavailable_intervals"][0]["reason"]
    data["output"]["observables"] = [series]
    public = case_from_mapping(data, base_directory=case.base_directory)
    with pytest.raises(CaseValidationError, match="undefined in segment 'step_2'"):
        compile_case(public)


def test_rf_off_defined_voltage_remains_a_saved_zero(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
) -> None:
    case, _ = smoke_inputs
    data = case.to_dict()
    step = deepcopy(data["run"]["steps"][0])
    step["commands"]["power_ports"]["wafer_bias"] = {"absorbed_power_W": 0.0}
    data["run"]["steps"].append(step)
    name = "port_rf_voltage_rms_V[wafer_bias]"
    data["output"] = {"profile": "minimal", "observables": [name]}
    result = simulate(case_from_mapping(data, base_directory=case.base_directory))
    off = result.time_s > data["run"]["steps"][0]["duration_s"]
    assert off.any()
    assert (result.series(name)[off] == 0.0).all()


@pytest.mark.parametrize("input_kind", ["yaml", "python"])
def test_local_field_rejects_downstream_heating_without_electron_energy_balance(
    tmp_path: Path,
    input_kind: str,
) -> None:
    import yaml

    case = load_case(SMOKE.parent / "argon_lxcat.yaml")
    document = case.to_dict(relative_to=tmp_path)
    model = document["reactor"]["power_ports"]["source_rf"]["model"]
    model.update(downstream_zone_id="process", downstream_fraction=0.25)
    if input_kind == "yaml":
        path = tmp_path / "invalid.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(CaseValidationError, match=r"requires.*energy_balance"):
            load_case(path)
    else:
        with pytest.raises(CaseValidationError, match=r"requires.*energy_balance"):
            case_from_mapping(document, base_directory=tmp_path)
