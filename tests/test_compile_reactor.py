from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from plasma_global import simulate
from plasma_global.build import compile_case
from plasma_global.errors import CaseValidationError
from plasma_global.input._schema_power import (
    ExternalTableCommand,
    ExternalTableModel,
)
from plasma_global.input.compile_reactor import (
    compile_external_binding,
    compile_initial_densities,
)
from plasma_global.input.load import case_from_mapping, load_case
from plasma_global.input.schema import CaseSpec
from plasma_global.models.external_table import ExternalTableStore
from tools.generate_schema_docs import minimal_case_document

FIXTURE = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"


def _updated_case(
    case: CaseSpec,
    update: Callable[[dict[str, Any]], None],
) -> CaseSpec:
    data = case.model_dump(mode="python")
    update(data)
    return case_from_mapping(data, base_directory=case.base_directory)


def test_initial_densities_keep_read_only_mapping_and_state_array_contract() -> None:
    compiled = compile_case(load_case(FIXTURE))
    densities = compile_initial_densities(compiled.case, compiled.chemistry)

    assert densities["plasma"] == {"Ar": 2.0e20, "Ar_plus": 1.0e12}
    with pytest.raises(TypeError):
        cast(dict[str, Any], densities)["other"] = {}
    with pytest.raises(TypeError):
        cast(dict[str, float], densities["plasma"])["Ar"] = 0.0

    state = compiled.model.initial_state(compiled.initial_state)
    assert state.shape == (compiled.model.layout.size,)
    assert state.dtype == np.float64


@pytest.mark.parametrize("use_mole_fractions", [True, False])
def test_an_added_species_starts_at_zero_when_omitted_from_initial_state(
    tmp_path: Path, use_mole_fractions: bool
) -> None:
    document = minimal_case_document()
    document["chemistry"]["species"].append({"id": "Ar_star", "mass_amu": 39.948})
    if not use_mole_fractions:
        initial = document["reactor"]["zones"]["plasma"]["initial"]
        initial["densities_m3"] = {"Ar": 2e20, "Ar_plus": 1e12}
        initial["pressure_Pa"] = (2e20 + 1e12) * 1.380649e-23 * 300.0
        del initial["mole_fractions"], initial["seed_densities_m3"]
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = simulate(load_case(path))
    np.testing.assert_array_equal(result.series("n[plasma,Ar_star]"), 0.0)


def test_explicit_density_validation_reports_unknown_species_before_pressure() -> None:
    def update(data: dict[str, Any]) -> None:
        zone = data["reactor"]["zones"]["plasma"]["initial"]
        zone["densities_m3"] = {"unknown": 1.0}

    case = _updated_case(load_case(FIXTURE), update)
    compiled = compile_case(load_case(FIXTURE))

    with pytest.raises(
        CaseValidationError,
        match=r"initializes unknown species \['unknown'\]",
    ):
        compile_initial_densities(case, compiled.chemistry)


def test_mole_fraction_validation_reports_unknown_species_before_charge_roles() -> None:
    def update(data: dict[str, Any]) -> None:
        zone = data["reactor"]["zones"]["plasma"]["initial"]
        zone.pop("densities_m3")
        zone["mole_fractions"] = {"unknown": 1.0}
        zone["seed_densities_m3"] = {"Ar": 1.0}

    case = _updated_case(load_case(FIXTURE), update)
    compiled = compile_case(load_case(FIXTURE))

    with pytest.raises(
        CaseValidationError,
        match=r"initializes unknown species \['unknown'\]",
    ):
        compile_initial_densities(case, compiled.chemistry)


def test_external_table_command_overrides_only_explicit_fields(tmp_path: Path) -> None:
    default_path = tmp_path / "default.csv"
    override_path = tmp_path / "override.csv"
    for path, power_W in ((default_path, 1.0), (override_path, 2.0)):
        path.write_text(
            f"time_s,electron_power_W,voltage_V,current_A\n"
            f"0.0,{power_W},2,1\n1.0,{power_W},2,1\n",
            encoding="utf-8",
        )
    model = ExternalTableModel(
        kind="external_table",
        file=default_path,
        interpolation="linear",
        bounds="error",
        power_scale=3.0,
        voltage_scale=4.0,
        current_scale=5.0,
        gap_m=0.01,
        total_density_m3=2.0e20,
        plasma_potential_V=6.0,
    )
    command = ExternalTableCommand(
        kind="external_table",
        file=override_path,
        interpolation="previous",
        bounds="hold",
        power_scale=7.0,
        time_offset_s=0.25,
    )

    binding = compile_external_binding(model, ExternalTableStore(), command)

    assert binding.data.electron_power_W[0] == 2.0
    assert binding.interpolation == "previous"
    assert binding.bounds == "hold"
    assert binding.power_scale == 7.0
    assert binding.voltage_scale == 4.0
    assert binding.current_scale == 5.0
    assert binding.gap_m == 0.01
    assert binding.total_density_m3 == 2.0e20
    assert binding.plasma_potential_V == 6.0
    assert binding.time_offset_s == 0.25
