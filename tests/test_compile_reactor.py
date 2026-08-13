from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from plasma_global.build import compile_case
from plasma_global.errors import CaseValidationError
from plasma_global.input.compile_reactor import (
    compile_external_binding,
    compile_initial_densities,
)
from plasma_global.input.load import load_case
from plasma_global.input.schema import (
    CaseSpec,
    ExternalTableCommand,
    ExternalTableModel,
)
from plasma_global.models.external_table import ExternalTableStore

FIXTURE = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"


def _updated_case(
    case: CaseSpec,
    update: Callable[[dict[str, Any]], None],
) -> CaseSpec:
    data = case.model_dump(mode="python")
    update(data)
    return CaseSpec.model_validate(data)


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


def test_explicit_density_validation_reports_unknown_species_before_pressure() -> None:
    def update(data: dict[str, Any]) -> None:
        zone = data["reactor"]["zones"][0]
        zone["initial_densities_m3"] = {"unknown": 1.0}

    case = _updated_case(load_case(FIXTURE), update)
    compiled = compile_case(load_case(FIXTURE))

    with pytest.raises(
        CaseValidationError,
        match=r"initializes unknown species \['unknown'\]",
    ):
        compile_initial_densities(case, compiled.chemistry)


def test_mole_fraction_validation_reports_unknown_species_before_charge_roles() -> None:
    def update(data: dict[str, Any]) -> None:
        zone = data["reactor"]["zones"][0]
        zone.pop("initial_densities_m3")
        zone["initial_mole_fractions"] = {"unknown": 1.0}
        zone["initial_seed_densities_m3"] = {"Ar": 1.0}

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
            f"time_s,electron_power_W\n0.0,{power_W}\n1.0,{power_W}\n",
            encoding="utf-8",
        )
    model = ExternalTableModel(
        kind="external_table",
        file=default_path,
        interpolation="linear",
        bounds_policy="error",
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
        bounds_policy="hold",
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
