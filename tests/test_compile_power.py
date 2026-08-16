from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from plasma_global.chemistry.compile import CompiledChemistry, compile_chemistry
from plasma_global.chemistry.data import load_chemistry
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.power import CCPPowerPort
from plasma_global.input._compile_power import compile_power_coordinator
from plasma_global.input.load import load_case
from plasma_global.input.schema import CaseSpec
from plasma_global.models.external_table import ExternalTableStore

SMOKE = Path(__file__).parents[1] / "examples" / "v3" / "cases" / "smoke.yaml"
_INVALID_TARGETS = (
    ("source_rf", "source", "does not support coupling_target"),
    ("wafer_bias", "missing", "must reference a reactor surface"),
    ("wafer_bias", "source_wall", "belongs to zone 'source'"),
)


@pytest.fixture(scope="module")
def smoke_inputs() -> tuple[CaseSpec, CompiledChemistry]:
    case = load_case(SMOKE)
    chemistry = compile_chemistry(load_chemistry(case.chemistry.manifest))
    return case, chemistry


def _with_target(case: CaseSpec, port_id: str, target: str) -> CaseSpec:
    ports = [
        port.model_copy(update={"coupling_target": target})
        if port.port_id == port_id
        else port
        for port in case.reactor.power_ports
    ]
    reactor = case.reactor.model_copy(update={"power_ports": ports})
    return case.model_copy(update={"reactor": reactor})


def _compiled_ccp(case: CaseSpec, chemistry: CompiledChemistry) -> CCPPowerPort:
    coordinator = compile_power_coordinator(case, chemistry, ExternalTableStore())
    assert coordinator is not None
    port = next(port for port in coordinator.ports if port.port_id == "wafer_bias")
    assert isinstance(port, CCPPowerPort)
    return port


def test_ccp_target_selects_one_same_zone_surface(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
) -> None:
    case, chemistry = smoke_inputs

    port = _compiled_ccp(case, chemistry)

    wafer = next(
        surface for surface in case.reactor.surfaces if surface.surface_id == "wafer"
    )
    assert port.powered_area_m2 == wafer.area_m2


def test_ccp_without_target_keeps_the_half_area_default(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
) -> None:
    case, chemistry = smoke_inputs
    untargeted = _with_target(case, "wafer_bias", "")

    port = _compiled_ccp(untargeted, chemistry)

    process_area = sum(
        surface.area_m2
        for surface in case.reactor.surfaces
        if surface.zone_id == "process"
    )
    assert port.powered_area_m2 == process_area / 2.0


@pytest.mark.parametrize(
    ("port_id", "target", "message"),
    _INVALID_TARGETS,
)
def test_power_target_contract_is_part_of_the_input_schema(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
    port_id: str,
    target: str,
    message: str,
) -> None:
    case, _ = smoke_inputs
    invalid = _with_target(case, port_id, target)

    with pytest.raises(ValidationError, match=message):
        CaseSpec.model_validate(invalid.model_dump(mode="python"))


@pytest.mark.parametrize(
    ("port_id", "target", "message"),
    _INVALID_TARGETS,
)
def test_power_compiler_rejects_targets_that_would_be_ignored(
    smoke_inputs: tuple[CaseSpec, CompiledChemistry],
    port_id: str,
    target: str,
    message: str,
) -> None:
    case, chemistry = smoke_inputs

    with pytest.raises(CaseValidationError, match=message):
        compile_power_coordinator(
            _with_target(case, port_id, target),
            chemistry,
            ExternalTableStore(),
        )
