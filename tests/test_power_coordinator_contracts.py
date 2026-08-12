from __future__ import annotations

import importlib
import inspect
import pickle
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.kinetics import TabulatedElectronKinetics
from plasma_global.models.power import (
    CompiledPowerCommand,
    DCSeriesPort,
    ExternalTablePowerPort,
    PowerCoordinator,
    PowerCouplingResult,
    PowerPort,
    PowerPortResult,
    PowerState,
    PrescribedPowerPort,
)


class _MismatchedPort:
    @property
    def port_id(self) -> str:
        return "declared"

    @property
    def zone_id(self) -> str:
        return "plasma"

    @property
    def time_dependent(self) -> bool:
        return False

    @property
    def produces_reduced_field(self) -> bool:
        return False

    def evaluate(
        self,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        del time_s, state, command
        return PowerPortResult("returned", self.zone_id, electron_power_W=1.0)


class _UndeclaredFieldPort:
    def __init__(self, port_id: str) -> None:
        self._port_id = port_id

    @property
    def port_id(self) -> str:
        return self._port_id

    @property
    def zone_id(self) -> str:
        return "plasma"

    @property
    def time_dependent(self) -> bool:
        return False

    @property
    def produces_reduced_field(self) -> bool:
        return False

    def evaluate(
        self,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        del time_s, state, command
        return PowerPortResult(
            self.port_id,
            self.zone_id,
            electron_power_W=0.0,
            reduced_field_Td=10.0,
        )


class _SettlingPowerPort:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def port_id(self) -> str:
        return "field"

    @property
    def zone_id(self) -> str:
        return "plasma"

    @property
    def time_dependent(self) -> bool:
        return False

    @property
    def produces_reduced_field(self) -> bool:
        return True

    def evaluate(
        self,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        del time_s, state, command
        self.calls += 1
        power_W = 1.0 if self.calls == 1 else 2.0
        return PowerPortResult(
            self.port_id,
            self.zone_id,
            electron_power_W=power_W,
            reduced_field_Td=10.0,
        )


def _evaluate(
    coordinator: PowerCoordinator,
    *,
    mean_energy_eV: float | None = 3.0,
    kinetics: TabulatedElectronKinetics | None = None,
) -> PowerCouplingResult:
    return coordinator.evaluate(
        time_s=0.0,
        commands={},
        electron_density_m3_by_zone={"plasma": 1.0e15},
        neutral_density_m3_by_zone={"plasma": 1.0e20},
        mean_energy_eV_by_zone={"plasma": mean_energy_eV},
        kinetics_by_zone={} if kinetics is None else {"plasma": kinetics},
    )


def test_coordinator_rejects_a_port_result_with_mismatched_identity() -> None:
    coordinator = PowerCoordinator((_MismatchedPort(),), ("plasma",))

    with pytest.raises(ModelDomainError, match="returned mismatched identity"):
        _evaluate(coordinator)


def test_coordinator_rejects_multiple_fields_returned_by_undeclared_ports() -> None:
    coordinator = PowerCoordinator(
        (_UndeclaredFieldPort("first"), _UndeclaredFieldPort("second")),
        ("plasma",),
    )

    with pytest.raises(CaseValidationError, match="multiple E/N-producing"):
        _evaluate(coordinator)


def test_coupling_waits_until_electron_power_also_converges() -> None:
    port = _SettlingPowerPort()
    table = TabulatedElectronKinetics(
        source=Path("settling-power.h5"),
        lookup="local_field",
        bounds="error",
        axis=np.array([10.0, 20.0]),
        mean_energy_eV=np.array([3.0, 3.0]),
        mobility_m2_V_s=np.array([0.5, 0.5]),
        effective_field_Td=np.array([10.0, 20.0]),
        rate_tables={},
    )
    coordinator = PowerCoordinator((port,), ("plasma",))

    result = _evaluate(coordinator, mean_energy_eV=None, kinetics=table)

    assert result.iterations_by_zone["plasma"] == 3
    assert result.electron_power_W_by_zone["plasma"] == 2.0
    assert port.calls == 3


@pytest.mark.parametrize(
    "public_type",
    [
        CompiledPowerCommand,
        DCSeriesPort,
        ExternalTablePowerPort,
        PowerCoordinator,
        PowerCouplingResult,
        PowerPort,
        PowerPortResult,
        PowerState,
        PrescribedPowerPort,
    ],
)
def test_public_power_types_keep_their_module_identity(
    public_type: type[object],
) -> None:
    assert public_type.__module__ == "plasma_global.models.power"
    assert get_type_hints(public_type) is not None
    assert f"class {public_type.__name__}" in inspect.getsource(public_type)
    pickle_payload = pickle.dumps(public_type)
    assert b"plasma_global.models.power" in pickle_payload
    assert (
        getattr(importlib.import_module(public_type.__module__), public_type.__name__)
        is public_type
    )
    assert all(
        not base.__module__.startswith("plasma_global.models._power")
        for base in public_type.__bases__
    )


def test_coordinator_returns_only_public_power_value_types() -> None:
    port = PrescribedPowerPort("source", "plasma", default_power_W=3.0)
    coordinator = PowerCoordinator((port,), ("plasma",))

    result = _evaluate(coordinator)

    assert type(result) is PowerCouplingResult
    assert type(result.power_state_by_zone["plasma"]) is PowerState
    assert type(result.port_results["source"]) is PowerPortResult
    assert isinstance(result.port_results["source"], PowerPortResult)
