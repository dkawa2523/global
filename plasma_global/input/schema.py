"""Public schema-v3 facade and cross-section case validation.

The private schema modules group input data by domain. This facade preserves the
historical import and pickle identity of every public model while keeping
whole-case relationships in :class:`CaseSpec`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, assert_never

from pydantic import PrivateAttr, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    InputPath,
    StrictModel,
)
from plasma_global.input._schema_models import (
    ApproximateTwoTermElectronModel,
    ElectronClosure,
    ElectronDensityModel,
    ElectronEnergyClosure,
    ElectronModel,
    EvolvedGasEnergy,
    FixedGasEnergy,
    GasEnergyModel,
    LocalFieldClosure,
    MaxwellianElectronModel,
    ModelsConfig,
    PrescribedElectronDensity,
    QuasiNeutralElectronDensity,
    SurfaceKineticsConfig,
    TableElectronModel,
)
from plasma_global.input._schema_models import CacheConfig as CacheConfig
from plasma_global.input._schema_models import EnergyGridConfig as EnergyGridConfig
from plasma_global.input._schema_models import (
    ReducedFieldGridConfig as ReducedFieldGridConfig,
)
from plasma_global.input._schema_power import (
    ContinuousWaveform,
    DCSeriesCommand,
    DCSeriesModel,
    ExperimentalCCPCommand,
    ExperimentalCCPModel,
    ExperimentalICPCommand,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    PowerModel,
    PowerPortCommand,
    PowerPortConfig,
    PrescribedPowerCommand,
    PrescribedPowerModel,
    SquarePulseWaveform,
    Waveform,
    _power_command_matches_model,
)
from plasma_global.input._schema_reactor import (
    AmbipolarWallTransport,
    BohmWallTransport,
    EdgeConfig,
    GasInletConfig,
    OffWallTransport,
    PrescribedFrequencyWallTransport,
    PumpConfig,
    ReactorConfig,
    SurfaceConfig,
    WallTransport,
    ZoneConfig,
)
from plasma_global.input._schema_recipe import (
    GasInletCommand,
    RecipeCommands,
    RecipeConfig,
    RecipeStepConfig,
    SurfaceCommand,
)
from plasma_global.input._schema_run import (
    ExperimentalConfig,
    ExperimentalExtensionsConfig,
    ExperimentalFilmConfig,
    ExperimentalInventoryEventConfig,
    ExperimentalWallInventoryConfig,
    OutputConfig,
    SolverConfig,
)


class CaseMetadata(StrictModel):
    name: Identifier
    description: str = ""


class ChemistryConfig(StrictModel):
    manifest: InputPath


class CaseSpec(StrictModel):
    schema_version: Literal[3]
    case: CaseMetadata
    chemistry: ChemistryConfig
    reactor: ReactorConfig
    recipe: RecipeConfig
    models: ModelsConfig
    solver: SolverConfig
    output: OutputConfig
    experimental: ExperimentalConfig | None = None

    _source_path: Path | None = PrivateAttr(default=None)
    _included_files: tuple[Path, ...] = PrivateAttr(default=())

    def _validate_recipe_commands(self) -> None:
        inlet_ids = {item.inlet_id for item in self.reactor.gas_inlets}
        ports = {item.port_id: item for item in self.reactor.power_ports}
        surface_ids = {item.surface_id for item in self.reactor.surfaces}
        for step in self.recipe.steps:
            commands = step.commands
            for label, references, available in (
                ("gas inlet", set(commands.gas_inlets), inlet_ids),
                ("power port", set(commands.power_ports), set(ports)),
                ("surface", set(commands.surfaces), surface_ids),
            ):
                unknown = references - available
                if unknown:
                    values = ", ".join(sorted(unknown))
                    raise ValueError(
                        f"recipe step {step.step_id!r} references unknown {label}(s): "
                        f"{values}"
                    )
            for port_id, command in commands.power_ports.items():
                port = ports[port_id]
                if not _power_command_matches_model(port.model, command):
                    raise ValueError(
                        f"recipe step {step.step_id!r} command kind "
                        f"{command.kind!r} does not match power port {port_id!r} "
                        f"model kind {port.model.kind!r}"
                    )

    def _validate_initial_electron_state(
        self, closure: Literal["electron_energy", "local_field"]
    ) -> None:
        for zone in self.reactor.zones:
            if closure == "electron_energy" and zone.initial_mean_energy_eV is None:
                raise ValueError(
                    f"zone {zone.zone_id!r} requires initial_mean_energy_eV for "
                    "the electron_energy closure"
                )
            if closure == "local_field" and zone.initial_mean_energy_eV is not None:
                raise ValueError(
                    f"zone {zone.zone_id!r} initial_mean_energy_eV is incompatible "
                    "with the local_field closure"
                )

    def _validate_electron_model(
        self, closure: Literal["electron_energy", "local_field"]
    ) -> None:
        electron_model = self.models.electrons
        if isinstance(electron_model, MaxwellianElectronModel):
            if closure != "electron_energy":
                raise ValueError(
                    "maxwellian electrons require the electron_energy closure"
                )
        elif isinstance(electron_model, ApproximateTwoTermElectronModel):
            if closure != "local_field":
                raise ValueError(
                    "experimental.approximate_two_term electrons require the "
                    "local_field closure"
                )
        elif isinstance(electron_model, TableElectronModel):
            expected_lookup = (
                "mean_energy" if closure == "electron_energy" else "local_field"
            )
            if electron_model.lookup != expected_lookup:
                raise ValueError(
                    f"table lookup {electron_model.lookup!r} must match electron "
                    f"closure {closure!r}"
                )
        else:
            assert_never(electron_model)

    def _validate_gas_energy_references(self, zone_ids: set[str]) -> None:
        gas_energy = self.models.gas_energy
        if isinstance(gas_energy, EvolvedGasEnergy):
            unknown = set(gas_energy.wall_energy_relaxation_s_inv_by_zone) - zone_ids
            if unknown:
                values = ", ".join(sorted(unknown))
                raise ValueError(
                    "models.gas_energy.wall_energy_relaxation_s_inv_by_zone "
                    f"references unknown zones: {values}"
                )
        elif not isinstance(gas_energy, FixedGasEnergy):
            assert_never(gas_energy)

    def _validate_wall_inventory_references(self, surface_ids: set[str]) -> None:
        wall_inventory = (
            None if self.experimental is None else self.experimental.wall_inventory
        )
        if wall_inventory is None:
            return
        unknown = set(wall_inventory.initial_by_surface) - surface_ids
        if unknown:
            values = ", ".join(sorted(unknown))
            raise ValueError(
                "experimental.wall_inventory.initial_by_surface references "
                f"unknown surfaces: {values}"
            )
        unknown_event_surfaces = {
            event.surface_id for event in wall_inventory.events
        } - surface_ids
        if unknown_event_surfaces:
            raise ValueError(
                "experimental.wall_inventory.events references unknown "
                f"surfaces: {', '.join(sorted(unknown_event_surfaces))}"
            )
        undeclared_inventory = {
            (event.surface_id, inventory_id)
            for event in wall_inventory.events
            for inventory_id in event.inventory_particles_per_event
            if inventory_id
            not in wall_inventory.initial_by_surface.get(event.surface_id, {})
        }
        if undeclared_inventory:
            raise ValueError(
                "experimental.wall_inventory.events writes undeclared "
                f"inventory states: {sorted(undeclared_inventory)}"
            )

    def _validate_save_times(self) -> None:
        save_at = self.solver.save_at_s
        if save_at is not None and (
            save_at[0] < self.recipe.start_time_s
            or save_at[-1] > self.recipe.end_time_s
        ):
            raise ValueError(
                "solver.save_at_s must lie within the recipe time interval"
            )

    @model_validator(mode="after")
    def valid_commands(self) -> Self:
        zone_ids = {item.zone_id for item in self.reactor.zones}
        surface_ids = {item.surface_id for item in self.reactor.surfaces}
        closure: Literal["electron_energy", "local_field"] = (
            self.models.electron_closure.kind
        )
        self._validate_recipe_commands()
        self._validate_initial_electron_state(closure)
        self._validate_electron_model(closure)
        self._validate_gas_energy_references(zone_ids)
        self._validate_wall_inventory_references(surface_ids)
        self._validate_save_times()
        return self

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def included_files(self) -> tuple[Path, ...]:
        return self._included_files

    def _with_source(self, source_path: Path, included_files: tuple[Path, ...]) -> Self:
        object.__setattr__(self, "_source_path", source_path)
        object.__setattr__(self, "_included_files", included_files)
        return self


__all__ = [
    "AmbipolarWallTransport",
    "ApproximateTwoTermElectronModel",
    "BohmWallTransport",
    "CaseMetadata",
    "CaseSpec",
    "ChemistryConfig",
    "ContinuousWaveform",
    "DCSeriesCommand",
    "DCSeriesModel",
    "EdgeConfig",
    "ElectronClosure",
    "ElectronDensityModel",
    "ElectronEnergyClosure",
    "ElectronModel",
    "EvolvedGasEnergy",
    "ExperimentalCCPCommand",
    "ExperimentalCCPModel",
    "ExperimentalConfig",
    "ExperimentalExtensionsConfig",
    "ExperimentalFilmConfig",
    "ExperimentalICPCommand",
    "ExperimentalICPModel",
    "ExperimentalInventoryEventConfig",
    "ExperimentalRFEnvelopeCommand",
    "ExperimentalRFEnvelopeModel",
    "ExperimentalWallInventoryConfig",
    "ExternalTableCommand",
    "ExternalTableModel",
    "FixedGasEnergy",
    "GasEnergyModel",
    "GasInletCommand",
    "GasInletConfig",
    "InputPath",
    "LocalFieldClosure",
    "MaxwellianElectronModel",
    "ModelsConfig",
    "OffWallTransport",
    "OutputConfig",
    "PowerModel",
    "PowerPortCommand",
    "PowerPortConfig",
    "PrescribedElectronDensity",
    "PrescribedFrequencyWallTransport",
    "PrescribedPowerCommand",
    "PrescribedPowerModel",
    "PumpConfig",
    "QuasiNeutralElectronDensity",
    "ReactorConfig",
    "RecipeCommands",
    "RecipeConfig",
    "RecipeStepConfig",
    "SolverConfig",
    "SquarePulseWaveform",
    "SurfaceCommand",
    "SurfaceConfig",
    "SurfaceKineticsConfig",
    "TableElectronModel",
    "WallTransport",
    "Waveform",
    "ZoneConfig",
]
