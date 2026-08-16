"""Compiled, volume-averaged plasma chemistry and energy equations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

import numpy as np
from scipy.sparse import csr_matrix

from plasma_global.core._compiled_chemistry import (
    ChemistrySource,
    CompiledChemistryData,
    compile_chemistry_data,
)
from plasma_global.core._compiled_domain import DomainValidator
from plasma_global.core._compiled_jacobian import JacobianBuilder
from plasma_global.core._compiled_runtime import RuntimeEvaluator
from plasma_global.core._initial_state import build_initial_state
from plasma_global.core._runtime_assembly import RuntimeEvaluation, RuntimeZoneLedger
from plasma_global.core.domain import InitialState, RecipeSegment, Zone
from plasma_global.core.transport import CompiledTransport, SegmentTransport
from plasma_global.errors import ModelConfigurationError
from plasma_global.models.electrons import ElectronClosure, ElectronState
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)
from plasma_global.models.power import PowerCoordinator, PowerCouplingResult
from plasma_global.models.rates import RateEvaluator
from plasma_global.models.surface import CompiledSurfaceModel
from plasma_global.models.walls import (
    CompiledWallBoundary,
    WallBoundary,
    WallFluxRecord,
    compile_wall_boundary,
)


class CompiledChemistryLike(ChemistrySource, Protocol):
    """Narrow interface expected from an input or chemistry compiler.

    Species arrays contain only evolved heavy species.  Electron reactant
    orders are separate because electron density is imposed by quasineutrality.
    Stoichiometry is indexed ``[reaction, species]``.
    """


class ElectronDensityProvider(Protocol):
    """Narrow optional closure for prescribed electron-density profiles."""

    def __call__(self, time_s: float, zone_id: str) -> float: ...


class ExtensionStateAccumulator(Protocol):
    """One-way optional state block supplied by the composition root."""

    @property
    def labels(self) -> Sequence[str]: ...

    @property
    def lower_bounds(self) -> Sequence[float | None]: ...

    @property
    def upper_bounds(self) -> Sequence[float | None]: ...

    def initial_state(self) -> np.ndarray: ...

    def rhs(
        self,
        values: np.ndarray,
        *,
        drivers: Mapping[str, Mapping[str, float]],
        surface_rates_m2_s: Mapping[str, float],
    ) -> np.ndarray: ...


ElasticHeatingEvaluator = Callable[
    [str, ElectronState, np.ndarray, float, ElectronKineticsResult | None], float
]

_K = TypeVar("_K")
_V = TypeVar("_V")


def _frozen_mapping(values: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(values))


def _required_cached_derivative(derivative: np.ndarray | None) -> np.ndarray:
    if derivative is None:
        raise RuntimeError("RHS derivative cache is internally inconsistent")
    return derivative


def _readonly_csr(value: csr_matrix) -> csr_matrix:
    for array in (value.data, value.indices, value.indptr):
        array.setflags(write=False)
    return value


@dataclass(frozen=True)
class StateLayout:
    species_ids: tuple[str, ...]
    zone_ids: tuple[str, ...]
    evolves_electron_energy: bool
    evolves_heavy_energy: bool = False
    surface_coverage_keys: tuple[tuple[str, str], ...] = ()
    extension_labels: tuple[str, ...] = ()
    density_slices: Mapping[str, slice] = field(init=False)
    electron_energy_indices: Mapping[str, int] = field(init=False)
    heavy_energy_indices: Mapping[str, int] = field(init=False)
    surface_coverage_indices: Mapping[tuple[str, str], int] = field(init=False)
    surface_coverage_slice: slice = field(init=False)
    extension_slice: slice = field(init=False)
    labels: tuple[str, ...] = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        density_slices: dict[str, slice] = {}
        energy_indices: dict[str, int] = {}
        heavy_energy_indices: dict[str, int] = {}
        coverage_indices: dict[tuple[str, str], int] = {}
        labels: list[str] = []
        cursor = 0
        for zone_id in self.zone_ids:
            density_slices[zone_id] = slice(cursor, cursor + len(self.species_ids))
            labels.extend(
                f"n[{zone_id},{species_id}]" for species_id in self.species_ids
            )
            cursor += len(self.species_ids)
            if self.evolves_electron_energy:
                energy_indices[zone_id] = cursor
                labels.append(f"electron_energy[{zone_id}]")
                cursor += 1
            if self.evolves_heavy_energy:
                heavy_energy_indices[zone_id] = cursor
                labels.append(f"gas_internal_energy[{zone_id}]")
                cursor += 1
        coverage_start = cursor
        for surface_id, species_id in self.surface_coverage_keys:
            coverage_indices[(surface_id, species_id)] = cursor
            labels.append(f"coverage[{surface_id},{species_id}]")
            cursor += 1
        object.__setattr__(self, "density_slices", _frozen_mapping(density_slices))
        object.__setattr__(
            self, "electron_energy_indices", _frozen_mapping(energy_indices)
        )
        object.__setattr__(
            self,
            "heavy_energy_indices",
            _frozen_mapping(heavy_energy_indices),
        )
        object.__setattr__(
            self,
            "surface_coverage_indices",
            _frozen_mapping(coverage_indices),
        )
        object.__setattr__(
            self, "surface_coverage_slice", slice(coverage_start, cursor)
        )
        extension_start = cursor
        labels.extend(self.extension_labels)
        cursor += len(self.extension_labels)
        object.__setattr__(self, "extension_slice", slice(extension_start, cursor))
        if len(set(labels)) != len(labels):
            raise ModelConfigurationError("State labels must be globally unique")
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "size", cursor)


@dataclass(frozen=True)
class ZoneTermLedger:
    zone_id: str
    reaction_rates_m3_s: Mapping[str, float]
    absorbed_power_J_m3_s: float
    reaction_energy_loss_J_m3_s: float
    wall_energy_loss_J_m3_s: float
    gas_power_J_m3_s: float
    gas_reaction_heating_J_m3_s: float
    surface_reaction_heating_J_m3_s: float
    wall_heavy_energy_exchange_J_m3_s: float
    elastic_heating_J_m3_s: float
    transport_species_source_m3_s: Mapping[str, float]
    transport_electron_energy_J_m3_s: float
    transport_heavy_energy_J_m3_s: float
    inlet_heavy_energy_J_m3_s: float
    surface_rates_m2_s: Mapping[str, float]
    wall_fluxes: tuple[WallFluxRecord, ...]
    wall_species_energy_J_m3_s: float = 0.0
    surface_species_energy_J_m3_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reaction_rates_m3_s",
            _frozen_mapping(self.reaction_rates_m3_s),
        )
        object.__setattr__(
            self,
            "transport_species_source_m3_s",
            _frozen_mapping(self.transport_species_source_m3_s),
        )
        object.__setattr__(
            self,
            "surface_rates_m2_s",
            _frozen_mapping(self.surface_rates_m2_s),
        )


@dataclass(frozen=True)
class ModelEvaluation:
    derivative: np.ndarray
    electron_states: Mapping[str, ElectronState]
    kinetics_by_zone: Mapping[str, ElectronKineticsResult]
    gas_temperature_K_by_zone: Mapping[str, float]
    charge_residual_m3_by_zone: Mapping[str, float]
    ledger_by_zone: Mapping[str, ZoneTermLedger]
    power_coupling: PowerCouplingResult | None = None

    def __post_init__(self) -> None:
        derivative = np.array(self.derivative, dtype=float, copy=True)
        derivative.setflags(write=False)
        object.__setattr__(self, "derivative", derivative)
        object.__setattr__(
            self, "electron_states", _frozen_mapping(self.electron_states)
        )
        object.__setattr__(
            self, "kinetics_by_zone", _frozen_mapping(self.kinetics_by_zone)
        )
        object.__setattr__(
            self,
            "gas_temperature_K_by_zone",
            _frozen_mapping(self.gas_temperature_K_by_zone),
        )
        object.__setattr__(
            self,
            "charge_residual_m3_by_zone",
            _frozen_mapping(self.charge_residual_m3_by_zone),
        )
        object.__setattr__(self, "ledger_by_zone", _frozen_mapping(self.ledger_by_zone))


def _public_zone_ledger(payload: RuntimeZoneLedger) -> ZoneTermLedger:
    """Materialize the stable public ledger at the facade boundary."""

    return ZoneTermLedger(
        zone_id=payload.zone_id,
        reaction_rates_m3_s=payload.reaction_rates_m3_s,
        absorbed_power_J_m3_s=payload.absorbed_power_J_m3_s,
        reaction_energy_loss_J_m3_s=payload.reaction_energy_loss_J_m3_s,
        wall_energy_loss_J_m3_s=payload.wall_energy_loss_J_m3_s,
        gas_power_J_m3_s=payload.gas_power_J_m3_s,
        gas_reaction_heating_J_m3_s=payload.gas_reaction_heating_J_m3_s,
        surface_reaction_heating_J_m3_s=payload.surface_reaction_heating_J_m3_s,
        wall_heavy_energy_exchange_J_m3_s=(payload.wall_heavy_energy_exchange_J_m3_s),
        elastic_heating_J_m3_s=payload.elastic_heating_J_m3_s,
        transport_species_source_m3_s=payload.transport_species_source_m3_s,
        transport_electron_energy_J_m3_s=(payload.transport_electron_energy_J_m3_s),
        transport_heavy_energy_J_m3_s=payload.transport_heavy_energy_J_m3_s,
        inlet_heavy_energy_J_m3_s=payload.inlet_heavy_energy_J_m3_s,
        surface_rates_m2_s=payload.surface_rates_m2_s,
        wall_fluxes=payload.wall_fluxes,
        wall_species_energy_J_m3_s=payload.wall_species_energy_J_m3_s,
        surface_species_energy_J_m3_s=payload.surface_species_energy_J_m3_s,
    )


def _public_model_evaluation(payload: RuntimeEvaluation) -> ModelEvaluation:
    """Convert private runtime data to the documented diagnostic object."""

    return ModelEvaluation(
        derivative=payload.derivative,
        electron_states=payload.electron_states,
        kinetics_by_zone=payload.kinetics_by_zone,
        gas_temperature_K_by_zone=payload.gas_temperature_K_by_zone,
        charge_residual_m3_by_zone=payload.charge_residual_m3_by_zone,
        ledger_by_zone={
            zone_id: _public_zone_ledger(ledger)
            for zone_id, ledger in payload.ledger_by_zone.items()
        },
        power_coupling=payload.power_coupling,
    )


@dataclass
class BoundSegmentRHS:
    """RHS with one segment captured and exact repeated evaluations cached."""

    model: CompiledGlobalModel
    segment: RecipeSegment
    domain_atol: np.ndarray
    last_time_s: float | None = field(init=False, default=None)
    _cached_time_s: float | None = field(init=False, default=None, repr=False)
    _cached_state: np.ndarray | None = field(init=False, default=None, repr=False)
    _cached_derivative: np.ndarray | None = field(init=False, default=None, repr=False)

    def __call__(self, time_s: float, state: np.ndarray) -> np.ndarray:
        time = float(time_s)
        values = np.asarray(state, dtype=float)
        self.last_time_s = time
        if (
            self._cached_time_s == time
            and self._cached_state is not None
            and np.array_equal(self._cached_state, values)
        ):
            return _required_cached_derivative(self._cached_derivative)
        derivative = self.model.evaluate_derivative(
            time,
            values,
            self.segment,
            domain_atol=self.domain_atol,
        )
        self._cached_time_s = time
        self._cached_state = np.array(values, copy=True)
        self._cached_derivative = derivative
        return derivative

    def rhs(self, time_s: float, state: np.ndarray) -> np.ndarray:
        return self(time_s, state)

    @property
    def jac_sparsity(self) -> csr_matrix:
        return self.model.jac_sparsity


@dataclass
class CompiledGlobalModel:
    """Executable 0D/multi-zone global model with compiled term providers.

    Chemistry, wall loss, directed inter-zone transport, electron kinetics,
    and power ports are composed once here so the stiff RHS performs only
    array operations and local model evaluations.
    """

    chemistry: CompiledChemistryLike
    zones: tuple[Zone, ...]
    segments: tuple[RecipeSegment, ...]
    electron_closure: ElectronClosure
    wall_boundaries: tuple[WallBoundary, ...] = ()
    transport: CompiledTransport | None = None
    power_coordinator: PowerCoordinator | None = None
    electron_kinetics_by_zone: Mapping[str, TabulatedElectronKinetics] = field(
        default_factory=dict
    )
    heavy_energy_closure: HeavyEnergyClosure | None = None
    surface_model: CompiledSurfaceModel | None = None
    extension_accumulator: ExtensionStateAccumulator | None = None
    elastic_heating_evaluator: ElasticHeatingEvaluator | None = None
    electron_density_provider: ElectronDensityProvider | None = None
    domain_atol: float | np.ndarray = 0.0
    layout: StateLayout = field(init=False)
    state_lower_bounds: np.ndarray = field(init=False)
    state_upper_bounds: np.ndarray = field(init=False)
    _jac_sparsity: csr_matrix = field(init=False, repr=False)
    _chemistry_data: CompiledChemistryData = field(init=False, repr=False)
    _zone_by_id: Mapping[str, Zone] = field(init=False, repr=False)
    _active_reaction_indices_by_zone: Mapping[str, tuple[int, ...]] = field(
        init=False, repr=False
    )
    _walls_by_zone: Mapping[str, tuple[CompiledWallBoundary, ...]] = field(
        init=False, repr=False
    )
    _transport_by_segment: Mapping[str, SegmentTransport] = field(
        init=False, repr=False
    )
    _compiled_immutable: bool = field(
        init=False, default=False, repr=False, compare=False
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_compiled_immutable", False):
            raise AttributeError("CompiledGlobalModel is immutable after compilation")
        object.__setattr__(self, name, value)

    @property
    def jac_sparsity(self) -> csr_matrix:
        """Return a mutable solver-compatible copy of the compiled pattern."""

        return self._jac_sparsity.copy()

    @property
    def species_ids(self) -> tuple[str, ...]:
        return self._chemistry_data.species_ids

    @property
    def _species_index(self) -> Mapping[str, int]:
        return self._chemistry_data.species_index

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return self._chemistry_data.reaction_ids

    @property
    def charges(self) -> np.ndarray:
        return self._chemistry_data.charges

    @property
    def masses_kg(self) -> np.ndarray:
        return self._chemistry_data.masses_kg

    @property
    def stoichiometry(self) -> np.ndarray:
        return self._chemistry_data.stoichiometry

    @property
    def reactant_orders(self) -> np.ndarray:
        return self._chemistry_data.reactant_orders

    @property
    def electron_orders(self) -> np.ndarray:
        return self._chemistry_data.electron_orders

    @property
    def energy_loss_eV(self) -> np.ndarray:
        return self._chemistry_data.energy_loss_eV

    @property
    def electron_energy_transfer_eV(self) -> np.ndarray:
        """Signed reaction energy deposited into electrons per event."""

        return self._chemistry_data.electron_energy_transfer_eV

    @property
    def gas_heating_eV(self) -> np.ndarray:
        return self._chemistry_data.gas_heating_eV

    @property
    def chemistry_jacobian_species_pattern(self) -> np.ndarray:
        return self._chemistry_data.jacobian_species_pattern

    @property
    def reaction_zones(self) -> tuple[tuple[str, ...], ...]:
        return self._chemistry_data.reaction_zones

    @property
    def rate_evaluators(self) -> tuple[RateEvaluator, ...]:
        return self._chemistry_data.rate_evaluators

    def __post_init__(self) -> None:
        self.zones = tuple(self.zones)
        self.segments = tuple(self.segments)
        self.wall_boundaries = tuple(self.wall_boundaries)
        self.electron_kinetics_by_zone = _frozen_mapping(self.electron_kinetics_by_zone)
        self._chemistry_data = compile_chemistry_data(self.chemistry)
        extension_labels = self._extension_labels()
        DomainValidator(self).validate()
        self.layout = StateLayout(
            species_ids=self.species_ids,
            zone_ids=tuple(zone.zone_id for zone in self.zones),
            evolves_electron_energy=bool(self.electron_closure.evolves_energy),
            evolves_heavy_energy=self.heavy_energy_closure is not None,
            surface_coverage_keys=self._surface_coverage_keys(),
            extension_labels=extension_labels,
        )
        self.domain_atol = self._compile_domain_atol(self.domain_atol)
        self.state_lower_bounds, self.state_upper_bounds = self._compile_state_bounds()
        self._zone_by_id = _frozen_mapping({zone.zone_id: zone for zone in self.zones})
        active_reactions_by_zone: dict[str, tuple[int, ...]] = {}
        for zone in self.zones:
            active_reactions_by_zone[zone.zone_id] = tuple(
                reaction_index
                for reaction_index, selected_zones in enumerate(self.reaction_zones)
                if not selected_zones or zone.zone_id in selected_zones
            )
        self._active_reaction_indices_by_zone = _frozen_mapping(
            active_reactions_by_zone
        )
        walls_by_zone: dict[str, list[CompiledWallBoundary]] = {
            zone.zone_id: [] for zone in self.zones
        }
        for boundary in self.wall_boundaries:
            walls_by_zone[boundary.zone_id].append(
                compile_wall_boundary(
                    boundary=boundary,
                    species_ids=self.species_ids,
                    charges=self.charges,
                    masses_kg=self.masses_kg,
                )
            )
        self._walls_by_zone = _frozen_mapping(
            {zone_id: tuple(values) for zone_id, values in walls_by_zone.items()}
        )
        self._transport_by_segment = self._compile_segment_transport()
        self._jac_sparsity = _readonly_csr(JacobianBuilder(self).build())
        object.__setattr__(self, "_compiled_immutable", True)

    def _compile_domain_atol(self, value: float | np.ndarray) -> np.ndarray:
        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            array = np.full(self.layout.size, float(raw), dtype=float)
        elif raw.shape == (self.layout.size,):
            array = np.array(raw, dtype=float, copy=True)
        else:
            raise ModelConfigurationError(
                f"domain_atol has shape {raw.shape}, expected scalar or "
                f"{(self.layout.size,)}"
            )
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ModelConfigurationError("domain_atol must be finite and nonnegative")
        array.setflags(write=False)
        return array

    def _extension_bounds(self, name: str, unbounded: float) -> np.ndarray:
        accumulator = self.extension_accumulator
        if accumulator is None:
            return np.empty(0, dtype=float)
        raw = tuple(getattr(accumulator, name))
        expected = len(self.layout.extension_labels)
        if len(raw) != expected:
            description = name.replace("_", " ")
            raise ModelConfigurationError(
                f"Extension {description} must match its compiled state block"
            )
        try:
            bounds = np.asarray(
                [unbounded if value is None else float(value) for value in raw],
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelConfigurationError(
                f"Extension {name.replace('_', ' ')} must be numeric or None"
            ) from exc
        if np.any(np.isnan(bounds)):
            raise ModelConfigurationError(
                f"Extension {name.replace('_', ' ')} must not contain NaN"
            )
        return bounds

    def _compile_state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.layout.size, dtype=float)
        upper = np.full(self.layout.size, float("inf"), dtype=float)
        if self.extension_accumulator is not None:
            extension_lower = self._extension_bounds("lower_bounds", float("-inf"))
            extension_upper = self._extension_bounds("upper_bounds", float("inf"))
            if np.any(extension_lower > extension_upper):
                raise ModelConfigurationError(
                    "Extension upper bounds must not be below lower bounds"
                )
            lower[self.layout.extension_slice] = extension_lower
            upper[self.layout.extension_slice] = extension_upper
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def _surface_coverage_keys(self) -> tuple[tuple[str, str], ...]:
        if self.surface_model is None:
            return ()
        state_index = self.surface_model.layout.state_index
        return tuple(sorted(state_index, key=state_index.__getitem__))

    def _extension_labels(self) -> tuple[str, ...]:
        if self.extension_accumulator is None:
            return ()
        labels = tuple(str(value) for value in self.extension_accumulator.labels)
        if not labels or "" in labels:
            raise ModelConfigurationError(
                "Extension accumulator needs nonempty state labels"
            )
        if len(set(labels)) != len(labels):
            raise ModelConfigurationError("Extension state labels must be unique")
        initial = np.asarray(self.extension_accumulator.initial_state(), dtype=float)
        if initial.shape != (len(labels),) or not np.all(np.isfinite(initial)):
            raise ModelConfigurationError(
                "Extension initial state must be one finite value per label"
            )
        return labels

    def _compile_segment_transport(self) -> Mapping[str, SegmentTransport]:
        if self.transport is None:
            return _frozen_mapping({})
        result: dict[str, SegmentTransport] = {}
        expected_particle_shape = (len(self.zones), len(self.species_ids))
        for segment in self.segments:
            forcing = segment.transport
            if forcing is None:
                forcing = SegmentTransport.zeros(*expected_particle_shape)
            if not isinstance(forcing, SegmentTransport):
                raise ModelConfigurationError(
                    f"Segment {segment.segment_id!r} transport must be SegmentTransport"
                )
            if forcing.particle_source_m3_s.shape != expected_particle_shape:
                raise ModelConfigurationError(
                    f"Segment {segment.segment_id!r} transport source has shape "
                    f"{forcing.particle_source_m3_s.shape}, expected "
                    f"{expected_particle_shape}"
                )
            result[segment.segment_id] = forcing
        return _frozen_mapping(result)

    def bind_segment(
        self,
        segment: RecipeSegment,
        *,
        domain_atol: float | np.ndarray | None = None,
    ) -> BoundSegmentRHS:
        if segment not in self.segments:
            raise ModelConfigurationError(
                f"Segment {segment.segment_id!r} is not part of this model"
            )
        tolerance = (
            np.asarray(self.domain_atol, dtype=float)
            if domain_atol is None
            else self._compile_domain_atol(domain_atol)
        )
        return BoundSegmentRHS(
            model=self,
            segment=segment,
            domain_atol=tolerance,
        )

    def initial_state(self, initial: InitialState) -> np.ndarray:
        return build_initial_state(self, initial)

    def evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool = True,
    ) -> ModelEvaluation:
        """Evaluate the full diagnostic view used by output and audit paths."""

        return _public_model_evaluation(
            RuntimeEvaluator(self).evaluate(
                time_s,
                state,
                segment,
                collect_ledger=collect_ledger,
            )
        )

    def evaluate_derivative(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        domain_atol: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate only ``dy/dt`` without materializing diagnostic wrappers."""

        return RuntimeEvaluator(self).evaluate_derivative(
            time_s,
            state,
            segment,
            domain_atol=domain_atol,
        )


__all__ = [
    "BoundSegmentRHS",
    "CompiledChemistryLike",
    "CompiledGlobalModel",
    "ElasticHeatingEvaluator",
    "ElectronDensityProvider",
    "ExtensionStateAccumulator",
    "ModelEvaluation",
    "StateLayout",
    "ZoneTermLedger",
]
