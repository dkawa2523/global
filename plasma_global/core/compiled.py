"""Compiled, volume-averaged plasma chemistry and energy equations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
from scipy.sparse import csr_matrix

from plasma_global.core.domain import InitialState, RecipeSegment, Zone
from plasma_global.core.exceptions import ModelConfigurationError, StateDomainError
from plasma_global.core.transport import CompiledTransport, SegmentTransport
from plasma_global.models.electrons import (
    ELEMENTARY_CHARGE_C,
    ElectronClosure,
    ElectronState,
    resolved_electron_density,
)
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)
from plasma_global.models.power import PowerCoordinator, PowerCouplingResult
from plasma_global.models.rates import (
    ConstantRate,
    DensityView,
    RateContext,
    RateEvaluator,
)
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceEvaluation
from plasma_global.models.walls import (
    BoundaryReaction,
    CompiledWallBoundary,
    WallBoundary,
    WallFluxRecord,
    compile_wall_boundary,
    evaluate_compiled_wall_boundary,
)


class CompiledChemistryLike(Protocol):
    """Narrow interface expected from an input or chemistry compiler.

    Species arrays contain only evolved heavy species.  Electron reactant
    orders are separate because electron density is imposed by quasineutrality.
    Stoichiometry is indexed ``[reaction, species]``.
    """

    @property
    def species_ids(self) -> Sequence[str]: ...

    @property
    def charges(self) -> Any: ...

    @property
    def masses_kg(self) -> Any: ...

    @property
    def reaction_ids(self) -> Sequence[str]: ...

    @property
    def stoichiometry(self) -> Any: ...

    @property
    def reactant_orders(self) -> Any: ...

    @property
    def electron_orders(self) -> Any: ...

    @property
    def rate_evaluators(self) -> Sequence[Any]: ...

    @property
    def energy_loss_eV(self) -> Any: ...

    @property
    def gas_heating_eV(self) -> Any: ...

    @property
    def reaction_zones(self) -> Sequence[Sequence[str]]: ...

    @property
    def jacobian_species_pattern(self) -> Any: ...


class ElectronDensityProvider(Protocol):
    """Narrow optional closure for prescribed electron-density profiles."""

    def __call__(self, time_s: float, zone_id: str) -> float: ...


class ExtensionStateAccumulator(Protocol):
    """One-way optional state block supplied by the composition root."""

    @property
    def labels(self) -> Sequence[str]: ...

    @property
    def lower_bounds(self) -> Sequence[float | None]: ...

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


def _required_cached_derivative(derivative: np.ndarray | None) -> np.ndarray:
    if derivative is None:
        raise RuntimeError("RHS derivative cache is internally inconsistent")
    return derivative


def _required_heavy_energy_closure(
    closure: HeavyEnergyClosure | None,
) -> HeavyEnergyClosure:
    if closure is None:
        raise RuntimeError("Heavy-energy state requires an energy closure")
    return closure


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
        object.__setattr__(self, "density_slices", MappingProxyType(density_slices))
        object.__setattr__(
            self, "electron_energy_indices", MappingProxyType(energy_indices)
        )
        object.__setattr__(
            self,
            "heavy_energy_indices",
            MappingProxyType(heavy_energy_indices),
        )
        object.__setattr__(
            self,
            "surface_coverage_indices",
            MappingProxyType(coverage_indices),
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reaction_rates_m3_s",
            MappingProxyType(dict(self.reaction_rates_m3_s)),
        )
        object.__setattr__(
            self,
            "transport_species_source_m3_s",
            MappingProxyType(dict(self.transport_species_source_m3_s)),
        )
        object.__setattr__(
            self,
            "surface_rates_m2_s",
            MappingProxyType(dict(self.surface_rates_m2_s)),
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
            self, "electron_states", MappingProxyType(dict(self.electron_states))
        )
        object.__setattr__(
            self, "kinetics_by_zone", MappingProxyType(dict(self.kinetics_by_zone))
        )
        object.__setattr__(
            self,
            "gas_temperature_K_by_zone",
            MappingProxyType(dict(self.gas_temperature_K_by_zone)),
        )
        object.__setattr__(
            self,
            "charge_residual_m3_by_zone",
            MappingProxyType(dict(self.charge_residual_m3_by_zone)),
        )
        object.__setattr__(
            self, "ledger_by_zone", MappingProxyType(dict(self.ledger_by_zone))
        )


@dataclass(frozen=True, slots=True)
class _PreparedEvaluationState:
    values: np.ndarray
    domain_atol: np.ndarray
    density_by_zone: np.ndarray
    reaction_density_by_zone: np.ndarray
    electron_energy_by_zone: np.ndarray | None
    closure_electron_energy_by_zone: np.ndarray | None
    heavy_energy_by_zone: np.ndarray | None
    net_charge_by_zone: dict[str, float]
    electron_density_by_zone: dict[str, float]
    charge_residual_by_zone: dict[str, float]
    neutral_density_by_zone: dict[str, float]
    mean_energy_for_power: dict[str, float | None]
    gas_temperature_by_zone: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedZoneState:
    density: np.ndarray
    reaction_density: np.ndarray
    electron_energy: float | None
    closure_electron_energy: float | None
    heavy_energy: float | None
    net_charge: float
    electron_density: float
    charge_residual: float
    neutral_density: float
    mean_energy_for_power: float | None


@dataclass(frozen=True, slots=True)
class _PowerEvaluation:
    coupling: PowerCouplingResult | None
    kinetics_by_zone: dict[str, ElectronKineticsResult]
    electron_power_W_by_zone: Mapping[str, float]
    gas_power_W_by_zone: Mapping[str, float]
    reduced_field_Td_by_zone: dict[str, float]
    electron_states: dict[str, ElectronState]


@dataclass(frozen=True, slots=True)
class _TransportEvaluation:
    density_rhs: np.ndarray
    electron_energy_rhs: np.ndarray | None
    heavy_energy_rhs: np.ndarray | None
    inlet_heavy_energy: np.ndarray


@dataclass(frozen=True, slots=True)
class _ReactionWallEvaluation:
    reaction_rates_by_zone: np.ndarray
    reaction_energy_loss: np.ndarray
    gas_reaction_heating: np.ndarray
    wall_species_rhs: np.ndarray
    wall_energy_loss: np.ndarray
    wall_records_by_zone: dict[str, list[WallFluxRecord]]
    ion_flux_m2_s: dict[tuple[str, str], float]
    ion_energy_eV: dict[str, float]


@dataclass(frozen=True, slots=True)
class _ZoneReactionWallEvaluation:
    rates: np.ndarray
    reaction_energy_loss: float
    gas_reaction_heating: float
    wall_species_rhs: np.ndarray
    wall_energy_loss: float
    wall_records: list[WallFluxRecord]
    ion_flux_m2_s: dict[tuple[str, str], float]
    ion_energy_eV: dict[str, float]


@dataclass(frozen=True, slots=True)
class _SurfaceTerms:
    evaluation: SurfaceEvaluation | None
    rates_by_zone: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class _EnergyTerms:
    wall_heavy_exchange: np.ndarray
    elastic_heating: np.ndarray


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
    jac_sparsity: csr_matrix = field(init=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_compiled_immutable", False):
            raise AttributeError("CompiledGlobalModel is immutable after compilation")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.zones = tuple(self.zones)
        self.segments = tuple(self.segments)
        self.wall_boundaries = tuple(self.wall_boundaries)
        self.electron_kinetics_by_zone = MappingProxyType(
            dict(self.electron_kinetics_by_zone)
        )
        self._compile_chemistry()
        extension_labels = self._extension_labels()
        self._validate_domain()
        self.layout = StateLayout(
            species_ids=self.species_ids,
            zone_ids=tuple(zone.zone_id for zone in self.zones),
            evolves_electron_energy=bool(self.electron_closure.evolves_energy),
            evolves_heavy_energy=self.heavy_energy_closure is not None,
            surface_coverage_keys=self._surface_coverage_keys(),
            extension_labels=extension_labels,
        )
        self.domain_atol = self._compile_domain_atol(self.domain_atol)
        self.state_lower_bounds = self._compile_state_lower_bounds()
        self._zone_by_id = {zone.zone_id: zone for zone in self.zones}
        self._active_reactions_by_zone = {
            zone.zone_id: np.array(
                [
                    not selected_zones or zone.zone_id in selected_zones
                    for selected_zones in self.reaction_zones
                ],
                dtype=bool,
            )
            for zone in self.zones
        }
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
        self._walls_by_zone = {
            zone_id: tuple(values) for zone_id, values in walls_by_zone.items()
        }
        self._transport_by_segment = self._compile_segment_transport()
        self.jac_sparsity = self._build_jac_sparsity()
        object.__setattr__(self, "_compiled_immutable", True)

    def _compile_chemistry(self) -> None:
        self.species_ids = tuple(str(value) for value in self.chemistry.species_ids)
        self._species_index = MappingProxyType(
            {species_id: index for index, species_id in enumerate(self.species_ids)}
        )
        self.reaction_ids = tuple(str(value) for value in self.chemistry.reaction_ids)
        if not self.species_ids or len(set(self.species_ids)) != len(self.species_ids):
            raise ModelConfigurationError(
                "Chemistry needs unique, non-empty heavy-species IDs"
            )
        if len(set(self.reaction_ids)) != len(self.reaction_ids):
            raise ModelConfigurationError("Reaction IDs must be unique")
        n_species = len(self.species_ids)
        n_reactions = len(self.reaction_ids)
        self.charges = self._readonly_array(
            self.chemistry.charges, shape=(n_species,), name="charges"
        )
        self.masses_kg = self._readonly_array(
            self.chemistry.masses_kg, shape=(n_species,), name="masses_kg"
        )
        self.stoichiometry = self._readonly_array(
            self.chemistry.stoichiometry,
            shape=(n_reactions, n_species),
            name="stoichiometry",
        )
        self.reactant_orders = self._readonly_array(
            self.chemistry.reactant_orders,
            shape=(n_reactions, n_species),
            name="reactant_orders",
        )
        self.electron_orders = self._readonly_array(
            self.chemistry.electron_orders,
            shape=(n_reactions,),
            name="electron_orders",
        )
        self.energy_loss_eV = self._readonly_array(
            self.chemistry.energy_loss_eV,
            shape=(n_reactions,),
            name="energy_loss_eV",
        )
        raw_jacobian_pattern = np.asarray(
            self.chemistry.jacobian_species_pattern, dtype=bool
        )
        if raw_jacobian_pattern.shape != (n_species, n_species):
            raise ModelConfigurationError(
                "jacobian_species_pattern has shape "
                f"{raw_jacobian_pattern.shape}, expected {(n_species, n_species)}"
            )
        self.chemistry_jacobian_species_pattern = np.array(
            raw_jacobian_pattern, dtype=bool, copy=True
        )
        self.chemistry_jacobian_species_pattern.setflags(write=False)
        self.gas_heating_eV = self._readonly_array(
            self.chemistry.gas_heating_eV,
            shape=(n_reactions,),
            name="gas_heating_eV",
        )
        raw_reaction_zones = self.chemistry.reaction_zones
        if len(raw_reaction_zones) != n_reactions:
            raise ModelConfigurationError(
                "reaction_zones length must match reaction_ids"
            )
        self.reaction_zones = tuple(
            tuple(str(zone_id) for zone_id in zones) for zones in raw_reaction_zones
        )
        if np.any(self.masses_kg <= 0.0):
            raise ModelConfigurationError(
                "Every evolved species must have positive mass_kg"
            )
        if np.any(self.reactant_orders < 0.0) or np.any(self.electron_orders < 0.0):
            raise ModelConfigurationError(
                "Mass-action reaction orders must be non-negative"
            )
        if np.any(self.energy_loss_eV < 0.0):
            raise ModelConfigurationError("Electron energy losses must be non-negative")
        if np.any(self.gas_heating_eV < 0.0):
            raise ModelConfigurationError("Gas reaction heating must be non-negative")
        raw_evaluators = tuple(self.chemistry.rate_evaluators)
        if len(raw_evaluators) != n_reactions:
            raise ModelConfigurationError(
                "rate_evaluators length must match reaction_ids"
            )
        evaluators: list[RateEvaluator] = []
        for evaluator in raw_evaluators:
            if isinstance(evaluator, (float, int)):
                evaluators.append(ConstantRate(float(evaluator)))
            elif callable(evaluator):
                evaluators.append(cast(RateEvaluator, evaluator))
            else:
                raise ModelConfigurationError(
                    "Every reaction rate evaluator must be callable or numeric"
                )
        self.rate_evaluators = tuple(evaluators)

    @staticmethod
    def _readonly_array(
        values: Any, *, shape: tuple[int, ...], name: str
    ) -> np.ndarray:
        array = np.array(values, dtype=float, copy=True)
        if array.shape != shape:
            raise ModelConfigurationError(
                f"{name} has shape {array.shape}, expected {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ModelConfigurationError(f"{name} must contain only finite values")
        array.setflags(write=False)
        return array

    def _compile_domain_atol(self, value: float | np.ndarray) -> np.ndarray:
        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            array = np.full(self.layout.size, float(raw), dtype=float)
        elif raw.shape == (self.layout.size,):
            array = np.array(raw, dtype=float, copy=True)
        else:
            raise ModelConfigurationError(
                f"domain_atol has shape {raw.shape}, expected scalar or {(self.layout.size,)}"
            )
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ModelConfigurationError("domain_atol must be finite and nonnegative")
        array.setflags(write=False)
        return array

    def _compile_state_lower_bounds(self) -> np.ndarray:
        bounds = np.zeros(self.layout.size, dtype=float)
        if self.extension_accumulator is not None:
            raw = tuple(self.extension_accumulator.lower_bounds)
            expected = (
                self.layout.extension_slice.stop - self.layout.extension_slice.start
            )
            if len(raw) != expected:
                raise ModelConfigurationError(
                    "Extension lower bounds must match its compiled state block"
                )
            extension_bounds = np.asarray(
                [float("-inf") if value is None else float(value) for value in raw]
            )
            if np.any(np.isnan(extension_bounds)):
                raise ModelConfigurationError("Extension lower bounds must not be NaN")
            bounds[self.layout.extension_slice] = extension_bounds
        bounds.setflags(write=False)
        return bounds

    def _surface_coverage_keys(self) -> tuple[tuple[str, str], ...]:
        if self.surface_model is None:
            return ()

        def state_position(item: tuple[tuple[str, str], int]) -> int:
            return item[1]

        return tuple(
            key
            for key, _index in sorted(
                self.surface_model.layout.state_index.items(),
                key=state_position,
            )
        )

    def _extension_labels(self) -> tuple[str, ...]:
        if self.extension_accumulator is None:
            return ()
        labels = tuple(str(value) for value in self.extension_accumulator.labels)
        if not labels or any(not value for value in labels):
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
            return MappingProxyType({})
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
                    f"{forcing.particle_source_m3_s.shape}, expected {expected_particle_shape}"
                )
            result[segment.segment_id] = forcing
        return MappingProxyType(result)

    def _validate_domain(self) -> None:
        zone_ids, known_zones = self._validate_identifiers()
        self._validate_optional_models(zone_ids)
        known_surfaces = self._validate_surfaces(zone_ids, known_zones)
        self._validate_transport_and_power(zone_ids)
        self._validate_kinetics_and_reactions(known_zones)
        self._validate_segments(known_zones, known_surfaces)
        self._validate_electron_closure()
        self._validate_wall_boundaries(known_zones)

    def _validate_identifiers(self) -> tuple[tuple[str, ...], set[str]]:
        if not self.zones:
            raise ModelConfigurationError("At least one zone is required")
        zone_ids = tuple(zone.zone_id for zone in self.zones)
        if len(set(zone_ids)) != len(zone_ids):
            raise ModelConfigurationError("Zone IDs must be unique")
        if not self.segments:
            raise ModelConfigurationError("At least one recipe segment is required")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ModelConfigurationError("Recipe segment IDs must be unique")
        known_zones = set(zone_ids)
        return zone_ids, known_zones

    def _validate_optional_models(self, zone_ids: tuple[str, ...]) -> None:
        if self.heavy_energy_closure is not None:
            if self.heavy_energy_closure.cv_over_kb.shape != (len(self.species_ids),):
                raise ModelConfigurationError(
                    "HeavyEnergyClosure heat capacities must match compiled species order"
                )
            expected_zone_shape = (len(self.zones),)
            if (
                self.heavy_energy_closure.wall_temperature_K.shape
                != expected_zone_shape
                or self.heavy_energy_closure.wall_relaxation_s_inv.shape
                != expected_zone_shape
            ):
                raise ModelConfigurationError(
                    "HeavyEnergyClosure wall arrays must match compiled zone order"
                )
        if self.elastic_heating_evaluator is not None and not callable(
            self.elastic_heating_evaluator
        ):
            raise ModelConfigurationError("elastic_heating_evaluator must be callable")
        if self.electron_density_provider is not None and not callable(
            self.electron_density_provider
        ):
            raise ModelConfigurationError("electron_density_provider must be callable")

    def _validate_surfaces(
        self, zone_ids: tuple[str, ...], known_zones: set[str]
    ) -> set[str]:
        if self.surface_model is None:
            return set()
        if self.surface_model.gas_species_ids != self.species_ids:
            raise ModelConfigurationError(
                "CompiledSurfaceModel gas species/order must match chemistry"
            )
        if self.surface_model.zone_ids != zone_ids:
            raise ModelConfigurationError(
                "CompiledSurfaceModel zone IDs/order must match model zones"
            )
        model_volumes = np.array([zone.volume_m3 for zone in self.zones])
        if not np.array_equal(self.surface_model.zone_volumes_m3, model_volumes):
            raise ModelConfigurationError(
                "CompiledSurfaceModel volumes/order must match model zones"
            )
        surface_ids = tuple(
            surface.surface_id for surface in self.surface_model.surfaces
        )
        if len(set(surface_ids)) != len(surface_ids):
            raise ModelConfigurationError("Surface IDs must be unique")
        unknown_surface_zones = {
            surface.zone_id for surface in self.surface_model.surfaces
        } - known_zones
        if unknown_surface_zones:
            raise ModelConfigurationError(
                f"Surfaces reference unknown zones {sorted(unknown_surface_zones)}"
            )
        self._validate_surface_wall_bindings(set(surface_ids))
        return set(surface_ids)

    def _validate_surface_wall_bindings(self, surface_ids: set[str]) -> None:
        boundary_surface_ids = [
            boundary.surface_id
            for boundary in self.wall_boundaries
            if boundary.surface_id is not None
        ]
        if len(set(boundary_surface_ids)) != len(boundary_surface_ids):
            raise ModelConfigurationError(
                "At most one wall transport boundary may own each surface"
            )
        unknown_wall_surfaces = set(boundary_surface_ids) - surface_ids
        if unknown_wall_surfaces:
            raise ModelConfigurationError(
                f"Wall boundaries reference unknown surfaces {sorted(unknown_wall_surfaces)}"
            )
        self._validate_wall_surface_zones()

    def _validate_wall_surface_zones(self) -> None:
        if self.surface_model is None:
            return
        surface_zones = {
            surface.surface_id: surface.zone_id
            for surface in self.surface_model.surfaces
        }
        mismatched_wall_zones = [
            boundary.surface_id
            for boundary in self.wall_boundaries
            if boundary.surface_id is not None
            and surface_zones[boundary.surface_id] != boundary.zone_id
        ]
        if mismatched_wall_zones:
            raise ModelConfigurationError(
                "Wall boundary zones disagree with their surfaces: "
                f"{sorted(mismatched_wall_zones)}"
            )

    def _validate_transport_and_power(self, zone_ids: tuple[str, ...]) -> None:
        if self.transport is not None:
            if self.transport.n_zones != len(self.zones):
                raise ModelConfigurationError(
                    "Transport zone count does not match model zones"
                )
            if self.transport.n_species != len(self.species_ids):
                raise ModelConfigurationError(
                    "Transport species count does not match compiled chemistry"
                )
            model_volumes = np.array([zone.volume_m3 for zone in self.zones])
            if not np.array_equal(self.transport.volumes_m3, model_volumes):
                raise ModelConfigurationError(
                    "Transport volumes/order must exactly match CompiledGlobalModel.zones"
                )
        elif any(segment.transport is not None for segment in self.segments):
            raise ModelConfigurationError(
                "Recipe segment transport forcing requires CompiledGlobalModel.transport"
            )
        if (
            self.power_coordinator is not None
            and tuple(self.power_coordinator.zone_ids) != zone_ids
        ):
            raise ModelConfigurationError(
                "PowerCoordinator zone_ids/order must match CompiledGlobalModel.zones"
            )

    def _validate_kinetics_and_reactions(self, known_zones: set[str]) -> None:
        unknown_kinetics = set(self.electron_kinetics_by_zone) - known_zones
        if unknown_kinetics:
            raise ModelConfigurationError(
                f"Electron kinetics reference unknown zones {sorted(unknown_kinetics)}"
            )
        required_lookup = (
            "mean_energy"
            if self.electron_closure.mode == "electron_energy"
            else "local_field"
        )
        for zone_id, kinetics in self.electron_kinetics_by_zone.items():
            if kinetics.lookup != required_lookup:
                raise ModelConfigurationError(
                    f"Zone {zone_id!r} electron closure {self.electron_closure.mode!r} "
                    f"requires a {required_lookup!r} kinetics table, got {kinetics.lookup!r}"
                )
        for reaction_id, selected_zones in zip(self.reaction_ids, self.reaction_zones):
            unknown = set(selected_zones) - known_zones
            if unknown:
                raise ModelConfigurationError(
                    f"Reaction {reaction_id!r} selects unknown zones {sorted(unknown)}"
                )

    def _validate_segments(
        self, known_zones: set[str], known_surfaces: set[str]
    ) -> None:
        previous_end: float | None = None
        has_segment_electron_density = any(
            segment.prescribed_electron_density_m3_by_zone for segment in self.segments
        )
        for segment in self.segments:
            if previous_end is not None and segment.start_s != previous_end:
                raise ModelConfigurationError(
                    f"Recipe segments must be exactly contiguous: {previous_end} != {segment.start_s}"
                )
            previous_end = segment.end_s
            self._validate_segment_references(
                segment,
                known_zones=known_zones,
                known_surfaces=known_surfaces,
                require_electron_density=has_segment_electron_density,
            )

    def _validate_segment_references(
        self,
        segment: RecipeSegment,
        *,
        known_zones: set[str],
        known_surfaces: set[str],
        require_electron_density: bool,
    ) -> None:
        zone_references = (
            set(segment.absorbed_power_W_by_zone)
            | set(segment.reduced_field_Td_by_zone)
            | set(segment.wall_temperature_K_by_zone)
            | set(segment.prescribed_electron_density_m3_by_zone)
        )
        unknown_zones = zone_references - known_zones
        if unknown_zones:
            raise ModelConfigurationError(
                f"Recipe segment {segment.segment_id!r} references unknown zones "
                f"{sorted(unknown_zones)}"
            )
        electron_density_zones = set(segment.prescribed_electron_density_m3_by_zone)
        if require_electron_density and electron_density_zones != known_zones:
            raise ModelConfigurationError(
                f"Recipe segment {segment.segment_id!r} must prescribe electron "
                f"density for every zone {sorted(known_zones)}"
            )
        unknown_surfaces = (
            set(segment.surface_temperature_K_by_surface) - known_surfaces
        )
        if unknown_surfaces:
            raise ModelConfigurationError(
                f"Recipe segment {segment.segment_id!r} references unknown surfaces "
                f"{sorted(unknown_surfaces)}"
            )
        self._validate_segment_electron_controls(segment, known_zones)

    def _validate_segment_electron_controls(
        self, segment: RecipeSegment, known_zones: set[str]
    ) -> None:
        if (
            self.electron_closure.mode == "local_field"
            and self.power_coordinator is None
        ):
            missing = known_zones - set(segment.reduced_field_Td_by_zone)
            if missing:
                raise ModelConfigurationError(
                    f"local_field segment {segment.segment_id!r} lacks E/N for zones {sorted(missing)}"
                )
        if self.power_coordinator is None and segment.port_commands:
            raise ModelConfigurationError(
                f"Segment {segment.segment_id!r} has port commands but no PowerCoordinator"
            )
        if self.power_coordinator is not None:
            self.power_coordinator.validate_commands(segment.port_commands)

    def _validate_electron_closure(self) -> None:
        if self.electron_closure.mode not in {"electron_energy", "local_field"}:
            raise ModelConfigurationError(
                f"Unknown electron closure mode {self.electron_closure.mode!r}"
            )
        if bool(self.electron_closure.evolves_energy) != (
            self.electron_closure.mode == "electron_energy"
        ):
            raise ModelConfigurationError(
                "Electron closure mode and evolves_energy flag are inconsistent"
            )

    def _validate_wall_boundaries(self, known_zones: set[str]) -> None:
        species_index = {species_id: i for i, species_id in enumerate(self.species_ids)}
        for boundary in self.wall_boundaries:
            self._validate_wall_boundary(boundary, known_zones, species_index)

    def _validate_wall_boundary(
        self,
        boundary: WallBoundary,
        known_zones: set[str],
        species_index: Mapping[str, int],
    ) -> None:
        if boundary.zone_id not in known_zones:
            raise ModelConfigurationError(
                f"Wall boundary references unknown zone {boundary.zone_id!r}"
            )
        for reaction in boundary.reactions:
            self._validate_boundary_reaction(reaction, species_index)
        if boundary.transport_kind != "off":
            self._validate_boundary_probabilities(boundary)

    def _validate_boundary_reaction(
        self, reaction: BoundaryReaction, species_index: Mapping[str, int]
    ) -> None:
        incident_index = species_index.get(reaction.incident_species)
        if incident_index is None:
            raise ModelConfigurationError(
                f"Boundary reaction {reaction.reaction_id!r} has unknown incident species"
            )
        if self.charges[incident_index] <= 0.0:
            raise ModelConfigurationError(
                f"Boundary incident species {reaction.incident_species!r} must be a positive ion"
            )
        unknown_products = set(reaction.products) - set(self.species_ids)
        if unknown_products:
            raise ModelConfigurationError(
                f"Boundary reaction {reaction.reaction_id!r} has unknown products {sorted(unknown_products)}"
            )

    def _validate_boundary_probabilities(self, boundary: WallBoundary) -> None:
        branch_probability = {
            species_id: sum(
                reaction.probability
                for reaction in boundary.reactions
                if reaction.incident_species == species_id
            )
            for species_id, charge in zip(self.species_ids, self.charges, strict=True)
            if charge > 0.0
        }
        incomplete = {
            species_id: probability
            for species_id, probability in branch_probability.items()
            if not math.isclose(probability, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        }
        if incomplete:
            raise ModelConfigurationError(
                f"Wall {boundary.surface_id or boundary.zone_id!r} must define "
                "unit-probability boundary products for every positive ion; "
                f"got {incomplete}"
            )

    def _build_jac_sparsity(self) -> csr_matrix:
        pattern = np.zeros((self.layout.size, self.layout.size), dtype=bool)

        def zone_bounds(zone_id: str) -> tuple[int, int]:
            density_slice = self.layout.density_slices[zone_id]
            stop = density_slice.stop
            if self.layout.evolves_electron_energy:
                stop = self.layout.electron_energy_indices[zone_id] + 1
            if self.layout.evolves_heavy_energy:
                stop = self.layout.heavy_energy_indices[zone_id] + 1
            return density_slice.start, stop

        for zone_id in self.layout.zone_ids:
            start, stop = zone_bounds(zone_id)
            density_slice = self.layout.density_slices[zone_id]
            pattern[density_slice, density_slice] = (
                self.chemistry_jacobian_species_pattern
            )
            if self.layout.evolves_electron_energy:
                energy_index = self.layout.electron_energy_indices[zone_id]
                pattern[density_slice, energy_index] = np.any(
                    self.stoichiometry != 0.0, axis=0
                )
                pattern[energy_index, start:stop] = True
            if self.layout.evolves_heavy_energy:
                heavy_index = self.layout.heavy_energy_indices[zone_id]
                pattern[density_slice, heavy_index] = np.any(
                    self.stoichiometry != 0.0, axis=0
                )
                pattern[heavy_index, start:stop] = True
            if self._walls_by_zone.get(zone_id):
                pattern[start:stop, start:stop] = True
        if self.transport is not None:
            for zone_id in self.layout.zone_ids:
                start, stop = zone_bounds(zone_id)
                pattern[start:stop, start:stop] |= np.eye(stop - start, dtype=bool)
            for source_index, target_index in zip(
                self.transport.edge_from, self.transport.edge_to
            ):
                source_zone = self.layout.zone_ids[int(source_index)]
                target_zone = self.layout.zone_ids[int(target_index)]
                source_start, source_stop = zone_bounds(source_zone)
                target_start, target_stop = zone_bounds(target_zone)
                pattern[
                    target_start:target_stop,
                    source_start:source_stop,
                ] = True
        if self.surface_model is not None:
            for surface in self.surface_model.surfaces:
                zone_start, zone_stop = zone_bounds(surface.zone_id)
                pattern[zone_start:zone_stop, zone_start:zone_stop] = True
                coverage_indices = [
                    self.layout.surface_coverage_indices[key]
                    for key in self.layout.surface_coverage_indices
                    if key[0] == surface.surface_id
                ]
                if not coverage_indices:
                    continue
                pattern[zone_start:zone_stop, coverage_indices] = True
                pattern[coverage_indices, zone_start:zone_stop] = True
                pattern[np.ix_(coverage_indices, coverage_indices)] = True
        if (
            self.power_coordinator is not None
            and self.electron_closure.mode == "local_field"
        ):
            for zone_id in self.layout.zone_ids:
                start, stop = zone_bounds(zone_id)
                pattern[start:stop, start:stop] = True
        if self.extension_accumulator is not None:
            # Experimental accumulators are one-way consumers of compiled
            # drivers. Their RHS may depend on any plasma/surface state, while
            # the physical core never depends on an extension state.
            pattern[self.layout.extension_slice, :] = True
        return csr_matrix(pattern)

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

    def _electron_density(
        self,
        time_s: float,
        zone_id: str,
        net_heavy_charge_density_m3: float,
        segment: RecipeSegment,
    ) -> float:
        prescribed = segment.prescribed_electron_density_m3_by_zone.get(zone_id)
        if prescribed is None and self.electron_density_provider is not None:
            prescribed = self.electron_density_provider(float(time_s), zone_id)
        return resolved_electron_density(net_heavy_charge_density_m3, prescribed)

    def initial_state(self, initial: InitialState) -> np.ndarray:
        expected_zones = set(self.layout.zone_ids)
        supplied_zones = set(initial.densities_m3_by_zone)
        if supplied_zones != expected_zones:
            raise ModelConfigurationError(
                f"Initial density zones must be exactly {sorted(expected_zones)}, got {sorted(supplied_zones)}"
            )
        if self.layout.evolves_electron_energy:
            energy_zones = set(initial.mean_energy_eV_by_zone)
            if energy_zones != expected_zones:
                raise ModelConfigurationError(
                    f"electron_energy closure needs initial mean energy for zones {sorted(expected_zones)}"
                )
        elif initial.mean_energy_eV_by_zone:
            raise ModelConfigurationError(
                "local_field closure derives mean energy and rejects an initial energy state"
            )
        if self.layout.evolves_heavy_energy:
            gas_temperature_zones = set(initial.gas_temperature_K_by_zone)
            if gas_temperature_zones and gas_temperature_zones != expected_zones:
                raise ModelConfigurationError(
                    "Evolved gas energy initial temperatures must name every zone"
                )
        elif initial.gas_temperature_K_by_zone:
            raise ModelConfigurationError(
                "Fixed gas energy rejects an initial gas-energy state"
            )
        if self.surface_model is None and initial.surface_coverages:
            raise ModelConfigurationError(
                "Initial surface coverages require CompiledSurfaceModel"
            )

        state = np.zeros(self.layout.size, dtype=float)
        density_rows: list[np.ndarray] = []
        gas_temperatures: list[float] = []
        for zone_id in self.layout.zone_ids:
            values = initial.densities_m3_by_zone[zone_id]
            if set(values) != set(self.species_ids):
                raise ModelConfigurationError(
                    f"Initial densities for zone {zone_id!r} must explicitly name every species "
                    f"{list(self.species_ids)}"
                )
            density = np.array(
                [values[species_id] for species_id in self.species_ids], dtype=float
            )
            state[self.layout.density_slices[zone_id]] = density
            density_rows.append(density)
            net_charge = float(self.charges @ density)
            electron_density = self._electron_density(
                self.segments[0].start_s,
                zone_id,
                net_charge,
                self.segments[0],
            )
            if self.layout.evolves_electron_energy:
                mean_energy = float(initial.mean_energy_eV_by_zone[zone_id])
                if electron_density == 0.0 and mean_energy != 0.0:
                    raise ModelConfigurationError(
                        f"Zone {zone_id!r} cannot initialize nonzero electron energy at zero electron density"
                    )
                state[self.layout.electron_energy_indices[zone_id]] = (
                    electron_density * ELEMENTARY_CHARGE_C * mean_energy
                )
            if self.layout.evolves_heavy_energy:
                gas_temperatures.append(
                    float(
                        initial.gas_temperature_K_by_zone.get(
                            zone_id, self._zone_by_id[zone_id].gas_temperature_K
                        )
                    )
                )
        if self.layout.evolves_heavy_energy:
            heavy_energy_closure = _required_heavy_energy_closure(
                self.heavy_energy_closure
            )
            initial_heavy_energy = heavy_energy_closure.energy_J_m3(
                np.asarray(density_rows), np.asarray(gas_temperatures)
            )
            for zone_id, energy in zip(self.layout.zone_ids, initial_heavy_energy):
                state[self.layout.heavy_energy_indices[zone_id]] = energy
        if self.surface_model is not None:
            coverage = self.surface_model.initial_state()
            known_surfaces = {
                surface.surface_id for surface in self.surface_model.surfaces
            }
            unknown_surfaces = set(initial.surface_coverages) - known_surfaces
            if unknown_surfaces:
                raise ModelConfigurationError(
                    f"Initial coverages reference unknown surfaces {sorted(unknown_surfaces)}"
                )
            for surface_id, values in initial.surface_coverages.items():
                for species_id, value in values.items():
                    local_index = self.surface_model.layout.state_index.get(
                        (surface_id, species_id)
                    )
                    if local_index is None:
                        raise ModelConfigurationError(
                            f"Initial coverage {(surface_id, species_id)!r} is not an independent state"
                        )
                    coverage[local_index] = value
            for surface in self.surface_model.surfaces:
                free_species = self.surface_model.layout.free_species_by_surface[
                    surface.surface_id
                ]
                if (
                    self.surface_model.coverage(
                        coverage, surface.surface_id, free_species
                    )
                    < -1.0e-12
                ):
                    raise ModelConfigurationError(
                        f"Initial coverage on surface {surface.surface_id!r} exceeds site occupancy"
                    )
            state[self.layout.surface_coverage_slice] = coverage
        if self.extension_accumulator is not None:
            extension_state = np.asarray(
                self.extension_accumulator.initial_state(), dtype=float
            )
            expected = (
                self.layout.extension_slice.stop - self.layout.extension_slice.start
            )
            if extension_state.shape != (expected,) or not np.all(
                np.isfinite(extension_state)
            ):
                raise ModelConfigurationError(
                    "Extension initial state must match its compiled state block"
                )
            state[self.layout.extension_slice] = extension_state
        return state

    def evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool = True,
    ) -> ModelEvaluation:
        """Evaluate the full diagnostic view used by output and audit paths."""

        result = self._evaluate(
            time_s,
            state,
            segment,
            collect_ledger=collect_ledger,
            derivative_only=False,
        )
        assert isinstance(result, ModelEvaluation)
        return result

    def evaluate_derivative(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        domain_atol: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate only ``dy/dt`` without materializing diagnostic wrappers."""

        result = self._evaluate(
            time_s,
            state,
            segment,
            collect_ledger=False,
            derivative_only=True,
            domain_atol=domain_atol,
        )
        assert isinstance(result, np.ndarray)
        return result

    def _prepare_zone_evaluation(
        self,
        *,
        time_s: float,
        values: np.ndarray,
        domain_atol: np.ndarray,
        segment: RecipeSegment,
        zone_index: int,
        zone: Zone,
    ) -> _PreparedZoneState:
        density_slice = self.layout.density_slices[zone.zone_id]
        density = values[density_slice]
        density_atol = domain_atol[density_slice]
        materially_negative = density < (-10.0 * density_atol)
        if np.any(materially_negative):
            bad = {
                species_id: float(number_density)
                for species_id, number_density, failed in zip(
                    self.species_ids, density, materially_negative, strict=True
                )
                if failed
            }
            raise StateDomainError(
                f"Negative density below -10*domain_atol in zone {zone.zone_id!r}: {bad}"
            )
        reaction_density = np.where(density < 0.0, 0.0, density)
        raw_net_charge = float(self.charges @ density)
        charge_atol = float(np.abs(self.charges) @ density_atol)
        net_charge = (
            0.0 if -10.0 * charge_atol <= raw_net_charge < 0.0 else raw_net_charge
        )
        electron_density = self._electron_density(
            time_s, zone.zone_id, net_charge, segment
        )
        electron_energy, closure_energy, mean_energy = self._prepare_electron_energy(
            values=values,
            domain_atol=domain_atol,
            segment=segment,
            zone_index=zone_index,
            zone=zone,
            net_charge=net_charge,
            electron_density=electron_density,
        )
        heavy_energy = self._prepare_heavy_energy(
            values=values, domain_atol=domain_atol, zone=zone
        )
        return _PreparedZoneState(
            density=density,
            reaction_density=reaction_density,
            electron_energy=electron_energy,
            closure_electron_energy=closure_energy,
            heavy_energy=heavy_energy,
            net_charge=net_charge,
            electron_density=electron_density,
            charge_residual=raw_net_charge - electron_density,
            neutral_density=float(np.sum(reaction_density[self.charges == 0.0])),
            mean_energy_for_power=mean_energy,
        )

    def _prepare_electron_energy(
        self,
        *,
        values: np.ndarray,
        domain_atol: np.ndarray,
        segment: RecipeSegment,
        zone_index: int,
        zone: Zone,
        net_charge: float,
        electron_density: float,
    ) -> tuple[float | None, float | None, float | None]:
        if not self.layout.evolves_electron_energy:
            return None, None, None
        energy_index = self.layout.electron_energy_indices[zone.zone_id]
        energy = float(values[energy_index])
        energy_atol = float(domain_atol[energy_index])
        if energy < -10.0 * energy_atol:
            raise StateDomainError(
                f"Electron energy below -10*domain_atol in zone {zone.zone_id!r}"
            )
        closure_energy = max(energy, 0.0)
        if electron_density == 0.0 and closure_energy <= 10.0 * energy_atol:
            closure_energy = 0.0
        initial_electrons = self.electron_closure.evaluate(
            net_heavy_charge_density_m3=net_charge,
            energy_density_J_m3=closure_energy,
            reduced_field_Td=segment.reduced_field_Td_by_zone.get(zone.zone_id),
            electron_density_m3=electron_density,
        )
        del zone_index
        return energy, closure_energy, initial_electrons.mean_energy_eV

    def _prepare_heavy_energy(
        self, *, values: np.ndarray, domain_atol: np.ndarray, zone: Zone
    ) -> float | None:
        if not self.layout.evolves_heavy_energy:
            return None
        energy_index = self.layout.heavy_energy_indices[zone.zone_id]
        energy = float(values[energy_index])
        if energy < -10.0 * domain_atol[energy_index]:
            raise StateDomainError(
                f"Gas internal energy below -10*domain_atol in zone {zone.zone_id!r}"
            )
        return energy

    def _prepare_evaluation_state(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        domain_atol: np.ndarray | None,
    ) -> _PreparedEvaluationState:
        values = np.asarray(state, dtype=float)
        if values.shape != (self.layout.size,):
            raise StateDomainError(
                f"State has shape {values.shape}, expected {(self.layout.size,)}"
            )
        if not np.all(np.isfinite(values)):
            raise StateDomainError("State must contain only finite values")
        active_domain_atol = (
            np.asarray(self.domain_atol, dtype=float)
            if domain_atol is None
            else domain_atol
        )
        zone_count = len(self.zones)
        density_by_zone = np.empty((zone_count, len(self.species_ids)), dtype=float)
        reaction_density_by_zone = np.empty_like(density_by_zone)
        electron_energy_by_zone = self._optional_zone_array(
            self.layout.evolves_electron_energy, zone_count
        )
        closure_electron_energy_by_zone = self._optional_zone_array(
            self.layout.evolves_electron_energy, zone_count
        )
        heavy_energy_by_zone = self._optional_zone_array(
            self.layout.evolves_heavy_energy, zone_count
        )
        net_charge_by_zone: dict[str, float] = {}
        electron_density_by_zone: dict[str, float] = {}
        charge_residual_by_zone: dict[str, float] = {}
        neutral_density_by_zone: dict[str, float] = {}
        mean_energy_for_power: dict[str, float | None] = {}
        for zone_index, zone in enumerate(self.zones):
            prepared = self._prepare_zone_evaluation(
                time_s=float(time_s),
                values=values,
                domain_atol=active_domain_atol,
                segment=segment,
                zone_index=zone_index,
                zone=zone,
            )
            density_by_zone[zone_index] = prepared.density
            reaction_density_by_zone[zone_index] = prepared.reaction_density
            self._store_optional(
                electron_energy_by_zone, zone_index, prepared.electron_energy
            )
            self._store_optional(
                closure_electron_energy_by_zone,
                zone_index,
                prepared.closure_electron_energy,
            )
            self._store_optional(
                heavy_energy_by_zone, zone_index, prepared.heavy_energy
            )
            net_charge_by_zone[zone.zone_id] = prepared.net_charge
            electron_density_by_zone[zone.zone_id] = prepared.electron_density
            charge_residual_by_zone[zone.zone_id] = prepared.charge_residual
            neutral_density_by_zone[zone.zone_id] = prepared.neutral_density
            mean_energy_for_power[zone.zone_id] = prepared.mean_energy_for_power
        gas_temperature_by_zone = self._gas_temperatures(
            density_by_zone, heavy_energy_by_zone
        )
        return _PreparedEvaluationState(
            values=values,
            domain_atol=active_domain_atol,
            density_by_zone=density_by_zone,
            reaction_density_by_zone=reaction_density_by_zone,
            electron_energy_by_zone=electron_energy_by_zone,
            closure_electron_energy_by_zone=closure_electron_energy_by_zone,
            heavy_energy_by_zone=heavy_energy_by_zone,
            net_charge_by_zone=net_charge_by_zone,
            electron_density_by_zone=electron_density_by_zone,
            charge_residual_by_zone=charge_residual_by_zone,
            neutral_density_by_zone=neutral_density_by_zone,
            mean_energy_for_power=mean_energy_for_power,
            gas_temperature_by_zone=gas_temperature_by_zone,
        )

    @staticmethod
    def _optional_zone_array(enabled: bool, zone_count: int) -> np.ndarray | None:
        return np.empty(zone_count, dtype=float) if enabled else None

    @staticmethod
    def _store_optional(
        target: np.ndarray | None, index: int, value: float | None
    ) -> None:
        if target is not None and value is not None:
            target[index] = value

    def _gas_temperatures(
        self, density_by_zone: np.ndarray, heavy_energy_by_zone: np.ndarray | None
    ) -> np.ndarray:
        if heavy_energy_by_zone is None:
            return np.array(
                [zone.gas_temperature_K for zone in self.zones], dtype=float
            )
        if self.heavy_energy_closure is None:
            raise ModelConfigurationError("Evolved gas energy needs its closure")
        closure_energy = np.where(heavy_energy_by_zone < 0.0, 0.0, heavy_energy_by_zone)
        temperatures = self.heavy_energy_closure.temperature_K(
            density_by_zone, closure_energy
        )
        if np.any(temperatures <= 0.0):
            raise StateDomainError("Evolved gas temperature must remain positive")
        return temperatures

    def _evaluate_power(
        self,
        time_s: float,
        segment: RecipeSegment,
        prepared: _PreparedEvaluationState,
    ) -> _PowerEvaluation:
        if self.power_coordinator is None:
            coupling = None
            electron_power = segment.absorbed_power_W_by_zone
            gas_power: Mapping[str, float] = MappingProxyType({})
            reduced_field = dict(segment.reduced_field_Td_by_zone)
            kinetics = self._uncoordinated_kinetics(prepared, reduced_field)
        else:
            field_models: dict[str, Any] = {}
            field_model = self.electron_closure.mean_energy_from_field
            if callable(field_model):
                field_models = {zone.zone_id: field_model for zone in self.zones}
            coupling = self.power_coordinator.evaluate(
                time_s=float(time_s),
                commands=segment.port_commands,
                electron_density_m3_by_zone=prepared.electron_density_by_zone,
                neutral_density_m3_by_zone=prepared.neutral_density_by_zone,
                mean_energy_eV_by_zone=prepared.mean_energy_for_power,
                prescribed_electron_power_W_by_zone=segment.absorbed_power_W_by_zone,
                prescribed_reduced_field_Td_by_zone=segment.reduced_field_Td_by_zone,
                kinetics_by_zone=self.electron_kinetics_by_zone,
                mean_energy_from_field_by_zone=field_models,
            )
            electron_power = coupling.electron_power_W_by_zone
            gas_power = coupling.gas_power_W_by_zone
            reduced_field = dict(coupling.reduced_field_Td_by_zone)
            kinetics = dict(coupling.kinetics_by_zone)
        electron_states = self._evaluate_electron_states(
            prepared, reduced_field, kinetics
        )
        return _PowerEvaluation(
            coupling=coupling,
            kinetics_by_zone=kinetics,
            electron_power_W_by_zone=electron_power,
            gas_power_W_by_zone=gas_power,
            reduced_field_Td_by_zone=reduced_field,
            electron_states=electron_states,
        )

    def _uncoordinated_kinetics(
        self,
        prepared: _PreparedEvaluationState,
        reduced_field_by_zone: dict[str, float],
    ) -> dict[str, ElectronKineticsResult]:
        results: dict[str, ElectronKineticsResult] = {}
        for zone in self.zones:
            table = self.electron_kinetics_by_zone.get(zone.zone_id)
            if table is None:
                continue
            if table.lookup == "mean_energy":
                mean_energy = prepared.mean_energy_for_power[zone.zone_id]
                if mean_energy is None:
                    raise StateDomainError(
                        f"Zone {zone.zone_id!r} has no mean energy for table lookup"
                    )
                result = table.evaluate(mean_energy_eV=mean_energy)
                reduced_field_by_zone.setdefault(
                    zone.zone_id, float(result.effective_field_Td)
                )
            else:
                field_value = reduced_field_by_zone.get(zone.zone_id)
                if field_value is None:
                    raise StateDomainError(
                        f"Zone {zone.zone_id!r} has no E/N for local-field table lookup"
                    )
                result = table.evaluate(reduced_field_Td=field_value)
            results[zone.zone_id] = result
        return results

    def _evaluate_electron_states(
        self,
        prepared: _PreparedEvaluationState,
        reduced_field_by_zone: Mapping[str, float],
        kinetics_by_zone: Mapping[str, ElectronKineticsResult],
    ) -> dict[str, ElectronState]:
        states: dict[str, ElectronState] = {}
        for zone_index, zone in enumerate(self.zones):
            field_value = reduced_field_by_zone.get(zone.zone_id)
            kinetics = kinetics_by_zone.get(zone.zone_id)
            electron_density = prepared.electron_density_by_zone[zone.zone_id]
            if self.electron_closure.mode == "local_field" and kinetics is not None:
                states[zone.zone_id] = self._local_field_electron_state(
                    electron_density, field_value, kinetics
                )
                continue
            energy_density = (
                None
                if prepared.closure_electron_energy_by_zone is None
                else float(prepared.closure_electron_energy_by_zone[zone_index])
            )
            states[zone.zone_id] = self.electron_closure.evaluate(
                net_heavy_charge_density_m3=prepared.net_charge_by_zone[zone.zone_id],
                energy_density_J_m3=energy_density,
                reduced_field_Td=field_value,
                electron_density_m3=electron_density,
            )
        return states

    @staticmethod
    def _local_field_electron_state(
        electron_density: float,
        field_value: float | None,
        kinetics: ElectronKineticsResult,
    ) -> ElectronState:
        mean_energy = float(kinetics.mean_energy_eV)
        return ElectronState(
            density_m3=electron_density,
            mean_energy_eV=mean_energy,
            temperature_eV=float(kinetics.electron_temperature_eV),
            energy_density_J_m3=(electron_density * ELEMENTARY_CHARGE_C * mean_energy),
            reduced_field_Td=(
                float(field_value)
                if field_value is not None
                else float(kinetics.effective_field_Td)
            ),
        )

    def _evaluate_transport(
        self, segment: RecipeSegment, prepared: _PreparedEvaluationState
    ) -> _TransportEvaluation:
        zone_count = len(self.zones)
        density_rhs = np.zeros_like(prepared.density_by_zone)
        electron_rhs = self._optional_zero_zone_array(
            self.layout.evolves_electron_energy, zone_count
        )
        heavy_rhs = self._optional_zero_zone_array(
            self.layout.evolves_heavy_energy, zone_count
        )
        inlet_heavy_energy = np.zeros(zone_count, dtype=float)
        if self.transport is not None:
            forcing = self._transport_by_segment[segment.segment_id]
            density_rhs, electron_rhs, heavy_rhs = self.transport.evaluate(
                prepared.density_by_zone,
                forcing,
                electron_energy_J_m3=prepared.electron_energy_by_zone,
                heavy_energy_J_m3=prepared.heavy_energy_by_zone,
            )
            inlet_heavy_energy = forcing.inlet_heavy_energy_J_m3_s
        return _TransportEvaluation(
            density_rhs=density_rhs,
            electron_energy_rhs=electron_rhs,
            heavy_energy_rhs=heavy_rhs,
            inlet_heavy_energy=inlet_heavy_energy,
        )

    @staticmethod
    def _optional_zero_zone_array(enabled: bool, zone_count: int) -> np.ndarray | None:
        return np.zeros(zone_count, dtype=float) if enabled else None

    def _evaluate_reactions_and_walls(
        self,
        time_s: float,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
        *,
        collect_ledger: bool,
    ) -> _ReactionWallEvaluation:
        zone_count = len(self.zones)
        reaction_rates = np.zeros((zone_count, len(self.reaction_ids)))
        reaction_energy_loss = np.zeros(zone_count)
        gas_reaction_heating = np.zeros(zone_count)
        wall_species_rhs = np.zeros_like(prepared.density_by_zone)
        wall_energy_loss = np.zeros(zone_count)
        wall_records_by_zone: dict[str, list[WallFluxRecord]] = {}
        ion_flux_m2_s: dict[tuple[str, str], float] = {}
        ion_energy_eV: dict[str, float] = {}
        for zone_index, zone in enumerate(self.zones):
            result = self._evaluate_zone_reactions_and_walls(
                time_s=time_s,
                zone_index=zone_index,
                zone=zone,
                prepared=prepared,
                power=power,
                collect_ledger=collect_ledger,
            )
            reaction_rates[zone_index] = result.rates
            reaction_energy_loss[zone_index] = result.reaction_energy_loss
            gas_reaction_heating[zone_index] = result.gas_reaction_heating
            wall_species_rhs[zone_index] = result.wall_species_rhs
            wall_energy_loss[zone_index] = result.wall_energy_loss
            wall_records_by_zone[zone.zone_id] = result.wall_records
            for key, flux in result.ion_flux_m2_s.items():
                ion_flux_m2_s[key] = ion_flux_m2_s.get(key, 0.0) + flux
            ion_energy_eV.update(result.ion_energy_eV)
        return _ReactionWallEvaluation(
            reaction_rates_by_zone=reaction_rates,
            reaction_energy_loss=reaction_energy_loss,
            gas_reaction_heating=gas_reaction_heating,
            wall_species_rhs=wall_species_rhs,
            wall_energy_loss=wall_energy_loss,
            wall_records_by_zone=wall_records_by_zone,
            ion_flux_m2_s=ion_flux_m2_s,
            ion_energy_eV=ion_energy_eV,
        )

    def _evaluate_zone_reactions_and_walls(
        self,
        *,
        time_s: float,
        zone_index: int,
        zone: Zone,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
        collect_ledger: bool,
    ) -> _ZoneReactionWallEvaluation:
        reaction_density = prepared.reaction_density_by_zone[zone_index]
        electrons = power.electron_states[zone.zone_id]
        kinetics = power.kinetics_by_zone.get(zone.zone_id)
        context = RateContext(
            time_s=float(time_s),
            zone_id=zone.zone_id,
            gas_temperature_K=float(prepared.gas_temperature_by_zone[zone_index]),
            pressure_Pa=(
                float(np.sum(reaction_density))
                * 1.380649e-23
                * float(prepared.gas_temperature_by_zone[zone_index])
            ),
            electron_density_m3=electrons.density_m3,
            mean_energy_eV=electrons.mean_energy_eV,
            electron_temperature_eV=electrons.temperature_eV,
            reduced_field_Td=electrons.reduced_field_Td,
            electron_mobility_m2_V_s=(
                None if kinetics is None else float(kinetics.mobility_m2_V_s)
            ),
            rate_coefficients=(
                MappingProxyType({}) if kinetics is None else kinetics.rate_coefficients
            ),
            densities_m3=DensityView(
                self.species_ids, self._species_index, reaction_density
            ),
        )
        rates = self._reaction_rates(
            context,
            reaction_density,
            self._active_reactions_by_zone[zone.zone_id],
        )
        wall_rhs = np.zeros(len(self.species_ids), dtype=float)
        wall_loss = 0.0
        wall_records: list[WallFluxRecord] = []
        ion_flux: dict[tuple[str, str], float] = {}
        ion_energy: dict[str, float] = {}
        for compiled_wall in self._walls_by_zone[zone.zone_id]:
            wall = evaluate_compiled_wall_boundary(
                compiled=compiled_wall,
                volume_m3=zone.volume_m3,
                species_ids=self.species_ids,
                densities_m3=reaction_density,
                electrons=electrons,
                collect_records=collect_ledger,
            )
            wall_rhs += wall.species_derivative_m3_s
            wall_loss += wall.electron_energy_loss_J_m3_s
            wall_records.extend(wall.records)
            self._record_wall_ion_flux(compiled_wall, wall, ion_flux, ion_energy)
        return _ZoneReactionWallEvaluation(
            rates=rates,
            reaction_energy_loss=float(self.energy_loss_eV @ rates)
            * ELEMENTARY_CHARGE_C,
            gas_reaction_heating=float(self.gas_heating_eV @ rates)
            * ELEMENTARY_CHARGE_C,
            wall_species_rhs=wall_rhs,
            wall_energy_loss=wall_loss,
            wall_records=wall_records,
            ion_flux_m2_s=ion_flux,
            ion_energy_eV=ion_energy,
        )

    def _record_wall_ion_flux(
        self,
        compiled_wall: CompiledWallBoundary,
        wall: Any,
        ion_flux: dict[tuple[str, str], float],
        ion_energy: dict[str, float],
    ) -> None:
        surface_id = compiled_wall.boundary.surface_id
        if surface_id is None:
            return
        ion_energy[surface_id] = wall.sheath_energy_eV
        for species_index in compiled_wall.ion_indices:
            flux = float(wall.incident_flux_m2_s[int(species_index)])
            if flux != 0.0:
                key = (surface_id, self.species_ids[int(species_index)])
                ion_flux[key] = ion_flux.get(key, 0.0) + flux

    def _evaluate_surface_terms(
        self,
        segment: RecipeSegment,
        prepared: _PreparedEvaluationState,
        reactions: _ReactionWallEvaluation,
        *,
        collect_ledger: bool,
    ) -> _SurfaceTerms:
        rates_by_zone: dict[str, dict[str, float]] = (
            {zone.zone_id: {} for zone in self.zones} if collect_ledger else {}
        )
        if self.surface_model is None:
            return _SurfaceTerms(None, rates_by_zone)
        evaluation = self.surface_model.evaluate(
            prepared.values[self.layout.surface_coverage_slice],
            prepared.reaction_density_by_zone,
            prepared.gas_temperature_by_zone,
            ion_flux_m2_s=reactions.ion_flux_m2_s,
            ion_energy_eV=reactions.ion_energy_eV,
            surface_temperature_K=segment.surface_temperature_K_by_surface,
            domain_atol=prepared.domain_atol[self.layout.surface_coverage_slice],
            collect_rates=(collect_ledger or self.extension_accumulator is not None),
        )
        if collect_ledger:
            surface_zone = {
                surface.surface_id: surface.zone_id
                for surface in self.surface_model.surfaces
            }
            for rate_id, rate in evaluation.rates_m2_s.items():
                surface_id = rate_id.rsplit("@", 1)[-1]
                rates_by_zone[surface_zone[surface_id]][rate_id] = rate
        return _SurfaceTerms(evaluation, rates_by_zone)

    def _evaluate_energy_terms(
        self,
        segment: RecipeSegment,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
    ) -> _EnergyTerms:
        return _EnergyTerms(
            wall_heavy_exchange=self._wall_heavy_exchange(segment, prepared),
            elastic_heating=self._elastic_heating(prepared, power),
        )

    def _wall_heavy_exchange(
        self, segment: RecipeSegment, prepared: _PreparedEvaluationState
    ) -> np.ndarray:
        result = np.zeros(len(self.zones))
        if prepared.heavy_energy_by_zone is None:
            return result
        if self.heavy_energy_closure is None:
            raise ModelConfigurationError("Evolved gas energy needs its closure")
        wall_temperature = np.asarray(
            [
                segment.wall_temperature_K_by_zone.get(
                    zone.zone_id,
                    self.heavy_energy_closure.wall_temperature_K[zone_index],
                )
                for zone_index, zone in enumerate(self.zones)
            ],
            dtype=float,
        )
        equilibrium = self.heavy_energy_closure.energy_J_m3(
            prepared.density_by_zone, wall_temperature
        )
        return -self.heavy_energy_closure.wall_relaxation_s_inv * (
            prepared.heavy_energy_by_zone - equilibrium
        )

    def _elastic_heating(
        self, prepared: _PreparedEvaluationState, power: _PowerEvaluation
    ) -> np.ndarray:
        result = np.zeros(len(self.zones))
        if self.elastic_heating_evaluator is None:
            return result
        for zone_index, zone in enumerate(self.zones):
            value = float(
                self.elastic_heating_evaluator(
                    zone.zone_id,
                    power.electron_states[zone.zone_id],
                    prepared.reaction_density_by_zone[zone_index],
                    float(prepared.gas_temperature_by_zone[zone_index]),
                    power.kinetics_by_zone.get(zone.zone_id),
                )
            )
            if not math.isfinite(value):
                raise StateDomainError(
                    f"Elastic heating evaluator returned {value!r} in zone {zone.zone_id!r}"
                )
            result[zone_index] = value
        return result

    def _extension_rhs(
        self,
        prepared: _PreparedEvaluationState,
        reactions: _ReactionWallEvaluation,
        surface: _SurfaceTerms,
    ) -> np.ndarray | None:
        if self.extension_accumulator is None:
            return None
        ion_flux_by_surface: dict[str, float] = {}
        for (surface_id, _species_id), flux in reactions.ion_flux_m2_s.items():
            ion_flux_by_surface[surface_id] = ion_flux_by_surface.get(
                surface_id, 0.0
            ) + float(flux)
        try:
            extension_rhs = np.asarray(
                self.extension_accumulator.rhs(
                    prepared.values[self.layout.extension_slice],
                    drivers={"ion_flux_m2_s": ion_flux_by_surface},
                    surface_rates_m2_s=(
                        MappingProxyType({})
                        if surface.evaluation is None
                        else surface.evaluation.rates_m2_s
                    ),
                ),
                dtype=float,
            )
        except (FloatingPointError, KeyError, ValueError) as exc:
            raise StateDomainError(
                f"Experimental accumulator state/domain failure: {exc}"
            ) from exc
        expected = self.layout.extension_slice.stop - self.layout.extension_slice.start
        if extension_rhs.shape != (expected,) or not np.all(np.isfinite(extension_rhs)):
            raise StateDomainError(
                "Experimental accumulator returned an invalid RHS vector"
            )
        return extension_rhs

    def _assemble_evaluation(
        self,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
        transport: _TransportEvaluation,
        reactions: _ReactionWallEvaluation,
        surface: _SurfaceTerms,
        energy: _EnergyTerms,
        *,
        collect_ledger: bool,
    ) -> tuple[np.ndarray, dict[str, ZoneTermLedger]]:
        derivative = np.zeros_like(prepared.values)
        surface_gas_rhs = self._surface_gas_rhs(prepared, surface)
        surface_heating = self._surface_heating(surface)
        if surface.evaluation is not None:
            derivative[self.layout.surface_coverage_slice] = (
                surface.evaluation.coverage_derivative_s_inv
            )
        extension_rhs = self._extension_rhs(prepared, reactions, surface)
        if extension_rhs is not None:
            derivative[self.layout.extension_slice] = extension_rhs
        ledger_by_zone: dict[str, ZoneTermLedger] = {}
        for zone_index, zone in enumerate(self.zones):
            ledger = self._assemble_zone_derivative(
                derivative=derivative,
                zone_index=zone_index,
                zone=zone,
                power=power,
                transport=transport,
                reactions=reactions,
                surface=surface,
                energy=energy,
                surface_gas_rhs=surface_gas_rhs,
                surface_heating=surface_heating,
                collect_ledger=collect_ledger,
            )
            if ledger is not None:
                ledger_by_zone[zone.zone_id] = ledger
        return derivative, ledger_by_zone

    def _surface_gas_rhs(
        self, prepared: _PreparedEvaluationState, surface: _SurfaceTerms
    ) -> np.ndarray:
        if surface.evaluation is None:
            return np.zeros_like(prepared.density_by_zone)
        return surface.evaluation.gas_derivative_m3_s

    def _surface_heating(self, surface: _SurfaceTerms) -> np.ndarray:
        if surface.evaluation is None:
            return np.zeros(len(self.zones))
        return surface.evaluation.gas_heating_J_m3_s

    def _assemble_zone_derivative(
        self,
        *,
        derivative: np.ndarray,
        zone_index: int,
        zone: Zone,
        power: _PowerEvaluation,
        transport: _TransportEvaluation,
        reactions: _ReactionWallEvaluation,
        surface: _SurfaceTerms,
        energy: _EnergyTerms,
        surface_gas_rhs: np.ndarray,
        surface_heating: np.ndarray,
        collect_ledger: bool,
    ) -> ZoneTermLedger | None:
        density_slice = self.layout.density_slices[zone.zone_id]
        rates = reactions.reaction_rates_by_zone[zone_index]
        derivative[density_slice] = (
            self.stoichiometry.T @ rates
            + transport.density_rhs[zone_index]
            + reactions.wall_species_rhs[zone_index]
            + surface_gas_rhs[zone_index]
        )
        power_density = (
            float(power.electron_power_W_by_zone.get(zone.zone_id, 0.0))
            / zone.volume_m3
        )
        gas_power_density = (
            float(power.gas_power_W_by_zone.get(zone.zone_id, 0.0)) / zone.volume_m3
        )
        electron_transport = self._optional_value(
            transport.electron_energy_rhs, zone_index
        )
        heavy_transport = self._optional_value(transport.heavy_energy_rhs, zone_index)
        if self.layout.evolves_electron_energy:
            derivative[self.layout.electron_energy_indices[zone.zone_id]] = (
                power_density
                - reactions.reaction_energy_loss[zone_index]
                - reactions.wall_energy_loss[zone_index]
                - energy.elastic_heating[zone_index]
                + electron_transport
            )
        if self.layout.evolves_heavy_energy:
            derivative[self.layout.heavy_energy_indices[zone.zone_id]] = (
                gas_power_density
                + reactions.gas_reaction_heating[zone_index]
                + surface_heating[zone_index]
                + energy.wall_heavy_exchange[zone_index]
                + energy.elastic_heating[zone_index]
                + heavy_transport
            )
        if not collect_ledger:
            return None
        return self._zone_ledger(
            zone_index=zone_index,
            zone=zone,
            rates=rates,
            power_density=power_density,
            gas_power_density=gas_power_density,
            electron_transport=electron_transport,
            heavy_transport=heavy_transport,
            transport=transport,
            reactions=reactions,
            surface=surface,
            energy=energy,
            surface_heating=surface_heating,
        )

    @staticmethod
    def _optional_value(values: np.ndarray | None, index: int) -> float:
        return 0.0 if values is None else float(values[index])

    def _zone_ledger(
        self,
        *,
        zone_index: int,
        zone: Zone,
        rates: np.ndarray,
        power_density: float,
        gas_power_density: float,
        electron_transport: float,
        heavy_transport: float,
        transport: _TransportEvaluation,
        reactions: _ReactionWallEvaluation,
        surface: _SurfaceTerms,
        energy: _EnergyTerms,
        surface_heating: np.ndarray,
    ) -> ZoneTermLedger:
        return ZoneTermLedger(
            zone_id=zone.zone_id,
            reaction_rates_m3_s={
                reaction_id: float(rate)
                for reaction_id, rate in zip(self.reaction_ids, rates, strict=True)
            },
            absorbed_power_J_m3_s=power_density,
            reaction_energy_loss_J_m3_s=float(
                reactions.reaction_energy_loss[zone_index]
            ),
            wall_energy_loss_J_m3_s=float(reactions.wall_energy_loss[zone_index]),
            gas_power_J_m3_s=gas_power_density,
            gas_reaction_heating_J_m3_s=float(
                reactions.gas_reaction_heating[zone_index]
            ),
            surface_reaction_heating_J_m3_s=float(surface_heating[zone_index]),
            wall_heavy_energy_exchange_J_m3_s=float(
                energy.wall_heavy_exchange[zone_index]
            ),
            elastic_heating_J_m3_s=float(energy.elastic_heating[zone_index]),
            transport_species_source_m3_s={
                species_id: float(source)
                for species_id, source in zip(
                    self.species_ids, transport.density_rhs[zone_index], strict=True
                )
            },
            transport_electron_energy_J_m3_s=electron_transport,
            transport_heavy_energy_J_m3_s=heavy_transport,
            inlet_heavy_energy_J_m3_s=float(transport.inlet_heavy_energy[zone_index]),
            surface_rates_m2_s=surface.rates_by_zone[zone.zone_id],
            wall_fluxes=tuple(reactions.wall_records_by_zone[zone.zone_id]),
        )

    def _evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool,
        derivative_only: bool,
        domain_atol: np.ndarray | None = None,
    ) -> ModelEvaluation | np.ndarray:
        prepared_state = self._prepare_evaluation_state(
            time_s, state, segment, domain_atol
        )

        power = self._evaluate_power(time_s, segment, prepared_state)
        transport = self._evaluate_transport(segment, prepared_state)
        reactions = self._evaluate_reactions_and_walls(
            time_s, prepared_state, power, collect_ledger=collect_ledger
        )
        surface = self._evaluate_surface_terms(
            segment, prepared_state, reactions, collect_ledger=collect_ledger
        )
        energy = self._evaluate_energy_terms(segment, prepared_state, power)
        derivative, ledger_by_zone = self._assemble_evaluation(
            prepared_state,
            power,
            transport,
            reactions,
            surface,
            energy,
            collect_ledger=collect_ledger,
        )

        if derivative_only:
            derivative.setflags(write=False)
            return derivative
        return ModelEvaluation(
            derivative=derivative,
            electron_states=power.electron_states,
            kinetics_by_zone=power.kinetics_by_zone,
            gas_temperature_K_by_zone={
                zone.zone_id: float(prepared_state.gas_temperature_by_zone[index])
                for index, zone in enumerate(self.zones)
            },
            charge_residual_m3_by_zone=prepared_state.charge_residual_by_zone,
            ledger_by_zone=ledger_by_zone,
            power_coupling=power.coupling,
        )

    def _reaction_rates(
        self,
        context: RateContext,
        density: np.ndarray,
        active_reactions: np.ndarray,
    ) -> np.ndarray:
        rates = np.empty(len(self.reaction_ids), dtype=float)
        for reaction_index, evaluator in enumerate(self.rate_evaluators):
            if not active_reactions[reaction_index]:
                rates[reaction_index] = 0.0
                continue
            coefficient = float(evaluator(context))
            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise StateDomainError(
                    f"Rate evaluator for {self.reaction_ids[reaction_index]!r} returned {coefficient!r}"
                )
            rate = coefficient
            electron_order = self.electron_orders[reaction_index]
            if electron_order != 0.0:
                rate *= context.electron_density_m3**electron_order
            for species_index, order in enumerate(self.reactant_orders[reaction_index]):
                if order != 0.0:
                    rate *= density[species_index] ** order
            if not math.isfinite(rate) or rate < 0.0:
                raise StateDomainError(
                    f"Mass-action rate for {self.reaction_ids[reaction_index]!r} is invalid: {rate!r}"
                )
            rates[reaction_index] = rate
        return rates


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
