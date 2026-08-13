"""Compiled surface-coverage chemistry with an algebraic free-site balance."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, SupportsFloat

import numpy as np

from plasma_global.chemistry._contracts import surface_reaction_shape_error
from plasma_global.chemistry.data import ChemistryData, RateModelData, ReactionData
from plasma_global.errors import CaseValidationError, ModelDomainError

BOLTZMANN_J_K = 1.380649e-23
E_CHARGE = 1.602176634e-19
_EMPTY_RATES: Mapping[str, float] = MappingProxyType({})


def _normalize_id(value: object) -> str:
    return str(value)


def _python_float(value: SupportsFloat) -> float:
    return float(value)


@dataclass(frozen=True, slots=True)
class SurfaceGeometry:
    surface_id: str
    zone_id: str
    area_m2: float
    site_density_m2: float
    temperature_K: float
    initial_coverages: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.surface_id or not self.zone_id:
            raise CaseValidationError("surface and zone IDs must be nonempty")
        values = self.area_m2, self.site_density_m2, self.temperature_K
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise CaseValidationError(
                "surface area, site density, and temperature must be positive"
            )
        coverages = {
            _normalize_id(name): _python_float(value)
            for name, value in self.initial_coverages.items()
        }
        if any(not math.isfinite(value) or value < 0.0 for value in coverages.values()):
            raise CaseValidationError(
                "initial surface coverages must be finite and nonnegative"
            )
        object.__setattr__(self, "initial_coverages", MappingProxyType(coverages))


@dataclass(frozen=True, slots=True)
class SurfaceStateLayout:
    labels: tuple[str, ...]
    state_index: Mapping[tuple[str, str], int]
    free_species_by_surface: Mapping[str, str]
    occupancy_by_species: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _CoveragePower:
    """One already-resolved coverage raised to a fixed power.

    ``state_index=None`` denotes the algebraic free-site coverage of the
    owning surface.  All species/configuration resolution is therefore done
    before the first RHS call.
    """

    state_index: int | None
    exponent: float

    def evaluate(self, state: np.ndarray, free_coverage: float) -> float:
        value = (
            free_coverage
            if self.state_index is None
            else float(state[self.state_index])
        )
        value = max(value, 0.0)
        if value == 0.0 and self.exponent > 0.0:
            return 0.0
        return value**self.exponent


class _NumericRateKernel(Protocol):
    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class _StickingRate:
    gas_index: int
    thermal_prefactor_m_s_K_half: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del ion_fluxes, ion_energy_eV, surface_temperature_K
        return (
            self.thermal_prefactor_m_s_K_half
            * math.sqrt(gas_temperature_K)
            * float(gas[self.gas_index])
        )


@dataclass(frozen=True, slots=True)
class _IonAssistedRate:
    flux_key: tuple[str, str]
    yield_factor: float
    threshold_eV: float
    inverse_energy_span_eV_inv: float
    exponent: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del gas, gas_temperature_K, surface_temperature_K
        if ion_energy_eV <= self.threshold_eV:
            return 0.0
        energy_factor = (
            (ion_energy_eV - self.threshold_eV) * self.inverse_energy_span_eV_inv
        ) ** self.exponent
        return float(
            ion_fluxes.get(self.flux_key, 0.0) * self.yield_factor * energy_factor
        )


@dataclass(frozen=True, slots=True)
class _ThermalRate:
    prefactor_m2_s: float
    activation_temperature_K: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del gas, gas_temperature_K, ion_fluxes, ion_energy_eV
        return self.prefactor_m2_s * math.exp(
            -self.activation_temperature_K / surface_temperature_K
        )


@dataclass(frozen=True, slots=True)
class _Kernel:
    reaction: ReactionData
    surface_index: int
    zone_index: int
    rate_id: str
    rate_kernel: _NumericRateKernel
    coverage_factor: _CoveragePower | None
    surface_mass_action: tuple[_CoveragePower, ...]
    gas_delta: tuple[tuple[int, float], ...]
    coverage_delta: tuple[tuple[int, float], ...]
    area_over_volume_m_inv: float
    inverse_site_density_m2: float
    gas_heating_J_m3_per_event_m2: float


def _coverage_multiplier(
    kernel: _Kernel, state: np.ndarray, free_coverage: float
) -> float:
    mass_action = 1.0
    for coverage_power in kernel.surface_mass_action:
        mass_action *= coverage_power.evaluate(state, free_coverage)
        if mass_action == 0.0:
            break
    coverage_factor = (
        1.0
        if kernel.coverage_factor is None
        else kernel.coverage_factor.evaluate(state, free_coverage)
    )
    return mass_action * coverage_factor


@dataclass(frozen=True, slots=True)
class _RateContext:
    gas_densities_m3: np.ndarray
    gas_temperature_K: np.ndarray
    ion_flux_m2_s: Mapping[tuple[str, str], float]
    ion_energy_eV_by_surface: np.ndarray
    temperature_K_by_surface: np.ndarray


@dataclass(frozen=True, slots=True)
class SurfaceEvaluation:
    gas_derivative_m3_s: np.ndarray
    coverage_derivative_s_inv: np.ndarray
    gas_heating_J_m3_s: np.ndarray
    rates_m2_s: Mapping[str, float]


@dataclass
class CompiledSurfaceModel:
    """Surface chemistry compiled for concrete reactor surfaces."""

    chemistry: ChemistryData
    gas_species_ids: tuple[str, ...]
    gas_masses_kg: np.ndarray
    zone_ids: tuple[str, ...]
    zone_volumes_m3: np.ndarray
    surfaces: tuple[SurfaceGeometry, ...]
    domain_atol: float = 1.0e-12
    layout: SurfaceStateLayout = field(init=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_compiled_immutable", False):
            raise AttributeError("CompiledSurfaceModel is immutable after compilation")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.gas_species_ids = tuple(self.gas_species_ids)
        self.zone_ids = tuple(self.zone_ids)
        self.surfaces = tuple(self.surfaces)
        self.gas_masses_kg = np.array(self.gas_masses_kg, dtype=float, copy=True)
        self.zone_volumes_m3 = np.array(self.zone_volumes_m3, dtype=float, copy=True)
        if self.gas_masses_kg.shape != (len(self.gas_species_ids),) or np.any(
            self.gas_masses_kg <= 0.0
        ):
            raise CaseValidationError("surface model gas masses are invalid")
        if self.zone_volumes_m3.shape != (len(self.zone_ids),) or np.any(
            self.zone_volumes_m3 <= 0.0
        ):
            raise CaseValidationError("surface model zone volumes are invalid")
        if self.domain_atol <= 0.0:
            raise CaseValidationError("surface domain_atol must be positive")
        self.gas_masses_kg.setflags(write=False)
        self.zone_volumes_m3.setflags(write=False)
        self._species = MappingProxyType(
            {item.id: item for item in self.chemistry.species}
        )
        self._gas_index = MappingProxyType(
            {item: index for index, item in enumerate(self.gas_species_ids)}
        )
        self._zone_index = MappingProxyType(
            {item: index for index, item in enumerate(self.zone_ids)}
        )
        self._surface_index = MappingProxyType(
            {item.surface_id: index for index, item in enumerate(self.surfaces)}
        )
        self.layout = self._build_layout()
        self._free_state_indices, self._free_state_occupancies = (
            self._compile_free_site_balances()
        )
        self._kernels = self._compile_kernels()
        object.__setattr__(self, "_compiled_immutable", True)

    def _build_layout(self) -> SurfaceStateLayout:
        labels: list[str] = []
        indexes: dict[tuple[str, str], int] = {}
        free: dict[str, str] = {}
        occupancy: dict[str, float] = {}
        surface_species = [
            item for item in self.chemistry.species if item.phase == "surface"
        ]
        for species in surface_species:
            occupancy[species.id] = _python_float(species.elements.get("site", 1.0))
        for surface in self.surfaces:
            applicable = [
                item
                for item in surface_species
                if not item.surfaces or surface.surface_id in item.surfaces
            ]
            free_candidates = [item for item in applicable if "site" in item.state_tags]
            if len(free_candidates) != 1:
                raise CaseValidationError(
                    f"surface {surface.surface_id!r} requires exactly one "
                    "free-site species"
                )
            free_id = free_candidates[0].id
            if free_id in surface.initial_coverages:
                raise CaseValidationError(
                    f"surface {surface.surface_id!r} must not initialize algebraic "
                    f"free site {free_id!r}"
                )
            free[surface.surface_id] = free_id
            for species in applicable:
                if species.id == free_id or "film_fragment" in species.state_tags:
                    continue
                indexes[(surface.surface_id, species.id)] = len(labels)
                labels.append(f"coverage[{surface.surface_id},{species.id}]")
            unknown = set(surface.initial_coverages) - {
                species_id for sid, species_id in indexes if sid == surface.surface_id
            }
            if unknown:
                raise CaseValidationError(
                    f"surface {surface.surface_id!r} initializes "
                    f"unknown/nonindependent coverages {sorted(unknown)}"
                )
            occupied = sum(
                occupancy[species_id] * surface.initial_coverages.get(species_id, 0.0)
                for sid, species_id in indexes
                if sid == surface.surface_id
            )
            if occupied > 1.0 + 1.0e-12:
                raise CaseValidationError(
                    f"surface {surface.surface_id!r} initial site occupancy exceeds one"
                )
        return SurfaceStateLayout(
            labels=tuple(labels),
            state_index=MappingProxyType(indexes),
            free_species_by_surface=MappingProxyType(free),
            occupancy_by_species=MappingProxyType(occupancy),
        )

    def _compile_free_site_balances(
        self,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        state_indices: list[np.ndarray] = []
        occupancies: list[np.ndarray] = []
        for surface in self.surfaces:
            entries = tuple(
                (index, self.layout.occupancy_by_species[species_id])
                for (surface_id, species_id), index in self.layout.state_index.items()
                if surface_id == surface.surface_id
            )
            indices = np.asarray([entry[0] for entry in entries], dtype=np.intp)
            weights = np.asarray([entry[1] for entry in entries], dtype=float)
            indices.setflags(write=False)
            weights.setflags(write=False)
            state_indices.append(indices)
            occupancies.append(weights)
        return tuple(state_indices), tuple(occupancies)

    def _compile_kernels(self) -> tuple[_Kernel, ...]:
        kernels: list[_Kernel] = []
        for reaction in self.chemistry.surface_reactions:
            model = self._validated_rate_model(reaction)
            kernels.extend(
                self._compile_kernel(reaction, model, surface_index, surface)
                for surface_index, surface in self._reaction_surfaces(reaction)
            )
        return tuple(kernels)

    def _validated_rate_model(self, reaction: ReactionData) -> RateModelData:
        """Resolve and validate the rate model before selecting any surfaces."""
        rate_model_id = reaction.rate_model
        if rate_model_id is None or rate_model_id not in self.chemistry.rate_models:
            raise CaseValidationError(
                f"surface reaction {reaction.id} has unknown rate model"
            )
        model = self.chemistry.rate_models[rate_model_id]
        shape_error = surface_reaction_shape_error(reaction, model, self._species)
        if shape_error is not None:
            raise CaseValidationError(shape_error)
        return model

    def _reaction_surfaces(
        self, reaction: ReactionData
    ) -> Iterator[tuple[int, SurfaceGeometry]]:
        target_ids = reaction.surfaces or tuple(
            surface.surface_id for surface in self.surfaces
        )
        for surface_id in target_ids:
            if surface_id not in self._surface_index:
                raise CaseValidationError(
                    f"surface reaction {reaction.id} selects unknown surface "
                    f"{surface_id!r}"
                )
            surface_index = self._surface_index[surface_id]
            surface = self.surfaces[surface_index]
            if reaction.zones and surface.zone_id not in reaction.zones:
                continue
            yield surface_index, surface

    def _compile_rate_terms(
        self,
        reaction: ReactionData,
        model: RateModelData,
        surface: SurfaceGeometry,
    ) -> tuple[
        _NumericRateKernel,
        _CoveragePower | None,
        tuple[_CoveragePower, ...],
    ]:
        gas_reactants: list[tuple[int, float, str]] = []
        surface_reactants: list[tuple[str, _CoveragePower]] = []
        for species_id, order in reaction.reactants.items():
            species = self._species[species_id]
            if species.phase == "gas":
                gas_reactants.append(
                    (self._gas_index[species_id], _python_float(order), species_id)
                )
            else:
                surface_reactants.append(
                    (
                        species_id,
                        self._coverage_power(
                            surface.surface_id, species_id, _python_float(order)
                        ),
                    )
                )
        coverage_factor, skip_species = self._compile_coverage_factor(
            model, surface.surface_id
        )
        mass_action = tuple(
            coverage_power
            for species_id, coverage_power in surface_reactants
            if species_id != skip_species
        )
        return (
            self._compile_numeric_rate(model, surface, gas_reactants),
            coverage_factor,
            mass_action,
        )

    def _compile_stoichiometry(
        self,
        reaction: ReactionData,
        model: RateModelData,
        surface_id: str,
    ) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[int, float], ...]]:
        gas_delta: list[tuple[int, float]] = []
        coverage_delta: list[tuple[int, float]] = []
        for species_id in set(reaction.reactants) | set(reaction.products):
            delta = _python_float(
                reaction.products.get(species_id, 0.0)
                - reaction.reactants.get(species_id, 0.0)
            )
            if delta == 0.0:
                continue
            species = self._species[species_id]
            if species.phase == "gas":
                # The wall-transport ledger already removes an incident ion.
                # An ion-assisted event consumes that shared flux, so the
                # surface kernel must not subtract it a second time.
                if model.kind == "ion_assisted" and species.charge > 0 and delta < 0.0:
                    continue
                gas_delta.append((self._gas_index[species_id], delta))
            else:
                index = self.layout.state_index.get((surface_id, species_id))
                if index is not None:
                    coverage_delta.append((index, delta))
        return tuple(gas_delta), tuple(coverage_delta)

    def _compile_kernel(
        self,
        reaction: ReactionData,
        model: RateModelData,
        surface_index: int,
        surface: SurfaceGeometry,
    ) -> _Kernel:
        numeric_rate, coverage_factor, mass_action = self._compile_rate_terms(
            reaction, model, surface
        )
        gas_delta, coverage_delta = self._compile_stoichiometry(
            reaction, model, surface.surface_id
        )
        zone_index = self._zone_index[surface.zone_id]
        area_over_volume = surface.area_m2 / self.zone_volumes_m3[zone_index]
        return _Kernel(
            reaction=reaction,
            surface_index=surface_index,
            zone_index=zone_index,
            rate_id=f"{reaction.id}@{surface.surface_id}",
            rate_kernel=numeric_rate,
            coverage_factor=coverage_factor,
            surface_mass_action=mass_action,
            gas_delta=gas_delta,
            coverage_delta=coverage_delta,
            area_over_volume_m_inv=area_over_volume,
            inverse_site_density_m2=1.0 / surface.site_density_m2,
            gas_heating_J_m3_per_event_m2=(
                area_over_volume * reaction.gas_heating_eV * E_CHARGE
            ),
        )

    def _coverage_power(
        self, surface_id: str, species_id: str, exponent: float
    ) -> _CoveragePower:
        state_index = self.layout.state_index.get((surface_id, species_id))
        if (
            state_index is None
            and species_id != self.layout.free_species_by_surface[surface_id]
        ):
            raise CaseValidationError(
                f"surface {surface_id!r} uses incompatible coverage species "
                f"{species_id!r}"
            )
        return _CoveragePower(state_index, exponent)

    def _compile_coverage_factor(
        self, model: RateModelData, surface_id: str
    ) -> tuple[_CoveragePower | None, str | None]:
        config = model.parameters.get("coverage")
        if config is None:
            return None, None
        if not isinstance(config, Mapping):
            raise CaseValidationError(
                f"surface rate model {model.id!r} coverage must be a mapping"
            )
        kind = config.get("kind")
        if kind == "constant":
            return None, None
        if kind == "site_blocking":
            species_key = "site_species"
        elif kind == "species_power":
            species_key = "species"
        else:
            raise CaseValidationError(
                f"surface rate model {model.id!r} has unsupported coverage kind "
                f"{kind!r}"
            )
        try:
            species_id = str(config[species_key])
            exponent = float(config["exponent"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaseValidationError(
                f"surface rate model {model.id!r} has invalid coverage parameters"
            ) from exc
        if not species_id or not math.isfinite(exponent) or exponent < 0.0:
            raise CaseValidationError(
                f"surface rate model {model.id!r} has invalid coverage parameters"
            )
        return self._coverage_power(surface_id, species_id, exponent), species_id

    @staticmethod
    def _numeric_parameter(model: RateModelData, name: str) -> float:
        try:
            value = float(model.parameters[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaseValidationError(
                f"surface rate model {model.id!r} requires numeric parameter {name!r}"
            ) from exc
        if not math.isfinite(value):
            raise CaseValidationError(
                f"surface rate model {model.id!r} parameter {name!r} must be finite"
            )
        return value

    def _compile_numeric_rate(
        self,
        model: RateModelData,
        surface: SurfaceGeometry,
        gas_reactants: list[tuple[int, float, str]],
    ) -> _NumericRateKernel:
        if model.kind == "sticking":
            gas_index, _order, _species_id = gas_reactants[0]
            value = self._numeric_parameter(model, "value")
            return _StickingRate(
                gas_index=gas_index,
                thermal_prefactor_m_s_K_half=(
                    0.25
                    * value
                    * math.sqrt(
                        8.0 * BOLTZMANN_J_K / (math.pi * self.gas_masses_kg[gas_index])
                    )
                ),
            )
        if model.kind == "ion_assisted":
            _gas_index, _order, ion_species_id = gas_reactants[0]
            threshold = self._numeric_parameter(model, "threshold_eV")
            reference = self._numeric_parameter(model, "reference_energy_eV")
            if reference <= threshold:
                raise CaseValidationError(
                    f"surface rate model {model.id!r} reference_energy_eV must "
                    "exceed threshold_eV"
                )
            return _IonAssistedRate(
                flux_key=(surface.surface_id, ion_species_id),
                yield_factor=self._numeric_parameter(model, "yield"),
                threshold_eV=threshold,
                inverse_energy_span_eV_inv=1.0 / (reference - threshold),
                exponent=self._numeric_parameter(model, "exponent"),
            )
        if model.kind == "desorption":
            amplitude = self._numeric_parameter(model, "frequency_s_inv")
        elif model.kind == "langmuir_hinshelwood":
            amplitude = self._numeric_parameter(model, "A_m2_s_inv")
        else:
            raise CaseValidationError(f"unsupported surface rate model {model.kind!r}")
        activation_eV = self._numeric_parameter(model, "activation_eV")
        return _ThermalRate(
            prefactor_m2_s=amplitude * surface.site_density_m2,
            activation_temperature_K=activation_eV * E_CHARGE / BOLTZMANN_J_K,
        )

    def initial_state(self) -> np.ndarray:
        values = np.zeros(len(self.layout.labels))
        for surface in self.surfaces:
            for species_id, coverage in surface.initial_coverages.items():
                values[self.layout.state_index[(surface.surface_id, species_id)]] = (
                    coverage
                )
        return values

    def coverage(self, state: np.ndarray, surface_id: str, species_id: str) -> float:
        if species_id == self.layout.free_species_by_surface[surface_id]:
            surface_index = self._surface_index[surface_id]
            indices = self._free_state_indices[surface_index]
            return 1.0 - float(
                self._free_state_occupancies[surface_index] @ state[indices]
            )
        index = self.layout.state_index.get((surface_id, species_id))
        if index is None:
            raise KeyError((surface_id, species_id))
        return float(state[index])

    def _free_coverages(self, state: np.ndarray) -> np.ndarray:
        values = np.empty(len(self.surfaces), dtype=float)
        for surface_index, (indices, occupancies) in enumerate(
            zip(self._free_state_indices, self._free_state_occupancies, strict=True)
        ):
            values[surface_index] = 1.0 - float(occupancies @ state[indices])
        return values

    def _validated_evaluation_arrays(
        self,
        coverage_state: np.ndarray,
        gas_densities_m3: np.ndarray,
        gas_temperature_K: np.ndarray,
        domain_atol: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coverage = np.asarray(coverage_state, dtype=float)
        gas = np.asarray(gas_densities_m3, dtype=float)
        temperatures = np.asarray(gas_temperature_K, dtype=float)
        if coverage.shape != (len(self.layout.labels),):
            raise ValueError("surface coverage state has the wrong shape")
        if gas.shape != (
            len(self.zone_ids),
            len(self.gas_species_ids),
        ) or temperatures.shape != (len(self.zone_ids),):
            raise ValueError("surface gas arrays have the wrong shape")
        tolerance = (
            np.full(coverage.shape, self.domain_atol, dtype=float)
            if domain_atol is None
            else np.asarray(domain_atol, dtype=float)
        )
        if tolerance.shape != coverage.shape or np.any(tolerance < 0.0):
            raise ValueError("surface domain_atol has the wrong shape")
        if np.any(coverage < -10.0 * tolerance):
            raise ModelDomainError("surface coverage left the nonnegative domain")
        return coverage, gas, temperatures, tolerance

    def _reactive_coverages(
        self, coverage_state: np.ndarray, tolerance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Round-off negatives are zero only for surface mass action. The ODE
        # state and returned RHS are never projected or renormalized.
        reactive_state = np.where(coverage_state < 0.0, 0.0, coverage_state)
        free_site_tolerance = 10.0 * float(np.max(tolerance, initial=self.domain_atol))
        free_coverages = self._free_coverages(reactive_state)
        for surface_index, surface in enumerate(self.surfaces):
            if free_coverages[surface_index] < -free_site_tolerance:
                raise ModelDomainError(
                    f"surface {surface.surface_id!r} has negative algebraic "
                    "free-site coverage"
                )
        return reactive_state, free_coverages

    def _rate_context(
        self,
        gas: np.ndarray,
        gas_temperatures: np.ndarray,
        ion_flux_m2_s: Mapping[tuple[str, str], float] | None,
        ion_energy_eV: Mapping[str, float] | None,
        surface_temperature_K: Mapping[str, float] | None,
    ) -> _RateContext:
        fluxes = ion_flux_m2_s or {}
        energies = ion_energy_eV or {}
        surface_temperatures = surface_temperature_K or {}
        ion_energies = np.asarray(
            [energies.get(surface.surface_id, 0.0) for surface in self.surfaces],
            dtype=float,
        )
        resolved_surface_temperatures = np.asarray(
            [
                surface_temperatures.get(surface.surface_id, surface.temperature_K)
                for surface in self.surfaces
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(resolved_surface_temperatures)) or np.any(
            resolved_surface_temperatures <= 0.0
        ):
            raise ModelDomainError("surface temperature must be finite and positive")
        return _RateContext(
            gas_densities_m3=gas,
            gas_temperature_K=gas_temperatures,
            ion_flux_m2_s=fluxes,
            ion_energy_eV_by_surface=ion_energies,
            temperature_K_by_surface=resolved_surface_temperatures,
        )

    @staticmethod
    def _evaluate_kernel(
        kernel: _Kernel,
        reactive_state: np.ndarray,
        free_coverages: np.ndarray,
        context: _RateContext,
    ) -> float:
        surface_index = kernel.surface_index
        coverage_multiplier = _coverage_multiplier(
            kernel, reactive_state, float(free_coverages[surface_index])
        )
        rate = kernel.rate_kernel.evaluate(
            context.gas_densities_m3[kernel.zone_index],
            float(context.gas_temperature_K[kernel.zone_index]),
            context.ion_flux_m2_s,
            float(context.ion_energy_eV_by_surface[surface_index]),
            float(context.temperature_K_by_surface[surface_index]),
        )
        rate *= coverage_multiplier
        if not math.isfinite(rate) or rate < 0.0:
            raise ModelDomainError(
                f"surface reaction {kernel.reaction.id!r} returned invalid rate "
                f"{rate!r}"
            )
        return rate

    @staticmethod
    def _accumulate_kernel(
        kernel: _Kernel,
        rate: float,
        gas_rhs: np.ndarray,
        coverage_rhs: np.ndarray,
        gas_heating: np.ndarray,
    ) -> None:
        gas_heating[kernel.zone_index] += kernel.gas_heating_J_m3_per_event_m2 * rate
        for index, delta in kernel.gas_delta:
            gas_rhs[kernel.zone_index, index] += (
                kernel.area_over_volume_m_inv * delta * rate
            )
        for index, delta in kernel.coverage_delta:
            coverage_rhs[index] += delta * rate * kernel.inverse_site_density_m2

    def evaluate(
        self,
        coverage_state: np.ndarray,
        gas_densities_m3: np.ndarray,
        gas_temperature_K: np.ndarray,
        *,
        ion_flux_m2_s: Mapping[tuple[str, str], float] | None = None,
        ion_energy_eV: Mapping[str, float] | None = None,
        surface_temperature_K: Mapping[str, float] | None = None,
        domain_atol: np.ndarray | None = None,
        collect_rates: bool = True,
    ) -> SurfaceEvaluation:
        coverage, gas, temperatures, tolerance = self._validated_evaluation_arrays(
            coverage_state,
            gas_densities_m3,
            gas_temperature_K,
            domain_atol,
        )
        reactive_state, free_coverages = self._reactive_coverages(coverage, tolerance)
        context = self._rate_context(
            gas,
            temperatures,
            ion_flux_m2_s,
            ion_energy_eV,
            surface_temperature_K,
        )
        gas_rhs = np.zeros_like(gas)
        coverage_rhs = np.zeros_like(coverage)
        gas_heating = np.zeros(len(self.zone_ids), dtype=float)
        rates: dict[str, float] | None = {} if collect_rates else None
        for kernel in self._kernels:
            rate = self._evaluate_kernel(
                kernel, reactive_state, free_coverages, context
            )
            if rates is not None:
                rates[kernel.rate_id] = rate
            self._accumulate_kernel(kernel, rate, gas_rhs, coverage_rhs, gas_heating)
        return SurfaceEvaluation(
            gas_rhs,
            coverage_rhs,
            gas_heating,
            _EMPTY_RATES if rates is None else MappingProxyType(rates),
        )


__all__ = [
    "CompiledSurfaceModel",
    "SurfaceEvaluation",
    "SurfaceGeometry",
    "SurfaceStateLayout",
]
