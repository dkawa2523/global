"""Compiled surface-coverage chemistry with an algebraic free-site balance."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global.chemistry._contracts import surface_reaction_shape_error
from plasma_global.chemistry.data import ChemistryData, RateModelData, ReactionData
from plasma_global.chemistry.surface_roles import (
    SurfaceSpeciesRoles,
    _surface_initial_state_error,
    surface_species_roles,
)
from plasma_global.errors import CaseValidationError
from plasma_global.models._surface_kernels import BOLTZMANN_J_K as BOLTZMANN_J_K
from plasma_global.models._surface_kernels import (
    E_CHARGE,
    _CoveragePower,
    _Kernel,
    _NumericRateKernel,
    compile_numeric_rate,
)
from plasma_global.models._surface_runtime import _SurfaceRuntime


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
            str(name): float(value) for name, value in self.initial_coverages.items()
        }
        if any(not math.isfinite(value) or value < 0.0 for value in coverages.values()):
            raise CaseValidationError(
                "initial surface coverages must be finite and nonnegative"
            )
        object.__setattr__(self, "initial_coverages", MappingProxyType(coverages))


def _validated_surface_state_species(
    surface: SurfaceGeometry,
    roles: SurfaceSpeciesRoles,
) -> tuple[str, tuple[str, ...]]:
    """Validate one surface and return its algebraic and evolved species IDs."""

    if len(roles.free_site_ids) != 1:
        raise CaseValidationError(
            f"surface {surface.surface_id!r} requires exactly one free-site species"
        )
    free_id = roles.free_site_ids[0]
    if error := _surface_initial_state_error(
        surface.surface_id, surface.initial_coverages, roles
    ):
        raise CaseValidationError(error)
    return free_id, roles.independent_ids


@dataclass(frozen=True, slots=True)
class SurfaceStateLayout:
    labels: tuple[str, ...]
    state_index: Mapping[tuple[str, str], int]
    free_species_by_surface: Mapping[str, str]
    occupancy_by_species: Mapping[str, float]


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
        self._runtime = self._build_runtime()
        object.__setattr__(self, "_compiled_immutable", True)

    def _build_runtime(self) -> _SurfaceRuntime:
        return _SurfaceRuntime(
            surface_ids=tuple(surface.surface_id for surface in self.surfaces),
            surface_temperatures_K=tuple(
                surface.temperature_K for surface in self.surfaces
            ),
            zone_count=len(self.zone_ids),
            gas_species_count=len(self.gas_species_ids),
            coverage_count=len(self.layout.labels),
            domain_atol=self.domain_atol,
            free_state_indices=self._free_state_indices,
            free_state_occupancies=self._free_state_occupancies,
            kernels=self._kernels,
        )

    def _build_layout(self) -> SurfaceStateLayout:
        labels: list[str] = []
        indexes: dict[tuple[str, str], int] = {}
        free: dict[str, str] = {}
        catalog = surface_species_roles(self.chemistry.species, None)
        for surface in self.surfaces:
            roles = surface_species_roles(
                self.chemistry.species,
                surface.surface_id,
            )
            free_id, independent_ids = _validated_surface_state_species(
                surface,
                roles,
            )
            free[surface.surface_id] = free_id
            for species_id in independent_ids:
                indexes[(surface.surface_id, species_id)] = len(labels)
                labels.append(f"coverage[{surface.surface_id},{species_id}]")
        return SurfaceStateLayout(
            labels=tuple(labels),
            state_index=MappingProxyType(indexes),
            free_species_by_surface=MappingProxyType(free),
            occupancy_by_species=catalog.occupancy_by_species,
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
                    (self._gas_index[species_id], float(order), species_id)
                )
            else:
                surface_reactants.append(
                    (
                        species_id,
                        self._coverage_power(
                            surface.surface_id, species_id, float(order)
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
            compile_numeric_rate(
                model,
                surface_id=surface.surface_id,
                site_density_m2=surface.site_density_m2,
                gas_reactants=gas_reactants,
                gas_masses_kg=self.gas_masses_kg,
            ),
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
            delta = float(
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

    def _continue_solver_coverages(self, state: np.ndarray) -> None:
        self._runtime.continue_coverages(state)

    def _validate_solver_coverages(
        self, state: np.ndarray, domain_atol: np.ndarray
    ) -> None:
        self._runtime.validate_coverages(state, domain_atol)

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
        result = self._runtime.evaluate(
            coverage_state,
            gas_densities_m3,
            gas_temperature_K,
            ion_flux_m2_s=ion_flux_m2_s,
            ion_energy_eV=ion_energy_eV,
            surface_temperature_K=surface_temperature_K,
            domain_atol=domain_atol,
            collect_rates=collect_rates,
        )
        return SurfaceEvaluation(
            result.gas_derivative_m3_s,
            result.coverage_derivative_s_inv,
            result.gas_heating_J_m3_s,
            result.rates_m2_s,
        )


__all__ = [
    "CompiledSurfaceModel",
    "SurfaceEvaluation",
    "SurfaceGeometry",
    "SurfaceStateLayout",
]
