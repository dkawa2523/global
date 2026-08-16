"""Compile the exploratory two-term closure into immutable local-field tables.

The approximation is deliberately confined to case compilation.  The ODE core
only receives :class:`~plasma_global.models.kinetics.TabulatedElectronKinetics`
objects and therefore never solves an EEDF while evaluating the RHS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from plasma_global.chemistry._contracts import normalized_cross_section_curve
from plasma_global.chemistry.data import ChemistryData, CrossSectionData, SpeciesData
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.eedf import ApproximateTwoTermEEDF, ElectronCollision
from plasma_global.models.kinetics import TabulatedElectronKinetics

_MOMENTUM_KINDS = frozenset(
    {
        "elastic",
        "elastic_momentum",
        "momentum",
        "momentum_transfer",
        "mt",
    }
)


@dataclass(frozen=True, slots=True)
class PreparedApproximateTwoTermKinetics:
    """Zone bindings and auditable assumptions produced during compilation."""

    by_zone: Mapping[str, TabulatedElectronKinetics]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_zone", MappingProxyType(dict(self.by_zone)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _collision_kind(
    cross_section: CrossSectionData,
) -> Literal["inelastic", "momentum"]:
    return (
        "momentum"
        if cross_section.kind.strip().lower() in _MOMENTUM_KINDS
        else "inelastic"
    )


def _is_neutral_gas_target(target: SpeciesData | None) -> bool:
    return target is not None and target.phase == "gas" and target.charge == 0


def _normalized_collision_curve(
    cross_section: CrossSectionData,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return normalized_cross_section_curve(cross_section)
    except ValueError as exc:
        raise CaseValidationError(str(exc)) from exc


def _compile_collision(cross_section: CrossSectionData) -> ElectronCollision:
    """Translate one validated-target cross section to the EEDF contract."""

    kind = _collision_kind(cross_section)
    electron_transfer = cross_section.resolved_electron_energy_transfer_eV
    if kind == "inelastic" and electron_transfer > 0.0:
        raise CaseValidationError(
            "experimental.approximate_two_term does not support superelastic "
            f"electron heating from cross section {cross_section.id!r}"
        )
    energy_eV, sigma_m2 = _normalized_collision_curve(cross_section)
    return ElectronCollision(
        collision_id=cross_section.id,
        target_species=cross_section.target,
        kind=kind,
        energy_eV=energy_eV,
        cross_section_m2=sigma_m2,
        energy_loss_eV=0.0 if kind == "momentum" else -electron_transfer,
    )


def _validate_collision_set(
    collisions: Sequence[ElectronCollision], invalid_targets: Sequence[str]
) -> None:
    """Report deferred target errors before checking the required kernel."""

    if invalid_targets:
        raise CaseValidationError(
            "experimental.approximate_two_term only accepts neutral gas targets; "
            f"invalid targets: {', '.join(sorted(set(invalid_targets)))}"
        )
    if not any(collision.kind == "momentum" for collision in collisions):
        raise CaseValidationError(
            "experimental.approximate_two_term requires a momentum-transfer "
            "cross section"
        )


def _compile_collisions(chemistry: ChemistryData) -> tuple[ElectronCollision, ...]:
    if not chemistry.cross_sections:
        raise CaseValidationError(
            "experimental.approximate_two_term requires electron-neutral cross sections"
        )
    species = {item.id: item for item in chemistry.species}
    invalid_targets: list[str] = []
    collisions: list[ElectronCollision] = []
    for cross_section in chemistry.cross_sections.values():
        target = species.get(cross_section.target)
        if not _is_neutral_gas_target(target):
            invalid_targets.append(cross_section.target)
            continue
        collisions.append(_compile_collision(cross_section))
    _validate_collision_set(collisions, invalid_targets)
    return tuple(collisions)


def _collision_targets(
    collisions: tuple[ElectronCollision, ...],
) -> tuple[str, ...]:
    return tuple(sorted({collision.target_species for collision in collisions}))


def _neutral_gas_species(chemistry: ChemistryData) -> tuple[str, ...]:
    return tuple(
        sorted(
            species.id
            for species in chemistry.species
            if species.phase == "gas" and species.charge == 0
        )
    )


def _validated_fixed_densities(
    zone_id: str,
    densities_m3: Mapping[str, float],
    targets: tuple[str, ...],
) -> tuple[dict[str, float], float]:
    values = {target: float(densities_m3.get(target, 0.0)) for target in targets}
    invalid = [
        target
        for target, density in values.items()
        if not math.isfinite(density) or density < 0.0
    ]
    if invalid:
        raise CaseValidationError(
            f"zone {zone_id!r} has invalid initial collision-target densities: "
            f"{', '.join(invalid)}"
        )
    total = sum(values.values())
    if total <= 0.0:
        raise CaseValidationError(
            f"zone {zone_id!r} has no positive initial neutral collision-target density"
        )
    return values, total


def _missing_momentum_targets(
    values: Mapping[str, float],
    momentum_targets: frozenset[str],
) -> list[str]:
    return sorted(
        target
        for target, density in values.items()
        if density > 0.0 and target not in momentum_targets
    )


def _validate_neutral_collision_coverage(
    zone_id: str,
    densities_m3: Mapping[str, float],
    neutral_species: tuple[str, ...],
    targets: tuple[str, ...],
) -> None:
    neutral_values = {
        species_id: float(densities_m3.get(species_id, 0.0))
        for species_id in neutral_species
    }
    invalid = sorted(
        species_id
        for species_id, density in neutral_values.items()
        if not math.isfinite(density) or density < 0.0
    )
    if invalid:
        raise CaseValidationError(
            f"zone {zone_id!r} has invalid initial neutral densities: "
            f"{', '.join(invalid)}"
        )
    missing = sorted(
        species_id
        for species_id, density in neutral_values.items()
        if density > 0.0 and species_id not in targets
    )
    if missing:
        raise CaseValidationError(
            f"zone {zone_id!r} has positive neutral species without electron "
            "collision cross sections: "
            f"{', '.join(missing)}"
        )


def _fixed_mixture(
    zone_id: str,
    densities_m3: Mapping[str, float],
    targets: tuple[str, ...],
    neutral_species: tuple[str, ...],
    momentum_targets: frozenset[str],
) -> tuple[dict[str, float], dict[str, float]]:
    _validate_neutral_collision_coverage(
        zone_id, densities_m3, neutral_species, targets
    )
    values, total = _validated_fixed_densities(zone_id, densities_m3, targets)
    missing_momentum = _missing_momentum_targets(values, momentum_targets)
    if missing_momentum:
        raise CaseValidationError(
            f"zone {zone_id!r} has positive collision targets without "
            "momentum-transfer "
            f"cross sections: {', '.join(missing_momentum)}"
        )
    fractions = {
        target: density / total for target, density in values.items() if density > 0.0
    }
    return values, fractions


def _mixture_identity(
    fractions: Mapping[str, float], targets: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    """Return the exact composition that determines one local-field table."""

    return tuple((target, fractions.get(target, 0.0).hex()) for target in targets)


def _readonly(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _prepare_table(
    *,
    source: Path,
    model: ApproximateTwoTermEEDF,
    field_grid_Td: np.ndarray,
    target_densities_m3: Mapping[str, float],
) -> TabulatedElectronKinetics:
    results = tuple(
        model.evaluate(float(field), target_densities_m3) for field in field_grid_Td
    )
    mean_energy = [result.mean_energy_eV for result in results]
    mobility = [result.mobility_m2_V_s for result in results]
    rate_ids = tuple(collision.collision_id for collision in model.collisions)
    rates = {
        rate_id: _readonly(
            [0.0, *(result.rate_coefficients_m3_s[rate_id] for result in results)]
        )
        for rate_id in rate_ids
    }
    return TabulatedElectronKinetics(
        source=source,
        lookup="local_field",
        # The explicit cold node closes only the interval to the first solved
        # field; values beyond the last solved field still fail explicitly.
        bounds="error",
        axis=_readonly([0.0, *field_grid_Td]),
        mean_energy_eV=_readonly([0.0, *mean_energy]),
        mobility_m2_V_s=_readonly([mobility[0], *mobility]),
        effective_field_Td=_readonly([0.0, *field_grid_Td]),
        rate_tables=MappingProxyType(rates),
        mobility_reference_neutral_density_m3=sum(target_densities_m3.values()),
    )


def _prepared_grid_and_model(
    collisions: tuple[ElectronCollision, ...],
    *,
    energy_min_eV: float,
    energy_max_eV: float,
    energy_points: int,
    field_min_Td: float,
    field_max_Td: float,
    field_points: int,
    max_iterations: int,
) -> tuple[np.ndarray, ApproximateTwoTermEEDF]:
    try:
        field_grid = np.geomspace(field_min_Td, field_max_Td, field_points)
        model = ApproximateTwoTermEEDF(
            collisions=collisions,
            energy_min_eV=energy_min_eV,
            energy_max_eV=energy_max_eV,
            energy_points=energy_points,
            max_iterations=max_iterations,
        )
    except (TypeError, ValueError) as exc:
        raise CaseValidationError(
            f"invalid approximate two-term preparation grid: {exc}"
        ) from exc
    if (
        field_grid.ndim != 1
        or field_grid.size < 2
        or not np.isfinite(field_grid).all()
        or np.any(np.diff(field_grid) <= 0.0)
    ):
        raise CaseValidationError(
            "approximate two-term reduced-field grid must be finite and strictly "
            "increasing"
        )
    field_grid.setflags(write=False)
    return field_grid, model


def _new_mixture_table(
    *,
    chemistry: ChemistryData,
    model: ApproximateTwoTermEEDF,
    field_grid_Td: np.ndarray,
    densities_m3: Mapping[str, float],
    zone_id: str,
) -> TabulatedElectronKinetics:
    try:
        return _prepare_table(
            source=chemistry.source,
            model=model,
            field_grid_Td=field_grid_Td,
            target_densities_m3=densities_m3,
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise CaseValidationError(
            f"failed to prepare approximate two-term kinetics for zone "
            f"{zone_id!r}: {exc}"
        ) from exc


def _zone_mixture_provenance(
    *,
    table_index: int,
    densities: Mapping[str, float],
    fractions: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "table_index": table_index,
        "target_densities_m3": dict(densities),
        "target_fractions": dict(fractions),
    }


def _prepare_zone_tables(
    chemistry: ChemistryData,
    initial_densities_m3_by_zone: Mapping[str, Mapping[str, float]],
    *,
    model: ApproximateTwoTermEEDF,
    field_grid_Td: np.ndarray,
    targets: tuple[str, ...],
    neutral_species: tuple[str, ...],
    momentum_targets: frozenset[str],
    cache_max_entries: int,
) -> tuple[
    dict[tuple[tuple[str, str], ...], TabulatedElectronKinetics],
    dict[str, TabulatedElectronKinetics],
    dict[str, Any],
]:
    tables: dict[tuple[tuple[str, str], ...], TabulatedElectronKinetics] = {}
    zone_tables: dict[str, TabulatedElectronKinetics] = {}
    zone_provenance: dict[str, Any] = {}
    for raw_zone_id, raw_densities in initial_densities_m3_by_zone.items():
        zone_id = str(raw_zone_id)
        densities, fractions = _fixed_mixture(
            zone_id,
            raw_densities,
            targets,
            neutral_species,
            momentum_targets,
        )
        identity = _mixture_identity(fractions, targets)
        if identity not in tables:
            if len(tables) >= cache_max_entries:
                raise CaseValidationError(
                    "distinct initial neutral mixtures exceed cache.max_entries "
                    f"({cache_max_entries})"
                )
            tables[identity] = _new_mixture_table(
                chemistry=chemistry,
                model=model,
                field_grid_Td=field_grid_Td,
                densities_m3=densities,
                zone_id=zone_id,
            )
        table_index = tuple(tables).index(identity)
        zone_tables[zone_id] = tables[identity]
        zone_provenance[zone_id] = _zone_mixture_provenance(
            table_index=table_index,
            densities=densities,
            fractions=fractions,
        )
    return tables, zone_tables, zone_provenance


def prepare_approximate_two_term_kinetics(
    chemistry: ChemistryData,
    initial_densities_m3_by_zone: Mapping[str, Mapping[str, float]],
    *,
    cache_max_entries: int = 12,
    energy_min_eV: float = 1.0e-3,
    energy_max_eV: float = 160.0,
    energy_points: int = 360,
    field_min_Td: float = 1.0,
    field_max_Td: float = 100.0,
    field_points: int = 48,
    max_iterations: int = 48,
) -> PreparedApproximateTwoTermKinetics:
    """Prepare one strict local-field table per exact neutral composition."""

    if not initial_densities_m3_by_zone:
        raise CaseValidationError("approximate two-term preparation requires zones")
    cache_max_entries = int(cache_max_entries)
    energy_min_eV = float(energy_min_eV)
    energy_max_eV = float(energy_max_eV)
    energy_points = int(energy_points)
    field_min_Td = float(field_min_Td)
    field_max_Td = float(field_max_Td)
    field_points = int(field_points)
    max_iterations = int(max_iterations)
    if cache_max_entries < 1:
        raise CaseValidationError("approximate two-term cache settings are invalid")
    collisions = _compile_collisions(chemistry)
    targets = _collision_targets(collisions)
    neutral_species = _neutral_gas_species(chemistry)

    field_grid, model = _prepared_grid_and_model(
        collisions,
        energy_min_eV=energy_min_eV,
        energy_max_eV=energy_max_eV,
        energy_points=energy_points,
        field_min_Td=field_min_Td,
        field_max_Td=field_max_Td,
        field_points=field_points,
        max_iterations=max_iterations,
    )

    momentum_targets = frozenset(
        collision.target_species
        for collision in collisions
        if collision.kind == "momentum"
    )
    tables, zone_tables, zone_provenance = _prepare_zone_tables(
        chemistry,
        initial_densities_m3_by_zone,
        model=model,
        field_grid_Td=field_grid,
        targets=targets,
        neutral_species=neutral_species,
        momentum_targets=momentum_targets,
        cache_max_entries=cache_max_entries,
    )

    provenance = {
        "kind": "experimental.approximate_two_term",
        "classification": "experimental_fixed_initial_neutral_mixture",
        "fixed_mixture": True,
        "runtime_eedf_solve": False,
        "closure": "local_field",
        "preparation": "compile_time_strict_geometric_grid",
        "bounds_policy": "error_outside_prepared_grid",
        "zero_field_policy": "continuous_cold_boundary_to_first_grid_node",
        "mobility_scaling": "inverse_current_neutral_density",
        "chemistry_manifest": str(chemistry.source),
        "cross_section_ids": tuple(collision.collision_id for collision in collisions),
        "mixture_identity": "exact_full_target_fractions",
        "energy_grid_eV": {
            "minimum": energy_min_eV,
            "maximum": energy_max_eV,
            "count": energy_points,
        },
        "reduced_field_grid_Td": {
            "minimum": float(field_grid[0]),
            "maximum": float(field_grid[-1]),
            "count": int(field_grid.size),
        },
        "unique_table_count": len(tables),
        "zones": zone_provenance,
    }
    return PreparedApproximateTwoTermKinetics(zone_tables, provenance)


__all__ = [
    "PreparedApproximateTwoTermKinetics",
    "prepare_approximate_two_term_kinetics",
]
