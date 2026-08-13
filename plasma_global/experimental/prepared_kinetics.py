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
from typing import Any, Literal, SupportsFloat, SupportsIndex

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


def _python_float(value: SupportsFloat) -> float:
    return float(value)


def _python_int(value: SupportsIndex) -> int:
    return int(value)


def _string_id(value: object) -> str:
    return str(value)


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


def _preparation_species(
    collisions: tuple[ElectronCollision, ...],
    mixture_key_species: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    targets = tuple(sorted({collision.target_species for collision in collisions}))
    requested_key = tuple(_string_id(value) for value in mixture_key_species)
    if len(set(requested_key)) != len(requested_key):
        raise CaseValidationError("mixture_key_species must be unique")
    unknown_key = sorted(set(requested_key) - set(targets))
    if unknown_key:
        raise CaseValidationError(
            "mixture_key_species are not electron-neutral collision targets: "
            f"{', '.join(unknown_key)}"
        )
    return targets, requested_key


def _fixed_mixture(
    zone_id: str,
    densities_m3: Mapping[str, float],
    targets: tuple[str, ...],
    momentum_targets: frozenset[str],
) -> tuple[dict[str, float], dict[str, float]]:
    values = {
        target: _python_float(densities_m3.get(target, 0.0)) for target in targets
    }
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
    missing_momentum = sorted(
        target
        for target, density in values.items()
        if density > 0.0 and target not in momentum_targets
    )
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
    densities_m3: Mapping[str, float], targets: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    # Exact float identities prevent two physically different initial mixtures
    # from silently sharing mobility or rate data.  Human-facing rounded keys
    # remain provenance only.
    return tuple((target, densities_m3[target].hex()) for target in targets)


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
    rate_ids = tuple(collision.collision_id for collision in model.collisions)
    rates = {
        rate_id: _readonly(
            [result.rate_coefficients_m3_s[rate_id] for result in results]
        )
        for rate_id in rate_ids
    }
    return TabulatedElectronKinetics(
        source=source,
        lookup="local_field",
        # A positive-field swarm state is not a valid substitute for E/N=0.
        # Ports outside the prepared domain therefore fail explicitly.
        bounds="error",
        axis=_readonly(field_grid_Td),
        mean_energy_eV=_readonly([result.mean_energy_eV for result in results]),
        mobility_m2_V_s=_readonly([result.mobility_m2_V_s for result in results]),
        effective_field_Td=_readonly(field_grid_Td),
        rate_tables=MappingProxyType(rates),
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
    key_species: tuple[str, ...],
    fraction_decimals: int,
) -> dict[str, Any]:
    return {
        "table_index": table_index,
        "target_densities_m3": dict(densities),
        "target_fractions": dict(fractions),
        "rounded_mixture_key": {
            species: round(fractions.get(species, 0.0), fraction_decimals)
            for species in key_species
        },
    }


def _prepare_zone_tables(
    chemistry: ChemistryData,
    initial_densities_m3_by_zone: Mapping[str, Mapping[str, float]],
    *,
    model: ApproximateTwoTermEEDF,
    field_grid_Td: np.ndarray,
    targets: tuple[str, ...],
    momentum_targets: frozenset[str],
    key_species: tuple[str, ...],
    cache_max_entries: int,
    fraction_decimals: int,
) -> tuple[
    dict[tuple[tuple[str, str], ...], TabulatedElectronKinetics],
    dict[str, TabulatedElectronKinetics],
    dict[str, Any],
]:
    tables: dict[tuple[tuple[str, str], ...], TabulatedElectronKinetics] = {}
    zone_tables: dict[str, TabulatedElectronKinetics] = {}
    zone_provenance: dict[str, Any] = {}
    for raw_zone_id, raw_densities in initial_densities_m3_by_zone.items():
        zone_id = _string_id(raw_zone_id)
        densities, fractions = _fixed_mixture(
            zone_id, raw_densities, targets, momentum_targets
        )
        identity = _mixture_identity(densities, targets)
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
            key_species=key_species,
            fraction_decimals=fraction_decimals,
        )
    return tables, zone_tables, zone_provenance


def prepare_approximate_two_term_kinetics(
    chemistry: ChemistryData,
    initial_densities_m3_by_zone: Mapping[str, Mapping[str, float]],
    *,
    mixture_key_species: Sequence[str] = (),
    cache_max_entries: int = 12,
    fraction_decimals: int = 3,
    energy_min_eV: float = 1.0e-3,
    energy_max_eV: float = 160.0,
    energy_points: int = 360,
    field_min_Td: float = 1.0,
    field_max_Td: float = 100.0,
    field_points: int = 48,
    max_iterations: int = 48,
) -> PreparedApproximateTwoTermKinetics:
    """Prepare one strict local-field table per distinct initial neutral mixture."""

    if not initial_densities_m3_by_zone:
        raise CaseValidationError("approximate two-term preparation requires zones")
    cache_max_entries = _python_int(cache_max_entries)
    fraction_decimals = _python_int(fraction_decimals)
    energy_min_eV = _python_float(energy_min_eV)
    energy_max_eV = _python_float(energy_max_eV)
    energy_points = _python_int(energy_points)
    field_min_Td = _python_float(field_min_Td)
    field_max_Td = _python_float(field_max_Td)
    field_points = _python_int(field_points)
    max_iterations = _python_int(max_iterations)
    if cache_max_entries < 1 or fraction_decimals < 0:
        raise CaseValidationError("approximate two-term cache settings are invalid")
    collisions = _compile_collisions(chemistry)
    targets, requested_key = _preparation_species(collisions, mixture_key_species)

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
        momentum_targets=momentum_targets,
        key_species=requested_key or targets,
        cache_max_entries=cache_max_entries,
        fraction_decimals=fraction_decimals,
    )

    provenance = {
        "kind": "experimental.approximate_two_term",
        "classification": "experimental_fixed_initial_neutral_mixture",
        "fixed_mixture": True,
        "runtime_eedf_solve": False,
        "closure": "local_field",
        "preparation": "compile_time_strict_geometric_grid",
        "bounds_policy": "error_outside_prepared_grid",
        "zero_field_policy": "cold_electron_zero_rates",
        "chemistry_manifest": str(chemistry.source),
        "cross_section_ids": tuple(collision.collision_id for collision in collisions),
        "mixture_key_species": requested_key or targets,
        "energy_grid_eV": {
            "minimum": energy_min_eV,
            "maximum": energy_max_eV,
            "count": energy_points,
        },
        "reduced_field_grid_Td": {
            "minimum": _python_float(field_grid[0]),
            "maximum": _python_float(field_grid[-1]),
            "count": _python_int(field_grid.size),
        },
        "unique_table_count": len(tables),
        "zones": zone_provenance,
    }
    return PreparedApproximateTwoTermKinetics(zone_tables, provenance)


__all__ = [
    "PreparedApproximateTwoTermKinetics",
    "prepare_approximate_two_term_kinetics",
]
