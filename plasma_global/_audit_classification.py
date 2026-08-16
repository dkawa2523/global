"""Classify the supported standard-model envelope for case audits."""

from __future__ import annotations

from typing import Any


def _heavy_ions(chemistry_data: Any) -> tuple[Any, ...]:
    return tuple(
        item
        for item in chemistry_data.species
        if item.phase == "gas" and item.id != "e" and item.charge != 0
    )


def _standard_wall_transport(
    surface: Any,
    positive_ions: tuple[Any, ...],
    has_negative_ion: bool,
) -> bool:
    if surface.wall_transport.kind == "off":
        return True
    return (
        all(
            (
                surface.wall_transport.kind == "bohm",
                len(positive_ions) == 1,
                not has_negative_ion,
                surface.ion_impact_energy_eV == 0.0,
            )
        )
        and positive_ions[0].charge == 1
    )


def _is_applicable_wall_branch(reaction: Any, surface: Any, ion_id: str) -> bool:
    return all(
        (
            reaction.reactants == {ion_id: 1.0},
            bool(reaction.products),
            not reaction.zones or surface.zone_id in reaction.zones,
            not reaction.surfaces or surface.surface_id in reaction.surfaces,
        )
    )


def _has_standard_wall_branch(
    surface: Any,
    chemistry_data: Any,
    positive_ions: tuple[Any, ...],
    has_negative_ion: bool,
) -> bool:
    if surface.wall_transport.kind == "off":
        return True
    if not _standard_wall_transport(surface, positive_ions, has_negative_ion):
        return False
    ion_id = positive_ions[0].id
    branches = [
        reaction
        for reaction in chemistry_data.boundary_reactions
        if _is_applicable_wall_branch(reaction, surface, ion_id)
    ]
    return len(branches) == 1


def _electron_reactions_use_cross_sections(chemistry_data: Any) -> bool:
    return all(
        "e" not in reaction.reactants
        or (
            reaction.rate_model is not None
            and chemistry_data.rate_models[str(reaction.rate_model)].kind
            == "electron_impact"
        )
        for reaction in chemistry_data.gas_reactions
    )


def case_classification(case: Any, chemistry_data: Any) -> str:
    """Classify the deliberately small, validated standard-model envelope."""

    heavy_ions = _heavy_ions(chemistry_data)
    positive_ions = tuple(item for item in heavy_ions if item.charge > 0)
    has_negative_ion = any(item.charge < 0 for item in heavy_ions)
    standard_walls = all(
        _has_standard_wall_branch(
            surface, chemistry_data, positive_ions, has_negative_ion
        )
        for surface in case.reactor.surfaces
    )
    one_singly_charged_positive_ion = (
        len(positive_ions) == 1 and positive_ions[0].charge == 1
    )
    standard = all(
        (
            case.models.electrons.kind == "maxwellian",
            case.models.electron_closure.kind == "electron_energy",
            case.models.electron_density.kind == "quasineutral",
            case.models.gas_energy.kind == "fixed",
            all(
                port.model.kind == "prescribed_power"
                for port in case.reactor.power_ports
            ),
            case.models.surface_kinetics is None,
            not chemistry_data.surface_reactions,
            standard_walls,
            _electron_reactions_use_cross_sections(chemistry_data),
            not has_negative_ion,
            one_singly_charged_positive_ion,
            not any(
                str(model.kind).startswith("experimental.")
                for model in chemistry_data.rate_models.values()
            ),
            not chemistry_data.experimental,
            case.experimental is None,
        )
    )
    return "standard" if standard else "experimental"


__all__ = ["case_classification"]
