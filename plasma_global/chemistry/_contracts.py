"""Pure reaction-shape contracts shared by chemistry compilation and models."""

from __future__ import annotations

from collections.abc import Mapping

from plasma_global.chemistry.data import RateModelData, ReactionData, SpeciesData


def surface_reaction_shape_error(
    reaction: ReactionData,
    rate_model: RateModelData,
    species_by_id: Mapping[str, SpeciesData],
) -> str | None:
    """Describe an unsupported flux-driven surface reaction, if present."""

    if rate_model.kind not in {"sticking", "ion_assisted"}:
        return None
    gas_reactants = tuple(
        (species_by_id[species_id], order)
        for species_id, order in reaction.reactants.items()
        if species_by_id[species_id].phase == "gas"
    )
    if rate_model.kind == "sticking":
        valid = (
            len(gas_reactants) == 1
            and gas_reactants[0][1] == 1.0
            and gas_reactants[0][0].charge == 0
        )
        expected = "neutral gas reactant"
    else:
        valid = (
            len(gas_reactants) == 1
            and gas_reactants[0][1] == 1.0
            and gas_reactants[0][0].charge > 0
        )
        expected = "positive-ion gas reactant"
    if valid:
        return None
    return (
        f"{rate_model.kind} surface reaction {reaction.id} must have exactly one "
        f"unit {expected}"
    )


__all__ = ["surface_reaction_shape_error"]
