"""Scientific and dimensional validation for chemistry compilation."""

from __future__ import annotations

import math
from collections.abc import Mapping

from plasma_global.chemistry._contracts import (
    cross_section_data_error,
    cross_section_support_error,
    surface_reaction_shape_error,
)
from plasma_global.chemistry.data import (
    ChemistryData,
    CrossSectionData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.errors import ChemistryError

_DIMENSIONAL_GAS_RATE_KINDS = frozenset(
    {
        "arrhenius",
        "constant",
        "experimental.electron_temperature_power_law",
        "tabulated_1d",
    }
)

# Published atomic weights and independently rounded fragment masses commonly
# differ by a few parts in 1e5.  Larger mismatches are input errors at the
# accuracy represented by this zero-dimensional heavy-particle model.
# Neutral/ion tables may either include or omit electron rest mass, so a
# reaction may satisfy the raw or charge-corrected heavy-mass convention.
_HEAVY_MASS_RELATIVE_TOLERANCE = 1.0e-4


def validate_unique_ids(values: list[str], kind: str) -> None:
    seen: set[str] = set()
    duplicate = sorted({value for value in values if value in seen or seen.add(value)})
    if duplicate:
        raise ChemistryError(f"duplicate {kind} ids: {', '.join(duplicate)}")


def _normalized_residual(delta: float, scale: float) -> float:
    return abs(delta) / max(abs(scale), 1.0)


def _reaction_elements(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                element
                for species_id in (*reaction.reactants, *reaction.products)
                for element in species[species_id].elements
            }
        )
    )


def _element_total(
    terms: Mapping[str, float],
    species: Mapping[str, SpeciesData],
    element: str,
) -> float:
    return sum(
        order * species[species_id].elements.get(element, 0.0)
        for species_id, order in terms.items()
    )


def _validate_element_balance(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> None:
    for element in _reaction_elements(reaction, species):
        before = _element_total(reaction.reactants, species, element)
        after = _element_total(reaction.products, species, element)
        if _normalized_residual(after - before, before) > 1.0e-12:
            raise ChemistryError(f"reaction {reaction.id} does not conserve {element}")


def _charge_total(
    terms: Mapping[str, float], species: Mapping[str, SpeciesData]
) -> float:
    return sum(
        order * species[species_id].charge for species_id, order in terms.items()
    )


def _heavy_mass_total(
    terms: Mapping[str, float], species: Mapping[str, SpeciesData]
) -> float:
    return sum(
        order * species[species_id].mass_kg
        for species_id, order in terms.items()
        if species_id != "e"
    )


def _charge_corrected_heavy_mass_total(
    terms: Mapping[str, float], species: Mapping[str, SpeciesData]
) -> float:
    electron_mass = species["e"].mass_kg
    return sum(
        order
        * (species[species_id].mass_kg + species[species_id].charge * electron_mass)
        for species_id, order in terms.items()
        if species_id != "e"
    )


def _mass_close(before: float, after: float) -> bool:
    return math.isclose(
        before,
        after,
        rel_tol=_HEAVY_MASS_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    )


def _validate_heavy_mass_balance(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> None:
    heavy_before = _heavy_mass_total(reaction.reactants, species)
    heavy_after = _heavy_mass_total(reaction.products, species)
    corrected_before = _charge_corrected_heavy_mass_total(reaction.reactants, species)
    corrected_after = _charge_corrected_heavy_mass_total(reaction.products, species)
    if not (
        _mass_close(heavy_before, heavy_after)
        or _mass_close(corrected_before, corrected_after)
    ):
        raise ChemistryError(
            f"reaction {reaction.id} does not conserve heavy-species mass"
        )


def _validate_reaction_balance(
    reaction: ReactionData,
    species: Mapping[str, SpeciesData],
    *,
    boundary: bool,
) -> None:
    unknown = sorted((set(reaction.reactants) | set(reaction.products)) - set(species))
    if unknown:
        raise ChemistryError(
            f"reaction {reaction.id} references unknown species: {', '.join(unknown)}"
        )

    _validate_element_balance(reaction, species)
    before_charge = _charge_total(reaction.reactants, species)
    after_charge = _charge_total(reaction.products, species)
    if (
        not boundary
        and _normalized_residual(after_charge - before_charge, before_charge) > 1.0e-12
    ):
        raise ChemistryError(f"reaction {reaction.id} does not conserve charge")


def _validate_cross_section_energy(cross_section: CrossSectionData) -> None:
    legacy_loss = cross_section.energy_loss_eV
    signed_transfer = cross_section.electron_energy_transfer_eV
    has_legacy_loss = legacy_loss is not None
    has_signed_transfer = signed_transfer is not None
    if has_legacy_loss == has_signed_transfer:
        raise ChemistryError(
            f"cross section {cross_section.id!r} must define exactly one of "
            "energy_loss_eV or electron_energy_transfer_eV"
        )
    if legacy_loss is not None and (
        not math.isfinite(legacy_loss) or legacy_loss < 0.0
    ):
        raise ChemistryError(
            f"cross section {cross_section.id!r} has invalid energy_loss_eV"
        )
    if signed_transfer is not None and not math.isfinite(signed_transfer):
        raise ChemistryError(
            f"cross section {cross_section.id!r} has invalid "
            "electron_energy_transfer_eV"
        )


def _validate_cross_section_target(
    cross_section: CrossSectionData,
    species_by_id: Mapping[str, SpeciesData],
) -> None:
    target = species_by_id.get(cross_section.target)
    if target is None or target.phase != "gas" or target.id == "e":
        raise ChemistryError(
            f"cross section {cross_section.id} target {cross_section.target!r} "
            "must be a declared heavy gas species"
        )


def _validate_cross_section(
    cross_section: CrossSectionData,
    species_by_id: Mapping[str, SpeciesData],
) -> None:
    _validate_cross_section_energy(cross_section)
    data_error = cross_section_data_error(
        cross_section_id=cross_section.id,
        kind=cross_section.kind,
        target=cross_section.target,
        threshold_eV=cross_section.threshold_eV,
        energy_eV=cross_section.energy_eV,
        sigma_m2=cross_section.sigma_m2,
    )
    if data_error is not None:
        raise ChemistryError(data_error)
    support_error = cross_section_support_error(cross_section)
    if support_error is not None:
        raise ChemistryError(support_error)
    _validate_cross_section_target(cross_section, species_by_id)


def validate_species(
    data: ChemistryData,
) -> tuple[dict[str, SpeciesData], tuple[SpeciesData, ...]]:
    species_by_id = {item.id: item for item in data.species}
    electron = species_by_id.get("e")
    if electron is None or electron.phase != "gas" or electron.charge != -1:
        raise ChemistryError(
            "canonical chemistry must contain exactly one gas electron species 'e' "
            "with charge -1"
        )
    for cross_section in data.cross_sections.values():
        _validate_cross_section(cross_section, species_by_id)
    gas_species = tuple(
        item for item in data.species if item.phase == "gas" and item.id != "e"
    )
    return species_by_id, gas_species


def _validate_electron_impact_reaction(
    reaction: ReactionData,
    rate_model: RateModelData,
    cross_sections: Mapping[str, CrossSectionData],
) -> None:
    cross_section = cross_sections.get(str(rate_model.parameters["cross_section"]))
    if cross_section is None:
        raise ChemistryError(
            f"reaction {reaction.id} references an unknown cross section"
        )
    if cross_section.kind == "momentum_transfer":
        raise ChemistryError(
            f"reaction {reaction.id} cannot use momentum-transfer cross section "
            f"{cross_section.id!r} as a reactive rate"
        )
    target = cross_section.target
    expected_reactants = {"e": 1.0, target: 1.0}
    if reaction.reactants != expected_reactants:
        raise ChemistryError(
            f"electron-impact reaction {reaction.id} must have exactly one unit "
            f"electron and one unit cross-section target {target!r} as reactants"
        )


def _validate_gas_rate_shape(
    data: ChemistryData,
    reaction: ReactionData,
    rate_model: RateModelData,
) -> None:
    if rate_model.kind == "electron_impact":
        _validate_electron_impact_reaction(reaction, rate_model, data.cross_sections)
    elif rate_model.kind == "first_order" and (
        len(reaction.reactants) != 1 or next(iter(reaction.reactants.values())) != 1.0
    ):
        raise ChemistryError(
            f"first-order reaction {reaction.id} must have exactly one unit reactant"
        )


def _validated_gas_rate_model(
    data: ChemistryData,
    reaction: ReactionData,
    species_by_id: Mapping[str, SpeciesData],
) -> RateModelData:
    _validate_reaction_balance(reaction, species_by_id, boundary=False)
    if any(
        species_by_id[item].phase != "gas"
        for item in (*reaction.reactants, *reaction.products)
    ):
        raise ChemistryError(f"gas reaction {reaction.id} contains a surface species")
    _validate_heavy_mass_balance(reaction, species_by_id)
    if reaction.rate_model not in data.rate_models:
        raise ChemistryError(
            f"reaction {reaction.id} references unknown rate model "
            f"{reaction.rate_model!r}"
        )
    rate_model = data.rate_models[reaction.rate_model]
    _validate_gas_rate_shape(data, reaction, rate_model)
    return rate_model


def validated_gas_rate_models(
    data: ChemistryData, species_by_id: Mapping[str, SpeciesData]
) -> tuple[RateModelData, ...]:
    return tuple(
        _validated_gas_rate_model(data, reaction, species_by_id)
        for reaction in data.gas_reactions
    )


def _reaction_order(reaction: ReactionData) -> float:
    orders = tuple(reaction.reactants.values())
    if any(not math.isfinite(value) or value < 0.0 for value in orders):
        raise ChemistryError(
            f"reaction {reaction.id!r} reactant orders must be finite and nonnegative"
        )
    return float(sum(orders))


def _rate_coefficient_unit(overall_order: float) -> str:
    length_exponent = 3.0 * (overall_order - 1.0)
    rounded_exponent = round(length_exponent)
    if math.isclose(length_exponent, rounded_exponent, rel_tol=0.0, abs_tol=1.0e-12):
        length_exponent = float(rounded_exponent)
    if length_exponent == 0.0:
        return "s^-1"
    return f"m{length_exponent:g}/s"


def _declared_rate_order(rate_model: RateModelData) -> float | None:
    value = rate_model.parameters.get("overall_order")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChemistryError(
            f"rate model {rate_model.id!r} overall_order must be a number"
        )
    order = float(value)
    if not math.isfinite(order) or order < 0.0:
        raise ChemistryError(
            f"rate model {rate_model.id!r} overall_order must be finite and nonnegative"
        )
    return order


def validate_gas_rate_dimensions(
    reactions: tuple[ReactionData, ...], rate_models: tuple[RateModelData, ...]
) -> None:
    inferred_order_by_model: dict[str, tuple[float, str]] = {}
    for reaction, rate_model in zip(reactions, rate_models, strict=True):
        if rate_model.kind not in _DIMENSIONAL_GAS_RATE_KINDS:
            continue
        actual_order = _reaction_order(reaction)
        declared_order = _declared_rate_order(rate_model)
        if declared_order is not None and not math.isclose(
            declared_order, actual_order, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ChemistryError(
                f"rate model {rate_model.id!r} declares overall_order "
                f"{declared_order:g} ({_rate_coefficient_unit(declared_order)}) but "
                f"reaction {reaction.id!r} has reactant order {actual_order:g} "
                f"({_rate_coefficient_unit(actual_order)})"
            )
        previous = inferred_order_by_model.get(rate_model.id)
        if previous is not None and not math.isclose(
            previous[0], actual_order, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ChemistryError(
                f"legacy rate model {rate_model.id!r} is reused by reactions "
                f"{previous[1]!r} and {reaction.id!r} with incompatible overall "
                f"orders {previous[0]:g} ({_rate_coefficient_unit(previous[0])}) "
                f"and {actual_order:g} ({_rate_coefficient_unit(actual_order)}); "
                "split the model and declare overall_order on each coefficient"
            )
        inferred_order_by_model[rate_model.id] = (actual_order, reaction.id)


def _validate_surface_reaction(
    data: ChemistryData,
    reaction: ReactionData,
    species_by_id: Mapping[str, SpeciesData],
) -> None:
    _validate_reaction_balance(reaction, species_by_id, boundary=False)
    if reaction.rate_model not in data.rate_models:
        raise ChemistryError(
            f"surface reaction {reaction.id} references unknown rate model "
            f"{reaction.rate_model!r}"
        )
    shape_error = surface_reaction_shape_error(
        reaction, data.rate_models[reaction.rate_model], species_by_id
    )
    if shape_error is not None:
        raise ChemistryError(shape_error)


def _validate_non_gas_energy_fields(reaction: ReactionData, *, family: str) -> None:
    if any(
        value is not None and value != 0.0
        for value in (
            reaction.energy_loss_eV,
            reaction.electron_energy_transfer_eV,
        )
    ):
        raise ChemistryError(
            f"{family} reaction {reaction.id!r} does not support electron-energy fields"
        )
    if family == "boundary" and reaction.gas_heating_eV != 0.0:
        raise ChemistryError(
            f"boundary reaction {reaction.id!r} does not support gas_heating_eV"
        )


def validate_non_gas_reactions(
    data: ChemistryData, species_by_id: Mapping[str, SpeciesData]
) -> None:
    for reaction in data.boundary_reactions:
        _validate_non_gas_energy_fields(reaction, family="boundary")
        _validate_reaction_balance(reaction, species_by_id, boundary=True)
        _validate_heavy_mass_balance(reaction, species_by_id)
    for reaction in data.surface_reactions:
        _validate_non_gas_energy_fields(reaction, family="surface")
        _validate_surface_reaction(data, reaction, species_by_id)
