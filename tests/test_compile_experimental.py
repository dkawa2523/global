from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import plasma_global.input.compile_experimental as compile_module
from plasma_global import load_case
from plasma_global.chemistry.data import ChemistryData, load_chemistry
from plasma_global.errors import CaseValidationError
from plasma_global.input.compile_experimental import (
    compile_experimental_accumulator,
)
from plasma_global.input.schema import CaseSpec

ROOT = Path(__file__).parents[1]
MINIMAL = ROOT / "tests" / "fixtures" / "v3_minimal" / "case.yaml"
SMOKE = ROOT / "examples" / "v3" / "cases" / "smoke.yaml"


def _smoke_inputs() -> tuple[CaseSpec, ChemistryData]:
    case = load_case(SMOKE)
    return case, load_chemistry(case.chemistry.manifest)


def _updated_case(case: CaseSpec, **updates: object) -> CaseSpec:
    data = case.model_dump(mode="python")
    data.update(updates)
    return CaseSpec.model_validate(data)


def test_compiles_film_and_inventory_event_contract() -> None:
    case, chemistry = _smoke_inputs()

    accumulator, features = compile_experimental_accumulator(case, chemistry, object())

    assert accumulator is not None
    assert features == ("film", "wall_inventory")
    assert accumulator.labels == (
        "film[wafer]",
        "film[grounded_wall]",
        "film[source_wall]",
        "inventory[wafer,F_reservoir]",
    )
    assert [event.event_id for event in accumulator.surface_event_templates] == [
        "S1001@wafer",
        "S1002@wafer",
    ]
    inventory_event, film_event = accumulator.surface_event_templates
    assert inventory_event.inventory_particles_per_event == {"F_reservoir": 1.0}
    assert inventory_event.film_layers_per_event == 0.0
    assert film_event.inventory_particles_per_event == {}
    assert film_event.film_layers_per_event == 1.0


def test_surface_event_templates_preserve_order_and_filter_applicability() -> None:
    case, chemistry = _smoke_inputs()
    surfaces = tuple(
        surface.model_copy(update={"site_density_m2": 0.0})
        if surface.surface_id == "source_wall"
        else surface
        for surface in case.reactor.surfaces
    )
    all_process_surfaces = replace(
        chemistry.surface_reactions[0],
        id="all_process_surfaces",
        surfaces=(),
        zones=("process",),
    )
    selected_surfaces = replace(
        chemistry.surface_reactions[1],
        id="selected_surfaces",
        surfaces=("source_wall", "missing", "wafer"),
        zones=(),
    )
    selected_chemistry = replace(
        chemistry,
        surface_reactions=(all_process_surfaces, selected_surfaces),
    )

    templates = compile_module._compile_surface_event_templates(
        selected_chemistry,
        surfaces,
        {"all_process_surfaces": 1.0},
        {("selected_surfaces", "source_wall"): {"tracked": 2.0}},
    )

    assert tuple(event.event_id for event in templates) == (
        "all_process_surfaces@wafer",
        "all_process_surfaces@grounded_wall",
        "selected_surfaces@source_wall",
    )
    assert tuple(event.zone_id for event in templates) == (
        "process",
        "process",
        "source",
    )
    assert templates[0].film_layers_per_event == 1.0
    assert templates[1].film_layers_per_event == 1.0
    assert templates[2].inventory_particles_per_event == {"tracked": 2.0}
    assert templates[2].site_density_m2 == 1.0


def test_surface_event_template_validation_precedes_unbound_inventory() -> None:
    case, chemistry = _smoke_inputs()
    surfaces = tuple(
        surface.model_copy(update={"site_density_m2": 0.0})
        if surface.surface_id == "wafer"
        else surface
        for surface in case.reactor.surfaces
    )

    with pytest.raises(
        CaseValidationError,
        match=r"experimental film event 'S1001' on 'wafer' requires positive",
    ):
        compile_module._compile_surface_event_templates(
            chemistry,
            surfaces,
            {"S1001": 1.0},
            {("missing", "wafer"): {"tracked": 1.0}},
        )


def test_absent_or_stop_only_experimental_state_needs_no_accumulator() -> None:
    case = load_case(MINIMAL)
    chemistry = load_chemistry(case.chemistry.manifest)

    assert compile_experimental_accumulator(case, chemistry, None) == (None, ())

    stop_only = _updated_case(
        case,
        experimental={
            "stop_when_quasi_steady": {
                "relative_rhs_norm_s_inv": 1.0e-3,
                "min_time_s": 0.0,
            }
        },
    )
    assert compile_experimental_accumulator(stop_only, chemistry, None) == (None, ())


def test_extensions_require_chemistry_declared_state() -> None:
    case = load_case(MINIMAL)
    chemistry = load_chemistry(case.chemistry.manifest)
    extensions = _updated_case(case, experimental={"extensions": {}})

    with pytest.raises(
        CaseValidationError,
        match=(
            r"experimental\.extensions requires "
            r"chemistry\.experimental\.state_variables"
        ),
    ):
        compile_experimental_accumulator(extensions, chemistry, None)

    invalid = replace(chemistry, experimental={"unexpected": {}})
    with pytest.raises(
        CaseValidationError,
        match="Cannot compile experimental chemistry extensions",
    ):
        compile_experimental_accumulator(extensions, invalid, None)


def test_film_requires_surface_kinetics_and_film_reaction() -> None:
    case, chemistry = _smoke_inputs()

    with pytest.raises(
        CaseValidationError,
        match=r"experimental\.film requires models\.surface_kinetics",
    ):
        compile_experimental_accumulator(case, chemistry, None)

    species = tuple(
        replace(
            item,
            state_tags=item.state_tags - {"film_fragment"},
        )
        for item in chemistry.species
    )
    without_film_fragments = replace(chemistry, species=species)
    with pytest.raises(
        CaseValidationError,
        match="requires a surface reaction with a film_fragment stoichiometric change",
    ):
        compile_experimental_accumulator(case, without_film_fragments, object())


def test_wall_inventory_rejects_empty_or_unbound_configuration() -> None:
    case, chemistry = _smoke_inputs()
    empty = _updated_case(
        case,
        experimental={"wall_inventory": {"initial_by_surface": {}, "events": []}},
    )
    with pytest.raises(
        CaseValidationError,
        match="requires at least one initial inventory key",
    ):
        compile_experimental_accumulator(empty, chemistry, object())

    data = case.model_dump(mode="python")
    data["experimental"]["film"] = None
    inventory_only = CaseSpec.model_validate(data)
    with pytest.raises(
        CaseValidationError,
        match=r"wall-inventory events require models\.surface_kinetics",
    ):
        compile_experimental_accumulator(inventory_only, chemistry, None)

    data["experimental"]["wall_inventory"]["events"][0]["reaction_id"] = "missing"
    unbound = CaseSpec.model_validate(data)
    with pytest.raises(
        CaseValidationError,
        match="reference unknown or inapplicable reaction/surface pairs",
    ):
        compile_experimental_accumulator(unbound, chemistry, object())

    inapplicable_reactions = tuple(
        replace(reaction, zones=("source",)) if reaction.id == "S1001" else reaction
        for reaction in chemistry.surface_reactions
    )
    inapplicable = replace(chemistry, surface_reactions=inapplicable_reactions)
    with pytest.raises(
        CaseValidationError,
        match="reference unknown or inapplicable reaction/surface pairs",
    ):
        compile_experimental_accumulator(inventory_only, inapplicable, object())


def test_film_event_requires_positive_site_density() -> None:
    case, chemistry = _smoke_inputs()
    data = case.model_dump(mode="python")
    data["reactor"]["surfaces"][0]["site_density_m2"] = 0.0
    zero_site_density = CaseSpec.model_validate(data)

    with pytest.raises(
        CaseValidationError,
        match="requires positive site_density_m2",
    ):
        compile_experimental_accumulator(zero_site_density, chemistry, object())


def test_runtime_construction_error_is_reported_as_case_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_case(MINIMAL)
    chemistry = load_chemistry(case.chemistry.manifest)
    inventory_only = _updated_case(
        case,
        experimental={
            "wall_inventory": {
                "initial_by_surface": {"wall": {"tracked": 0.0}},
                "events": [],
            }
        },
    )

    def invalid_runtime(**_kwargs: object) -> None:
        raise ValueError("invalid runtime")

    monkeypatch.setattr(
        compile_module, "ExperimentalRuntimeAccumulator", invalid_runtime
    )
    with pytest.raises(
        CaseValidationError,
        match="Cannot compile experimental runtime accumulator: invalid runtime",
    ):
        compile_module.compile_experimental_accumulator(inventory_only, chemistry, None)
