"""Static model discovery derived from the accepted configuration contracts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, get_args

from pydantic import BaseModel

from plasma_global.chemistry._rate_models import RATE_DEFINITIONS
from plasma_global.input._schema_run import ExperimentalConfig
from plasma_global.input.schema import (
    ModelsDefinition,
    PowerDefinition,
    ReactorDefinition,
    SurfaceDefinition,
)

# These entries describe usage and real example relationships. Accepted kinds,
# field names, required fields, and defaults come from the contracts below.
_DETAILS = {
    ("electrons", "energy_balance"): (
        "Integrate electron energy and derive electron temperature.",
        "Provide initial.mean_energy_eV; coefficients use Maxwellian cross sections "
        "or a supplied mean-energy table. Specify signed reaction energy transfers.",
        "argon_maxwellian",
        True,
    ),
    ("electrons", "local_field"): (
        "Obtain electron energy and coefficients from the local reduced field.",
        "Requires an E/N coefficient CSV and a power port that determines E/N "
        "throughout every run segment; no electron-energy ODE is integrated.",
        "argon_lxcat",
        True,
    ),
    ("electron_density", "quasineutral"): (
        "Compute electron density from heavy-species charge balance.",
        "Initial and evolving heavy-species charge must remain nonnegative.",
        "argon_maxwellian",
        True,
    ),
    ("electron_density", "experimental.prescribed_profile"): (
        "Use an imposed electron-density time series.",
        "Requires a density CSV and explicit zone columns; the imposed density "
        "replaces charge-neutrality closure, so inspect charge_residual_m3.",
        "prescribed_density",
        True,
    ),
    ("gas_energy", "fixed"): (
        "Keep each zone's gas temperature fixed.",
        "Uses initial.gas_temperature_K; heavy-gas energy is not integrated.",
        "argon_maxwellian",
        True,
    ),
    ("gas_energy", "evolved"): (
        "Integrate heavy-gas internal energy and temperature.",
        "Provide species heat capacities and consistent inlet, wall, transport "
        "and reaction heat terms; relaxation frequencies are optional.",
        "gas_heating_flow",
        True,
    ),
    ("power", "prescribed_power"): (
        "Prescribe absorbed power and its electron/gas partition.",
        "electron_fraction + gas_fraction must equal one. Set absorbed_power_W "
        "in the model or run commands; this source alone does not determine E/N.",
        "argon_maxwellian",
        True,
    ),
    ("power", "dc_series"): (
        "Solve a DC series circuit and plasma electrical power.",
        "Provide geometry, ballast resistance, source voltage and electron "
        "mobility data; local-field coupling requires a stable circuit root.",
        "swarm_argon_dc",
        True,
    ),
    ("power", "external_table"): (
        "Replay externally computed electrical commands and diagnostics.",
        "Requires a time-series CSV; declare column mapping, clock and bounds. "
        "Voltage needs a stated gap and density to determine E/N. The linked "
        "case is a power-port template: replace the port model and supply its CSV.",
        "argon_maxwellian",
        False,
    ),
    ("power", "experimental.rf_envelope"): (
        "Use a calibrated RF power/voltage envelope.",
        "Choose control and its matching setpoint; impedance and field response "
        "are configured model assumptions, not a resolved RF circuit.",
        "rf_envelope_calibration",
        True,
    ),
    ("power", "experimental.ccp"): (
        "Evaluate the reduced CCP sheath and power model.",
        "Electrode gap, powered/grounded areas and dominant ion are required. "
        "Choose absorbed-power or voltage control and its matching setpoint.",
        "smoke",
        True,
    ),
    ("power", "experimental.icp"): (
        "Evaluate reduced inductive coupling and optional downstream deposition.",
        "Requires delivered power in the model or commands. A positive "
        "downstream fraction requires a downstream zone and energy_balance closure.",
        "smoke",
        True,
    ),
    ("wall_transport", "bohm"): (
        "Remove positive ions through a Bohm wall flux.",
        "Use boundary reactions to define products; h_factor may be explicit "
        "or automatic with a characteristic length.",
        "argon_maxwellian",
        True,
    ),
    ("wall_transport", "prescribed_frequency"): (
        "Apply a specified positive-ion loss frequency.",
        "frequency_s_inv is nonnegative; boundary products still require "
        "applicable boundary reactions. The example is a wall-configuration "
        "template; replace its Bohm block.",
        "argon_maxwellian",
        False,
    ),
    ("wall_transport", "ambipolar"): (
        "Estimate positive-ion wall loss from ambipolar diffusion.",
        "Requires diffusion_coefficient_m2_s and diffusion_length_m; the electron "
        "temperature reference must match the diffusion data.",
        "zdplaskin_example2",
        True,
    ),
    ("wall_transport", "off"): (
        "Disable positive-ion transport at a modeled surface.",
        "This does not disable surface chemistry or heavy-gas thermal relaxation.",
        "surface_inventory",
        True,
    ),
    ("surface_kinetics", "mass_action"): (
        "Integrate surface coverages from declared surface reactions.",
        "Requires surface species, site density, free-site definition and "
        "applicable rate models; site occupancy is constrained.",
        "surface_inventory",
        True,
    ),
    ("accumulators", "film"): (
        "Accumulate chemistry-declared film thickness.",
        "Enabled by presence under experimental; requires declared film "
        "contributions and a positive monolayer thickness.",
        "smoke",
        True,
    ),
    ("accumulators", "wall_inventory"): (
        "Accumulate particle inventories from surface reaction events.",
        "Provide initial stocks and signed particles-per-event yields; "
        "stocks must remain nonnegative.",
        "surface_inventory",
        True,
    ),
    ("transport", "edges"): (
        "Transfer gas species between zones through directed conductance.",
        "from_zone and to_zone must differ; transport carries particles and "
        "the enabled energy states between the specified volumes.",
        "benchmarks/analytic-transport",
        True,
    ),
    ("transport", "gas_inlets"): (
        "Feed species with specified flows and inlet temperature.",
        "flow_sccm is keyed by known gas species; commands may override flows. "
        "The stated inlet temperature determines incoming heavy-particle energy.",
        "gas_heating_flow",
        True,
    ),
    ("transport", "pumps"): (
        "Remove gas through a zone's volumetric pumping speed.",
        "speed_m3_s is nonnegative; pumping removes particles and the enabled "
        "carried energy states from the referenced zone.",
        "gas_heating_flow",
        True,
    ),
    ("reaction_rates", "electron_table"): (
        "Read an electron-reaction coefficient from the electron kinetics table.",
        "rate_id must match a supplied coefficient column; coefficient units "
        "must match the reaction's total order.",
        "swarm_argon_dc",
        True,
    ),
    ("reaction_rates", "electron_impact"): (
        "Integrate a declared electron-impact cross section.",
        "cross_section references energy-eV/cross-section-m2 input data. "
        "Declare branching yield and signed electron-energy transfer consistently.",
        "argon_maxwellian",
        True,
    ),
    ("reaction_rates", "arrhenius"): (
        "Evaluate a gas-temperature Arrhenius coefficient.",
        "A carries units determined by reaction order; activation_eV is an "
        "activation energy and does not itself declare reaction heat.",
        "zdplaskin_example2",
        True,
    ),
    ("reaction_rates", "constant"): (
        "Use a constant mass-action gas coefficient.",
        "value carries units determined by reaction order; overall_order, when "
        "given, checks the declared reaction rather than replacing its order.",
        "crane_two_reaction_argon",
        True,
    ),
    ("reaction_rates", "first_order"): (
        "Apply a first-order gas-species transformation.",
        "rate_s_inv is a frequency; the reaction must satisfy the first-order "
        "stoichiometric contract.",
        "benchmarks/analytic-decay",
        True,
    ),
    ("reaction_rates", "experimental.electron_temperature_power_law"): (
        "Use an empirical electron-temperature power-law coefficient.",
        "A is defined at reference_temperature_K; exponent and reaction-order "
        "units must match the source fit's validity range.",
        "zdplaskin_example2",
        True,
    ),
    ("reaction_rates", "tabulated_1d"): (
        "Interpolate a gas-reaction coefficient along one physical state axis.",
        "CSV columns are x,value; choose axis and bounds=error or hold. The "
        "linked case is a chemistry template, not a tabulated-rate example.",
        "benchmarks/analytic-decay",
        False,
    ),
    ("reaction_rates", "sticking"): (
        "Adsorb gas species with a sticking probability.",
        "value lies in [0,1]; optional coverage dependence uses modeled site "
        "fractions and declared surface stoichiometry.",
        "smoke",
        True,
    ),
    ("reaction_rates", "ion_assisted"): (
        "Evaluate an ion-assisted surface reaction yield.",
        "Provide yield and energy response; reference_energy_eV must exceed "
        "threshold_eV. The modeled incident ion must appear unchanged on both "
        "sides of the surface reaction; ion capture is not supported.",
        "smoke",
        True,
    ),
    ("reaction_rates", "desorption"): (
        "Release adsorbates through thermal desorption.",
        "frequency_s_inv is a frequency; activation_eV uses the modeled "
        "surface temperature.",
        "smoke",
        True,
    ),
    ("reaction_rates", "langmuir_hinshelwood"): (
        "React adsorbates through a surface mass-action coefficient.",
        "A_m2_s has units m^2/s; activation_eV uses surface temperature. "
        "Declare site-balanced surface stoichiometry.",
        "smoke",
        True,
    ),
}


def _model_types(annotation: Any) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
    else:
        for item in get_args(annotation):
            yield from _model_types(item)


def _description(category: str, kind: str) -> dict[str, Any]:
    purpose, applicability, example, exact = _DETAILS[category, kind]
    return {
        "kind": kind,
        "purpose": purpose,
        "applicability": applicability,
        "examples": [
            {
                "case": example,
                "relationship": "uses_model" if exact else "template",
                "file": f"examples/cases/{example}.yaml",
                "initialize": f"plasma-global init {example} --output MY_CASE",
            }
        ],
    }


def _schema_entry(
    category: str, kind: str, model: type[BaseModel], path: str
) -> dict[str, Any]:
    schema = model.model_json_schema()
    return {
        **_description(category, kind),
        "configuration_path": path,
        "required_inputs": schema.get("required", []),
        "fields": schema["properties"],
        "definitions": schema.get("$defs", {}),
    }


def _schema_models(
    annotation: Any, category: str, path: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model in _model_types(annotation):
        field = model.model_fields["kind"]
        for kind in get_args(field.annotation):
            result[kind] = _schema_entry(category, kind, model, path)
    return result


def _presence_models(
    owner: type[BaseModel],
    fields: Iterable[str],
    category: str,
    prefix: str,
    *,
    keyed: bool = False,
) -> dict[str, dict[str, Any]]:
    suffix = ".<id>" if keyed else ""
    return {
        name: _schema_entry(
            category,
            name,
            next(_model_types(owner.model_fields[name].annotation)),
            f"{prefix}.{name}{suffix}",
        )
        for name in fields
    }


def _rate_catalog() -> dict[str, dict[str, Any]]:
    return {
        kind: {
            **_description("reaction_rates", kind),
            "configuration_path": "chemistry.rate_models.<id>",
            "required_inputs": ["kind", *sorted(definition.required)],
            "fields": {
                "kind": {"const": kind},
                **{
                    name: {"required": name in definition.required}
                    for name in sorted(definition.required | definition.optional)
                },
            },
            "family": definition.family,
            "state_dependency": definition.dependencies,
        }
        for kind, definition in RATE_DEFINITIONS.items()
    }


def _catalog() -> dict[str, dict[str, dict[str, Any]]]:
    fields = ModelsDefinition.model_fields
    sources = {
        "electrons": (fields["electrons"].annotation, "models.electrons"),
        "electron_density": (
            fields["electron_density"].annotation,
            "models.electron_density",
        ),
        "gas_energy": (fields["gas_energy"].annotation, "models.gas_energy"),
        "power": (
            PowerDefinition.model_fields["model"].annotation,
            "reactor.power_ports.<id>.model",
        ),
        "wall_transport": (
            SurfaceDefinition.model_fields["wall_transport"].annotation,
            "reactor.surfaces.<id>.wall_transport",
        ),
        "surface_kinetics": (
            fields["surface_kinetics"].annotation,
            "models.surface_kinetics",
        ),
    }
    return {
        **{
            category: _schema_models(annotation, category, path)
            for category, (annotation, path) in sources.items()
        },
        "accumulators": _presence_models(
            ExperimentalConfig,
            ExperimentalConfig.model_fields,
            "accumulators",
            "experimental",
        ),
        "transport": _presence_models(
            ReactorDefinition,
            ("edges", "gas_inlets", "pumps"),
            "transport",
            "reactor",
            keyed=True,
        ),
        "reaction_rates": _rate_catalog(),
    }


def list_models(category: str | None = None, kind: str | None = None) -> dict[str, Any]:
    """Describe accepted models without reading case files or compiling physics.

    Optional filters retain the category/kind structure. Field contracts and
    defaults are generated from the same definitions that validate actual input.
    Example relationships distinguish a working use from a modification template.
    """
    catalog = _catalog()
    if category is None:
        if kind is not None:
            raise ValueError("a model kind requires a category")
        return catalog
    if category not in catalog:
        raise ValueError(
            f"unknown model category {category!r}; available: {sorted(catalog)}"
        )
    selected = catalog[category]
    if kind is None:
        return {category: selected}
    if kind not in selected:
        raise ValueError(
            f"unknown {category} model {kind!r}; available: {sorted(selected)}"
        )
    return {category: {kind: selected[kind]}}


__all__ = ["list_models"]
