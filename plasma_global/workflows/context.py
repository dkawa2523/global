from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.chemistry.validators import validate_mechanism
from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.models import ResolvedPaths, RunConfig
from plasma_global.config.validator import validate_loaded_inputs, validate_run_config
from plasma_global.core.registry import Registry
from plasma_global.eedf.boltzmann_2term import Boltzmann2TermBackend
from plasma_global.eedf.maxwell import MaxwellEEDFBackend
from plasma_global.eedf.swarm_backend import SwarmEEDFBackend
from plasma_global.eedf.table import TableEEDFBackend
from plasma_global.electrical.ccp import CCPBackend
from plasma_global.electrical.dc_series import DCSeriesCircuitBackend
from plasma_global.electrical.direct_power import DirectPowerBackend
from plasma_global.electrical.external_table import ExternalCircuitTableBackend
from plasma_global.electrical.icp import ICPBackend
from plasma_global.electrical.rf_envelope import RFEnvelopeBackend
from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator
from plasma_global.numerics.state_layout import StateLayout, StateSlice
from plasma_global.numerics.system import GlobalPlasmaSystem
from plasma_global.reactor.io import load_chamber_config, load_recipe_config


EEDF_REGISTRY = Registry()
EEDF_REGISTRY.register(
    'swarm',
    lambda **kwargs: SwarmEEDFBackend(),
    description='Swarm wrapper with replaceable swarm model backend.',
)
EEDF_REGISTRY.register(
    'maxwell',
    lambda **kwargs: MaxwellEEDFBackend(),
    description='Analytic Maxwellian closure.',
)
EEDF_REGISTRY.register(
    'boltzmann_2term',
    lambda **kwargs: Boltzmann2TermBackend(),
    description='Internal two-term Boltzmann swarm backend.',
)
EEDF_REGISTRY.register(
    'rate_table',
    lambda **kwargs: TableEEDFBackend(),
    description='Interpolated external rate-table backend.',
)

ELECTRICAL_REGISTRY = Registry()
ELECTRICAL_REGISTRY.register(
    'direct_power',
    lambda **kwargs: DirectPowerBackend(),
    description='Direct absorbed-power prescription.',
)
ELECTRICAL_REGISTRY.register(
    'dc_series_circuit',
    lambda **kwargs: DCSeriesCircuitBackend(),
    description='Reduced voltage-source plus ballast-resistor circuit for DC and pulsed DC cases.',
)
ELECTRICAL_REGISTRY.register(
    'external_circuit_table',
    lambda **kwargs: ExternalCircuitTableBackend(),
    description='One-way coupling from measured or SPICE-generated circuit waveform CSV data.',
)
ELECTRICAL_REGISTRY.register(
    'rf_envelope',
    lambda **kwargs: RFEnvelopeBackend(),
    description='Cycle-averaged HF/LF RF power and bias envelope model.',
)
ELECTRICAL_REGISTRY.register(
    'ccp',
    lambda **kwargs: CCPBackend(),
    description='Reduced CCP / bias backend.',
)
ELECTRICAL_REGISTRY.register(
    'icp',
    lambda **kwargs: ICPBackend(),
    description='Reduced ICP source-coupling backend.',
)

INTEGRATOR_REGISTRY = Registry()
INTEGRATOR_REGISTRY.register(
    'scipy_bdf',
    lambda *, run_config: SciPyBDFIntegrator(
        rtol=run_config.numerics.rtol,
        atol=run_config.numerics.atol,
        first_step=run_config.numerics.first_step,
        max_step=run_config.numerics.max_step,
    ),
    description='SciPy solve_ivp(method="BDF") backend.',
)


@dataclass
class LoadedCase:
    run_config: RunConfig
    resolved_paths: ResolvedPaths
    chamber: Any
    recipe: Any
    mechanism: Any
    validation_messages: list[dict[str, Any]]


@dataclass
class BuiltCase:
    loaded: LoadedCase
    eedf_backend: Any
    electrical_backend: Any
    integrator: Any
    state_layout: StateLayout
    system: GlobalPlasmaSystem


def build_state_layout(mechanism: Any, chamber: Any, run_config: Any) -> StateLayout:
    gas_species = [s for s in mechanism.gas_state_species]
    zone_ids = [z.zone_id for z in chamber.zones]
    start = 0
    slices: dict[str, StateSlice] = {}
    gas_index: dict[str, dict[str, int]] = {}
    surface_index: dict[str, dict[str, int]] = {}
    inventory_index: dict[str, dict[str, int]] = {}
    film_index: dict[str, int] = {}
    electron_energy_index: dict[str, int] = {}
    gas_temperature_index: dict[str, int] = {}

    slices['gas_densities'] = StateSlice('gas_densities', start, start + len(gas_species) * len(zone_ids))
    for z_idx, zone_id in enumerate(zone_ids):
        gas_index[zone_id] = {}
        for s_idx, sp in enumerate(gas_species):
            gas_index[zone_id][sp.canonical_id] = start + z_idx * len(gas_species) + s_idx
    start = slices['gas_densities'].stop

    slices['electron_energy'] = StateSlice('electron_energy', start, start + len(zone_ids))
    for i, zone_id in enumerate(zone_ids):
        electron_energy_index[zone_id] = start + i
    start = slices['electron_energy'].stop

    if run_config.model.enable_gas_temperature:
        slices['gas_temperature'] = StateSlice('gas_temperature', start, start + len(zone_ids))
        for i, zone_id in enumerate(zone_ids):
            gas_temperature_index[zone_id] = start + i
        start = slices['gas_temperature'].stop

    if run_config.model.enable_surface_coverages:
        surf_start = start
        for surface in chamber.surfaces:
            surface_index[surface.surface_id] = {}
            for sp in mechanism.surface_species:
                if surface.surface_id in sp.surfaces:
                    surface_index[surface.surface_id][sp.canonical_id] = start
                    start += 1
        slices['surface_coverages'] = StateSlice('surface_coverages', surf_start, start)

    if run_config.model.enable_wall_inventory:
        inv_start = start
        for surface in chamber.surfaces:
            inventory_index[surface.surface_id] = {}
            for key in surface.initial_inventory.keys():
                inventory_index[surface.surface_id][key] = start
                start += 1
        slices['wall_inventory'] = StateSlice('wall_inventory', inv_start, start)

    film_start = start
    for surface in chamber.surfaces:
        film_index[surface.surface_id] = start
        start += 1
    slices['film_thickness'] = StateSlice('film_thickness', film_start, start)

    return StateLayout(
        slices=slices,
        size=start,
        gas_index=gas_index,
        electron_energy_index=electron_energy_index,
        gas_temperature_index=gas_temperature_index,
        surface_index=surface_index,
        inventory_index=inventory_index,
        film_index=film_index,
        gas_species_ids=[sp.canonical_id for sp in gas_species],
        zone_ids=zone_ids,
    )


def load_case_from_yaml(run_yaml_path: str | Path) -> LoadedCase:
    source = Path(run_yaml_path).resolve()
    run_config = load_run_config(source)
    resolved = resolve_run_paths(run_config, source)
    setattr(run_config, '_resolved_paths', resolved)

    cfg_report = validate_run_config(run_config, resolved)
    if cfg_report.has_errors:
        raise ValueError(
            'Configuration validation failed:\n' +
            '\n'.join(f'[{m.level}] {m.code}: {m.message}' for m in cfg_report.messages)
        )

    chamber = load_chamber_config(resolved.chamber_file)
    recipe = load_recipe_config(resolved.recipe_file)
    input_report = validate_loaded_inputs(run_config, chamber, recipe)
    if input_report.has_errors:
        raise ValueError(
            'Loaded input validation failed:\n' +
            '\n'.join(f'[{m.level}] {m.code}: {m.message}' for m in input_report.messages)
        )
    chemistry_source = resolved.chemistry_manifest or resolved.chemistry_dir
    mechanism = load_mechanism_bundle(chemistry_source)

    mech_report = validate_mechanism(mechanism)
    if mech_report.has_errors:
        raise ValueError(
            'Mechanism validation failed:\n' +
            '\n'.join(f'[{m.level}] {m.code}: {m.message} ({m.entity_id})' for m in mech_report.messages)
        )

    messages = [m.__dict__ for m in cfg_report.messages] + [m.__dict__ for m in input_report.messages] + [m.__dict__ for m in mech_report.messages]
    return LoadedCase(
        run_config=run_config,
        resolved_paths=resolved,
        chamber=chamber,
        recipe=recipe,
        mechanism=mechanism,
        validation_messages=messages,
    )


def build_case(loaded: LoadedCase) -> BuiltCase:
    eedf = EEDF_REGISTRY.build(loaded.run_config.model.eedf_backend)
    electrical = ELECTRICAL_REGISTRY.build(loaded.run_config.model.electrical_backend)
    integrator = INTEGRATOR_REGISTRY.build(loaded.run_config.model.integrator, run_config=loaded.run_config)

    eedf.prepare(mechanism=loaded.mechanism, chamber=loaded.chamber, run_config=loaded.run_config)
    electrical.prepare(chamber=loaded.chamber, recipe=loaded.recipe, run_config=loaded.run_config)

    layout = build_state_layout(loaded.mechanism, loaded.chamber, loaded.run_config)
    system = GlobalPlasmaSystem(
        mechanism=loaded.mechanism,
        chamber=loaded.chamber,
        recipe=loaded.recipe,
        run_config=loaded.run_config,
        eedf_backend=eedf,
        electrical_backend=electrical,
        state_layout=layout,
    )
    return BuiltCase(
        loaded=loaded,
        eedf_backend=eedf,
        electrical_backend=electrical,
        integrator=integrator,
        state_layout=layout,
        system=system,
    )
