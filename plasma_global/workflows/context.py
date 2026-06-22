from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.chemistry.validators import validate_mechanism
from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.models import ResolvedPaths, RunConfig
from plasma_global.config.validator import validate_loaded_inputs, validate_run_config
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


@dataclass(frozen=True)
class BackendSpec:
    builder: Callable[..., Any]
    description: str


@dataclass(frozen=True)
class BackendCatalog:
    entries: dict[str, BackendSpec]

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self.entries:
            raise KeyError(f"Unknown backend: {name}. Available: {self.names()}")
        return self.entries[name].builder(**kwargs)

    def names(self) -> list[str]:
        return sorted(self.entries)

    def details(self) -> dict[str, dict[str, str]]:
        return {
            name: {'description': spec.description}
            for name, spec in sorted(self.entries.items())
        }


EEDF_REGISTRY = BackendCatalog(
    {
        'swarm': BackendSpec(lambda **kwargs: SwarmEEDFBackend(), 'Swarm wrapper with replaceable swarm model backend.'),
        'maxwell': BackendSpec(lambda **kwargs: MaxwellEEDFBackend(), 'Analytic Maxwellian closure.'),
        'boltzmann_2term': BackendSpec(lambda **kwargs: Boltzmann2TermBackend(), 'Internal two-term Boltzmann swarm backend.'),
        'rate_table': BackendSpec(lambda **kwargs: TableEEDFBackend(), 'Interpolated external rate-table backend.'),
    }
)

ELECTRICAL_REGISTRY = BackendCatalog(
    {
        'direct_power': BackendSpec(lambda **kwargs: DirectPowerBackend(), 'Direct absorbed-power prescription.'),
        'dc_series_circuit': BackendSpec(
            lambda **kwargs: DCSeriesCircuitBackend(),
            'Reduced voltage-source plus ballast-resistor circuit for DC and pulsed DC cases.',
        ),
        'external_circuit_table': BackendSpec(
            lambda **kwargs: ExternalCircuitTableBackend(),
            'One-way coupling from measured or SPICE-generated circuit waveform CSV data.',
        ),
        'rf_envelope': BackendSpec(lambda **kwargs: RFEnvelopeBackend(), 'Cycle-averaged HF/LF RF power and bias envelope model.'),
        'ccp': BackendSpec(lambda **kwargs: CCPBackend(), 'Reduced CCP / bias backend.'),
        'icp': BackendSpec(lambda **kwargs: ICPBackend(), 'Reduced ICP source-coupling backend.'),
    }
)

INTEGRATOR_REGISTRY = BackendCatalog(
    {
        'scipy_bdf': BackendSpec(
            lambda *, run_config: SciPyBDFIntegrator(
                rtol=run_config.numerics.rtol,
                atol=run_config.numerics.atol,
                first_step=run_config.numerics.first_step,
                max_step=run_config.numerics.max_step,
            ),
            'SciPy solve_ivp(method="BDF") backend.',
        )
    }
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

    if run_config.physics.enable_gas_temperature:
        slices['gas_temperature'] = StateSlice('gas_temperature', start, start + len(zone_ids))
        for i, zone_id in enumerate(zone_ids):
            gas_temperature_index[zone_id] = start + i
        start = slices['gas_temperature'].stop

    if run_config.physics.enable_surface_coverages:
        surf_start = start
        for surface in chamber.surfaces:
            surface_index[surface.surface_id] = {}
            for sp in mechanism.surface_species:
                if surface.surface_id in sp.surfaces:
                    surface_index[surface.surface_id][sp.canonical_id] = start
                    start += 1
        slices['surface_coverages'] = StateSlice('surface_coverages', surf_start, start)

    if run_config.physics.enable_wall_inventory:
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

    cfg_report = validate_run_config(run_config, resolved)
    if cfg_report.has_errors:
        raise ValueError(
            'Configuration validation failed:\n' +
            '\n'.join(f'[{m.level}] {m.code}: {m.message}' for m in cfg_report.messages)
        )

    chamber = load_chamber_config(resolved.chamber_file)
    recipe = load_recipe_config(resolved.recipe_file)
    input_report = validate_loaded_inputs(run_config, resolved, chamber, recipe)
    if input_report.has_errors:
        raise ValueError(
            'Loaded input validation failed:\n' +
            '\n'.join(f'[{m.level}] {m.code}: {m.message}' for m in input_report.messages)
        )
    mechanism = load_mechanism_bundle(resolved.chemistry_manifest)

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
    eedf = EEDF_REGISTRY.build(loaded.run_config.physics.eedf_backend)
    electrical = ELECTRICAL_REGISTRY.build(loaded.run_config.physics.electrical_backend)
    integrator = INTEGRATOR_REGISTRY.build(loaded.run_config.physics.integrator, run_config=loaded.run_config)

    eedf.prepare(
        mechanism=loaded.mechanism,
        chamber=loaded.chamber,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
    )
    electrical.prepare(
        chamber=loaded.chamber,
        recipe=loaded.recipe,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
    )

    layout = build_state_layout(loaded.mechanism, loaded.chamber, loaded.run_config)
    system = GlobalPlasmaSystem(
        mechanism=loaded.mechanism,
        chamber=loaded.chamber,
        recipe=loaded.recipe,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
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
