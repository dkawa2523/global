from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.chemistry.validators import validate_mechanism
from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.models import ResolvedPaths, RunConfig
from plasma_global.config.validator import validate_loaded_inputs, validate_run_config
from plasma_global.numerics.state_layout import StateLayout, build_state_layout
from plasma_global.numerics.system import GlobalPlasmaSystem
from plasma_global.reactor.io import load_chamber_config, load_recipe_config
from plasma_global.workflows.registries import EEDF_REGISTRY, ELECTRICAL_REGISTRY, INTEGRATOR_REGISTRY

__all__ = [
    'LoadedCase',
    'BuiltCase',
    'load_case_from_yaml',
    'build_case',
]


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


def _build_backends(loaded: LoadedCase) -> tuple[Any, Any, Any]:
    run_config = loaded.run_config
    eedf = EEDF_REGISTRY.build(run_config.physics.eedf_backend)
    electrical = ELECTRICAL_REGISTRY.build(run_config.physics.electrical_backend)
    integrator = INTEGRATOR_REGISTRY.build(run_config.physics.integrator, run_config=run_config)
    return eedf, electrical, integrator


def _prepare_backends(loaded: LoadedCase, eedf: Any, electrical: Any) -> None:
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


def build_case(loaded: LoadedCase) -> BuiltCase:
    eedf, electrical, integrator = _build_backends(loaded)
    _prepare_backends(loaded, eedf, electrical)
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
