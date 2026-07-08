"""Load and validate case inputs without constructing solver runtime state."""

from __future__ import annotations

from pathlib import Path

from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.chemistry.validators import validate_mechanism
from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.validator import validate_loaded_inputs, validate_run_config
from plasma_global.reactor.io import load_chamber_config, load_recipe_config
from plasma_global.workflows.types import LoadedCase


def _format_validation_messages(title: str, messages: list) -> ValueError:
    return ValueError(
        title + ':\n' +
        '\n'.join(f'[{m.level}] {m.code}: {m.message}' for m in messages)
    )


def _format_mechanism_messages(messages: list) -> ValueError:
    return ValueError(
        'Mechanism validation failed:\n' +
        '\n'.join(f'[{m.level}] {m.code}: {m.message} ({m.entity_id})' for m in messages)
    )


def load_case_from_yaml(run_yaml_path: str | Path) -> LoadedCase:
    source = Path(run_yaml_path).resolve()
    run_config = load_run_config(source)
    resolved = resolve_run_paths(run_config, source)

    cfg_report = validate_run_config(run_config, resolved)
    if cfg_report.has_errors:
        raise _format_validation_messages('Configuration validation failed', cfg_report.messages)

    chamber = load_chamber_config(resolved.chamber_file)
    recipe = load_recipe_config(resolved.recipe_file)
    input_report = validate_loaded_inputs(run_config, resolved, chamber, recipe)
    if input_report.has_errors:
        raise _format_validation_messages('Loaded input validation failed', input_report.messages)

    mechanism = load_mechanism_bundle(resolved.chemistry_manifest)
    mech_report = validate_mechanism(mechanism, chamber)
    if mech_report.has_errors:
        raise _format_mechanism_messages(mech_report.messages)

    messages = [
        m.__dict__
        for report in (cfg_report, input_report, mech_report)
        for m in report.messages
    ]
    return LoadedCase(
        run_config=run_config,
        resolved_paths=resolved,
        chamber=chamber,
        recipe=recipe,
        mechanism=mechanism,
        validation_messages=messages,
    )


__all__ = ['load_case_from_yaml']
