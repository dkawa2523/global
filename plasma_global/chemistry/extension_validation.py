from __future__ import annotations

from typing import Any

from plasma_global.chemistry.models import MechanismBundle

STATE_SCOPES = {'zone', 'surface'}
PROCESS_KINDS = {'source', 'relaxation', 'ion_flux_source'}


def validate_state_variables(mechanism: MechanismBundle, report: Any, chamber: Any | None = None) -> None:
    seen: set[str] = set()
    zone_ids = set(getattr(chamber, 'zone_by_id', {}) or {})
    surface_ids = set(getattr(chamber, 'surface_by_id', {}) or {})
    for spec in mechanism.state_variables:
        if not spec.state_id:
            report.add('ERROR', 'STATE_VARIABLE_ID_MISSING', 'State variable id must not be empty')
            continue
        if spec.state_id in seen:
            report.add('ERROR', 'STATE_VARIABLE_DUPLICATE', f'Duplicate state variable id {spec.state_id}', spec.state_id)
        seen.add(spec.state_id)
        if spec.scope not in STATE_SCOPES:
            report.add('ERROR', 'STATE_VARIABLE_SCOPE_UNSUPPORTED', f'State variable {spec.state_id} scope must be one of {sorted(STATE_SCOPES)}', spec.state_id)
        if spec.lower_bound is not None and spec.initial < spec.lower_bound:
            report.add('ERROR', 'STATE_VARIABLE_INITIAL_BELOW_BOUND', f'State variable {spec.state_id} initial is below lower_bound', spec.state_id)
        if spec.scale <= 0.0:
            report.add('ERROR', 'STATE_VARIABLE_SCALE_INVALID', f'State variable {spec.state_id} scale must be positive', spec.state_id)
        if spec.scope == 'zone' and spec.surfaces:
            report.add('ERROR', 'STATE_VARIABLE_SCOPE_FILTER_MISMATCH', f'Zone state variable {spec.state_id} cannot define surfaces', spec.state_id)
        if spec.scope == 'surface' and spec.zones:
            report.add('ERROR', 'STATE_VARIABLE_SCOPE_FILTER_MISMATCH', f'Surface state variable {spec.state_id} cannot define zones', spec.state_id)
        if chamber is not None:
            for zone_id in spec.zones:
                if zone_id not in zone_ids:
                    report.add('ERROR', 'STATE_VARIABLE_UNKNOWN_ZONE', f'State variable {spec.state_id} references unknown zone {zone_id}', spec.state_id)
            for surface_id in spec.surfaces:
                if surface_id not in surface_ids:
                    report.add('ERROR', 'STATE_VARIABLE_UNKNOWN_SURFACE', f'State variable {spec.state_id} references unknown surface {surface_id}', spec.state_id)


def validate_processes(mechanism: MechanismBundle, report: Any, chamber: Any | None = None) -> None:
    zone_ids = set(getattr(chamber, 'zone_by_id', {}) or {})
    surface_ids = set(getattr(chamber, 'surface_by_id', {}) or {})
    for process in mechanism.processes:
        if not process.process_id:
            report.add('ERROR', 'PROCESS_ID_MISSING', 'Process id must not be empty')
            continue
        if process.kind not in PROCESS_KINDS:
            report.add('ERROR', 'PROCESS_KIND_UNSUPPORTED', f'Process {process.process_id} kind must be one of {sorted(PROCESS_KINDS)}', process.process_id)
        target = mechanism.state_variable_by_id.get(process.target)
        if target is None:
            report.add('ERROR', 'PROCESS_TARGET_UNKNOWN', f'Process {process.process_id} targets unknown state variable {process.target}', process.process_id)
            continue
        if process.zones and target.scope != 'zone':
            report.add('ERROR', 'PROCESS_SCOPE_FILTER_MISMATCH', f'Process {process.process_id} zones filter requires a zone target', process.process_id)
        if process.surfaces and target.scope != 'surface':
            report.add('ERROR', 'PROCESS_SCOPE_FILTER_MISMATCH', f'Process {process.process_id} surfaces filter requires a surface target', process.process_id)
        _validate_process_target_owners(process, target, report, zone_ids, surface_ids)
        if process.kind == 'source':
            _validate_process_number(report, process.process_id, process.parameters, 'value')
        if process.kind == 'relaxation':
            _validate_process_number(report, process.process_id, process.parameters, 'tau_s')
            try:
                if float(process.parameters.get('tau_s', 0.0)) <= 0.0:
                    report.add('ERROR', 'PROCESS_TAU_INVALID', f'Process {process.process_id} tau_s must be positive', process.process_id)
            except (TypeError, ValueError):
                pass
        if process.kind == 'ion_flux_source':
            _validate_process_number(report, process.process_id, process.parameters, 'coefficient')
            if target.scope != 'surface':
                report.add('ERROR', 'PROCESS_TARGET_SCOPE_INVALID', f'ion_flux_source process {process.process_id} requires a surface target', process.process_id)


def _validate_process_number(report: Any, process_id: str, params: dict[str, Any], key: str) -> None:
    try:
        float(params[key])
    except KeyError:
        report.add('ERROR', 'PROCESS_PARAMETER_MISSING', f'Process {process_id} requires {key}', process_id)
    except (TypeError, ValueError):
        report.add('ERROR', 'PROCESS_PARAMETER_INVALID', f'Process {process_id} {key} must be numeric', process_id)


def _validate_process_target_owners(
    process: Any,
    target: Any,
    report: Any,
    zone_ids: set[str],
    surface_ids: set[str],
) -> None:
    if target.scope == 'zone' and process.zones and target.zones:
        missing = sorted(set(process.zones) - set(target.zones))
        if missing:
            report.add('ERROR', 'PROCESS_TARGET_OWNER_UNAVAILABLE', f'Process {process.process_id} references zones outside target {target.state_id}: {missing}', process.process_id)
    if target.scope == 'surface' and process.surfaces and target.surfaces:
        missing = sorted(set(process.surfaces) - set(target.surfaces))
        if missing:
            report.add('ERROR', 'PROCESS_TARGET_OWNER_UNAVAILABLE', f'Process {process.process_id} references surfaces outside target {target.state_id}: {missing}', process.process_id)
    if target.scope == 'zone' and process.zones and zone_ids:
        unknown = sorted(set(process.zones) - zone_ids)
        if unknown:
            report.add('ERROR', 'PROCESS_UNKNOWN_ZONE', f'Process {process.process_id} references unknown zone {unknown[0]}', process.process_id)
    if target.scope == 'surface' and process.surfaces and surface_ids:
        unknown = sorted(set(process.surfaces) - surface_ids)
        if unknown:
            report.add('ERROR', 'PROCESS_UNKNOWN_SURFACE', f'Process {process.process_id} references unknown surface {unknown[0]}', process.process_id)
