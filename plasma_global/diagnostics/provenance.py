from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_yaml_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return data if isinstance(data, dict) else {'value': data}


def _h5_attrs(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import h5py
    except ImportError:
        return {'warning': 'h5py unavailable; HDF5 attributes were not inspected'}
    attrs: dict[str, Any] = {}
    with h5py.File(path, 'r') as h5:
        for key, value in h5.attrs.items():
            attrs[str(key)] = _plain(value)
    return attrs


def _rate_table_provenance(run_config: Any, resolved_paths: Any) -> dict[str, Any] | None:
    table_cfg = getattr(getattr(run_config, 'swarm', None), 'table', None)
    raw_file = getattr(table_cfg, 'file', None)
    if not raw_file:
        return None
    path = Path(str(raw_file))
    if not path.is_absolute():
        chemistry_dir = getattr(resolved_paths, 'chemistry_dir', None)
        path = Path(chemistry_dir or getattr(resolved_paths, 'base_dir', '.')) / path
    path = path.resolve()
    sidecar_dir = path.with_suffix('')
    sidecar_metadata = _read_yaml_if_present(sidecar_dir / 'metadata.yaml')
    return {
        'file': str(path),
        'exists': path.exists(),
        'lookup': getattr(table_cfg, 'lookup', None),
        'electron_energy_mode': getattr(table_cfg, 'electron_energy_mode', None),
        'energy_relaxation_time_s': getattr(table_cfg, 'energy_relaxation_time_s', None),
        'hdf5_attributes': _h5_attrs(path),
        'sidecar_metadata': sidecar_metadata,
    }


def build_run_provenance(loaded: Any, built: Any) -> dict[str, Any]:
    run_config = loaded.run_config
    resolved = loaded.resolved_paths
    mechanism = loaded.mechanism
    table = _rate_table_provenance(run_config, resolved)
    cross_sections = {}
    for cs_id, cs in mechanism.cross_sections.items():
        cross_sections[cs_id] = {
            'kind': cs.kind,
            'target_species': cs.target_species,
            'collider_species': cs.collider_species,
            'threshold_eV': cs.threshold_eV,
            'energy_loss_eV': cs.energy_loss_eV,
            'source': cs.source,
            'file': cs.file,
            'format': cs.format,
            'metadata': cs.metadata,
            'has_tabulated_data': cs.has_tabulated_data(),
        }
    rate_models = {}
    for key, model in mechanism.rate_models.items():
        rate_models[key] = {
            k: v
            for k, v in model.items()
            if k in {'backend', 'cross_section_id', 'A', 'beta', 'Ea_eV', 'value', 'rate_s_inv', 'Tref_K', 'alpha', 'energy_loss_eV'}
        }
    return _plain(
        {
            'tool': 'plasma_global_run_provenance',
            'case': {
                'name': run_config.case.name,
                'description': run_config.case.description,
                'tags': list(run_config.case.tags),
                'schema_version': run_config.case.schema_version,
                'source_config': resolved.source_config,
            },
            'inputs': {
                'chamber_file': resolved.chamber_file,
                'recipe_file': resolved.recipe_file,
                'chemistry_manifest': resolved.chemistry_manifest,
                'chemistry_dir': resolved.chemistry_dir,
                'external_inputs': dict(resolved.external_inputs),
            },
            'model_selection': {
                'mode': run_config.physics.mode,
                'gas_model': run_config.physics.gas_model,
                'eedf_backend': run_config.physics.eedf_backend,
                'electrical_backend': run_config.physics.electrical_backend,
                'integrator': run_config.physics.integrator,
                'quasi_neutrality': run_config.physics.quasi_neutrality,
                'electron_density_closure': getattr(run_config.physics, 'electron_density_closure', 'quasi_neutral'),
                'electron_density_state': (
                    'prescribed_external_profile_not_state_variable'
                    if getattr(built.system, 'prescribed_electron_profile', None) is not None
                    else 'algebraic_from_ion_charge_balance'
                ),
                'prescribed_electron_profile': built.system.prescribed_electron_profile.provenance() if getattr(built.system, 'prescribed_electron_profile', None) is not None else None,
            },
            'swarm': {
                'model_name': getattr(getattr(run_config, 'swarm', None), 'model_name', None),
                'closure': getattr(getattr(run_config, 'swarm', None), 'closure', None),
                'rate_table': table,
            },
            'chemistry': {
                'species_count': len(mechanism.species),
                'gas_state_species_count': len(mechanism.gas_state_species),
                'gas_reaction_count': len(mechanism.gas_reactions),
                'surface_reaction_count': len(mechanism.surface_reactions),
                'cross_sections': cross_sections,
                'rate_models': rate_models,
            },
            'chamber': {
                'zone_count': len(loaded.chamber.zones),
                'surface_count': len(loaded.chamber.surfaces),
                'power_ports': {
                    port.port_id: {
                        'kind': port.kind,
                        'zone_id': port.zone_id,
                        'coupling_target': port.coupling_target,
                        'parameters': dict(port.parameters),
                    }
                    for port in loaded.chamber.power_ports
                },
            },
            'backends': {
                'eedf_backend_class': type(built.eedf_backend).__name__,
                'electrical_backend_class': type(built.electrical_backend).__name__,
                'integrator_class': type(built.integrator).__name__,
            },
            'validation_messages': loaded.validation_messages,
        }
    )
