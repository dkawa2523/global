from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plasma_global.cli import main
from plasma_global import build_case, load_case_from_yaml
from plasma_global.eedf.registry import EEDF_REGISTRY
from plasma_global.electrical.registry import ELECTRICAL_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'


def test_smoke_case_validates() -> None:
    loaded = load_case_from_yaml(SMOKE_CASE)
    assert loaded.run_config.case.name == 'smoke_maxwell'
    assert not any(msg['level'] == 'ERROR' for msg in loaded.validation_messages)


def test_backend_registries_expose_descriptions() -> None:
    details = EEDF_REGISTRY.details()
    assert set(details) == {'maxwell', 'swarm'}
    assert 'description' in details['swarm']
    assert 'swarm.model_name' in details['swarm']['description']
    assert details['swarm']['maturity'] == 'configuration_dependent'
    assert 'Prepared HDF5' in details['swarm']['intended_use']
    assert 'approximate' in details['swarm']['caveat']
    electrical = ELECTRICAL_REGISTRY.details()
    assert 'description' in electrical['dc_series_circuit']
    assert 'ballast-resistor' in electrical['dc_series_circuit']['description']
    assert electrical['external_circuit_table']['maturity'] == 'data_driven'


def test_state_layout_labels_are_stable() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    labels = built.system.state_labels()
    assert len(labels) == built.state_layout.size
    assert len(set(labels)) == len(labels)
    assert any(label.startswith('n[') for label in labels)
    assert any(label.startswith('We[') for label in labels)
    assert 'film_thickness' in built.state_layout.slices
    assert any(label.startswith('film[') for label in labels)
    assert built.system.surface_core.enabled is True
    assert built.system.surface_core.surface_reactions


def test_cli_validate_returns_success() -> None:
    assert main(['validate', str(SMOKE_CASE)]) == 0


def test_schema_v1_run_yaml_shape_is_rejected(tmp_path: Path) -> None:
    case_path = tmp_path / 'run.yaml'
    case_path.write_text(
        yaml.safe_dump(
            {
                'project': {'name': 'schema_v1_case'},
                'paths': {
                    'chamber_file': 'chamber.yaml',
                    'recipe_file': 'recipe.yaml',
                    'output_dir': 'out',
                },
                'model': {'eedf_backend': 'maxwell'},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='Unsupported configuration schema'):
        load_case_from_yaml(case_path)


def test_schema_version_one_is_validation_error(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CASE.read_text(encoding='utf-8'))
    raw['include'] = str((SMOKE_CASE.parent / raw['include']).resolve())
    raw.setdefault('case', {})['schema_version'] = 1
    case_path = tmp_path / 'case_schema_v1.yaml'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')

    with pytest.raises(ValueError, match='UNSUPPORTED_SCHEMA_VERSION'):
        load_case_from_yaml(case_path)


def test_chemistry_directory_config_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CASE.read_text(encoding='utf-8'))
    raw['include'] = str((SMOKE_CASE.parent / raw['include']).resolve())
    raw['files']['chemistry'] = {'directory': str(ROOT / 'examples' / 'chemistry')}
    case_path = tmp_path / 'case_chemistry_directory.yaml'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')

    with pytest.raises(ValueError, match='Unsupported files.chemistry keys: directory'):
        load_case_from_yaml(case_path)


def test_unknown_outputs_key_is_rejected_at_load_time(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CASE.read_text(encoding='utf-8'))
    raw['include'] = str((SMOKE_CASE.parent / raw['include']).resolve())
    raw.setdefault('outputs', {})['debug_bundle'] = True
    case_path = tmp_path / 'case_unknown_outputs.yaml'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')

    with pytest.raises(ValueError, match='Unsupported outputs keys: debug_bundle'):
        load_case_from_yaml(case_path)


def test_unknown_swarm_key_is_rejected_at_load_time(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CASE.read_text(encoding='utf-8'))
    raw['include'] = str((SMOKE_CASE.parent / raw['include']).resolve())
    raw.setdefault('swarm', {})['debug_transport_dump'] = True
    case_path = tmp_path / 'case_unknown_swarm.yaml'
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')

    with pytest.raises(ValueError, match='Unsupported swarm keys: debug_transport_dump'):
        load_case_from_yaml(case_path)
