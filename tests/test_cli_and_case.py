from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plasma_global.cli import main
from plasma_global.workflows.context import EEDF_REGISTRY, ELECTRICAL_REGISTRY, build_case, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'


def test_smoke_case_validates() -> None:
    loaded = load_case_from_yaml(SMOKE_CASE)
    assert loaded.run_config.case.name == 'smoke_swarm'
    assert not any(msg['level'] == 'ERROR' for msg in loaded.validation_messages)


def test_backend_catalog_exposes_descriptions() -> None:
    details = EEDF_REGISTRY.details()
    assert 'description' in details['boltzmann_2term']
    assert 'Boltzmann' in details['boltzmann_2term']['description']
    electrical = ELECTRICAL_REGISTRY.details()
    assert 'description' in electrical['dc_series_circuit']
    assert 'ballast-resistor' in electrical['dc_series_circuit']['description']


def test_state_layout_labels_are_stable() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    labels = built.system.state_labels()
    assert len(labels) == built.state_layout.size
    assert len(set(labels)) == len(labels)
    assert any(label.startswith('n[') for label in labels)
    assert any(label.startswith('We[') for label in labels)


def test_cli_validate_returns_success() -> None:
    assert main(['validate', str(SMOKE_CASE)]) == 0


def test_legacy_run_yaml_shape_is_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / 'run.yaml'
    legacy.write_text(
        yaml.safe_dump(
            {
                'project': {'name': 'legacy_case'},
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
        load_case_from_yaml(legacy)


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
