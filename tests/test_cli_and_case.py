from __future__ import annotations

from pathlib import Path

from plasma_global.cli import main
from plasma_global.workflows.context import EEDF_REGISTRY, ELECTRICAL_REGISTRY, build_case, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'


def test_smoke_case_validates() -> None:
    loaded = load_case_from_yaml(SMOKE_CASE)
    assert loaded.run_config.case.name == 'smoke_swarm'
    assert not any(msg['level'] == 'ERROR' for msg in loaded.validation_messages)
    assert any(msg['code'] == 'SURFACE_MODELS_METADATA_ONLY' for msg in loaded.validation_messages)
    metadata_messages = [msg['message'] for msg in loaded.validation_messages if msg['code'] == 'SURFACE_MODELS_METADATA_ONLY']
    assert all('ion_loss' not in msg for msg in metadata_messages)


def test_backend_catalog_exposes_maturity() -> None:
    details = EEDF_REGISTRY.details()
    assert details['boltzmann_2term']['maturity'] == 'experimental'
    assert details['maxwell']['maturity'] == 'stable-debug'
    assert details['boltzmann_2term']['assumptions']
    electrical = ELECTRICAL_REGISTRY.details()
    assert electrical['dc_series_circuit']['maturity'] == 'experimental'
    assert electrical['dc_series_circuit']['assumptions']


def test_state_layout_labels_are_stable() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    labels = built.system.state_labels()
    assert len(labels) == built.state_layout.size
    assert len(set(labels)) == len(labels)
    assert any(label.startswith('n[') for label in labels)
    assert any(label.startswith('We[') for label in labels)


def test_cli_validate_returns_success() -> None:
    assert main(['validate', str(SMOKE_CASE)]) == 0
