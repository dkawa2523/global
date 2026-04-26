from __future__ import annotations

from pathlib import Path

from plasma_global.workflows.context import load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]
ARGON_CASE = ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml'


def test_argon_lxcat_case_loads_public_cross_section_bundle() -> None:
    loaded = load_case_from_yaml(ARGON_CASE)

    assert not any(msg['level'] == 'ERROR' for msg in loaded.validation_messages)
    assert loaded.resolved_paths.chamber_file.endswith('chamber_argon_icp.yaml')
    assert loaded.run_config.model.enable_gas_temperature is False

    cross_sections = loaded.mechanism.cross_sections
    assert 'xs_ar_momentum_zenodo' in cross_sections
    assert 'xs_ar_ionization_zenodo' in cross_sections
    assert sum(1 for cs in cross_sections.values() if cs.kind == 'excitation') == 30
    assert cross_sections['xs_ar_ionization_zenodo'].metadata['source_doi'] == '10.5281/zenodo.8192503'
