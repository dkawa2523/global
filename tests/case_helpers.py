from __future__ import annotations

from pathlib import Path

import yaml

from plasma_global.config.loader import load_run_config, resolve_run_paths


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'
BASE_CASE = ROOT / 'examples' / 'configs' / 'base_case.yaml'
ARGON_LXCAT_CASE = ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml'
ZDPLASKIN_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'


def write_case_with_output_dir(tmp_path: Path, source_case: Path, *, name: str = 'case.yaml') -> Path:
    run_config = load_run_config(source_case)
    resolved = resolve_run_paths(run_config, source_case)
    data = run_config.to_dict()
    data['files']['chamber'] = resolved.chamber_file
    data['files']['recipe'] = resolved.recipe_file
    data['files']['chemistry']['manifest'] = resolved.chemistry_manifest
    data['files']['external_inputs'] = resolved.external_inputs
    data['files']['output_dir'] = str(tmp_path / 'out')
    case_path = tmp_path / name
    case_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding='utf-8',
    )
    return case_path


def example_chamber() -> dict:
    return yaml.safe_load((ROOT / 'examples' / 'configs' / 'chamber.yaml').read_text(encoding='utf-8'))


def write_smoke_case_with_chamber(tmp_path: Path, chamber: dict) -> Path:
    chamber_path = tmp_path / 'chamber.yaml'
    case_path = tmp_path / 'case.yaml'
    chamber_path.write_text(yaml.safe_dump(chamber, sort_keys=False), encoding='utf-8')
    case_path.write_text(
        yaml.safe_dump(
            {
                'include': str(BASE_CASE),
                'case': {'name': 'wall_loss_test'},
                'files': {
                    'chamber': str(chamber_path),
                    'recipe': str(ROOT / 'examples' / 'configs' / 'recipe_smoke.yaml'),
                    'chemistry': {'manifest': str(ROOT / 'examples' / 'chemistry' / 'chemistry_manifest.yaml')},
                    'output_dir': str(tmp_path / 'out'),
                },
                'outputs': {'plots': {'enabled': False}},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    return case_path
