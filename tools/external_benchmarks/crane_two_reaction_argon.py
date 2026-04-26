from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.runner import run_from_yaml

DEFAULT_CASE = ROOT / 'examples' / 'configs' / 'case_crane_two_reaction_argon.yaml'
DEFAULT_REFERENCE = ROOT / 'examples' / 'external' / 'crane_two_reaction_argon_reference.yaml'


def _ratio(local: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return local / reference


def _relative_error(local: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return abs(local - reference) / abs(reference)


def _state_map(result: dict[str, Any]) -> dict[str, float]:
    labels = result['system'].state_labels()
    y_final = np.asarray(result['solution'].y[:, -1], dtype=float)
    values = {label: float(y_final[idx]) for idx, label in enumerate(labels)}
    ar_plus = values['n[plasma,Ar_plus]']
    values['electron_density_m3'] = ar_plus
    return values


def build_report(case_path: Path, reference_path: Path) -> dict[str, Any]:
    result = run_from_yaml(case_path)
    local_state = _state_map(result)
    reference = yaml.safe_load(reference_path.read_text(encoding='utf-8'))
    ref_final = reference['converted_output_final']
    local = {
        'output_dir': result['output_dir'],
        'final_time_s': float(result['solution'].t[-1]),
        'final_Ar_density_m3': local_state['n[plasma,Ar]'],
        'final_Ar_plus_density_m3': local_state['n[plasma,Ar_plus]'],
        'final_electron_density_m3': local_state['electron_density_m3'],
        'summary_final_electron_density_m3': float(result['summary']['final_electron_density_m3']),
    }
    ref = {
        'final_time_s': float(reference['committed_output_final']['time_s']),
        'final_Ar_density_m3': float(ref_final['Ar_density_m3']),
        'final_Ar_plus_density_m3': float(ref_final['Ar_plus_density_m3']),
        'final_electron_density_m3': float(ref_final['electron_density_m3']),
    }
    comparison = {}
    for key in ['final_Ar_density_m3', 'final_Ar_plus_density_m3', 'final_electron_density_m3']:
        comparison[f'{key}_ratio_local_over_crane'] = _ratio(local[key], ref[key])
        comparison[f'{key}_relative_error'] = _relative_error(local[key], ref[key])
    return {
        'source_reference': reference['source'],
        'case': {
            'case_file': str(case_path),
            'scope': 'CRANE TwoReactionArgon scalar ODE chemistry benchmark',
            'unit_policy': 'CRANE cm-based input and output converted to SI before running this code.',
        },
        'conditions': reference['converted_conditions'],
        'local_result': local,
        'crane_reference': ref,
        'comparison': comparison,
        'limitations': [
            'The public CRANE tutorial uses prescribed rate coefficients and does not solve electron energy.',
            'This comparison therefore checks reaction-network assembly, unit conversion, initial densities, and ODE integration, not Boltzmann kinetics or circuit physics.',
            'The local electron-energy state is present because it is part of this solver architecture, but it is not used by the constant-rate CRANE benchmark chemistry.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='External-tool comparison for the CRANE TwoReactionArgon reproduction.')
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--write', type=Path, default=None, help='Output YAML path; defaults to <case output_dir>/comparison_crane_two_reaction_argon.yaml')
    args = parser.parse_args()
    report = build_report(args.case.resolve(), args.reference.resolve())
    output_dir = Path(report['local_result']['output_dir'])
    out_path = args.write or (output_dir / 'comparison_crane_two_reaction_argon.yaml')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
    print(out_path.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
