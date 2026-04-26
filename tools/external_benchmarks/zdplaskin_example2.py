from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external_benchmarks.zdplaskin_eovern import build_eovern_report

DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate'
DEFAULT_REFERENCE = ROOT / 'examples' / 'external' / 'zdplaskin_example2_summary.yaml'


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError('h5py is required to read solution.h5; install the dev or io extra.') from exc
    return h5py


def _read_observables(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _ratio(local: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return local / reference


def _ref_float(mapping: dict[str, Any], key: str) -> float:
    return float(mapping[key])


def load_local_result(output_dir: Path) -> dict[str, Any]:
    h5py = _require_h5py()
    solution_path = output_dir / 'solution.h5'
    observables_path = output_dir / 'observables.csv'
    with h5py.File(solution_path, 'r') as h5:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['state_labels'][:]]
        t = np.asarray(h5['t_s'][:], dtype=float)
        y = np.asarray(h5['y'][:], dtype=float)
    idx = {label: i for i, label in enumerate(labels)}
    ar_plus = y[idx['n[plasma,Ar_plus]']]
    ar2_plus = y[idx['n[plasma,Ar2_plus]']]
    ne = ar_plus + ar2_plus
    peak_idx = int(np.argmax(ne))
    final_obs = _read_observables(observables_path)[-1]
    return {
        'final_time_s': float(t[-1]),
        'final_electron_density_m3': float(ne[-1]),
        'peak_electron_density_m3': float(ne[peak_idx]),
        'peak_electron_density_time_s': float(t[peak_idx]),
        'final_Ar_density_m3': float(y[idx['n[plasma,Ar]'], -1]),
        'final_Ar_star_density_m3': float(y[idx['n[plasma,Ar_star]'], -1]),
        'final_Ar_plus_density_m3': float(ar_plus[-1]),
        'final_Ar2_plus_density_m3': float(ar2_plus[-1]),
        'final_mean_electron_energy_eV': _float(final_obs, 'mean_electron_energy_eV'),
        'final_reduced_field_Td': _float(final_obs, 'EoverN_plasma_Td'),
        'final_pressure_Pa': _float(final_obs, 'pressure_plasma_Pa'),
        'final_absorbed_power_W': _float(final_obs, 'total_absorbed_power_W'),
    }


def build_report(output_dir: Path, reference_path: Path) -> dict[str, Any]:
    local = load_local_result(output_dir)
    reference = yaml.safe_load(reference_path.read_text(encoding='utf-8'))
    saved = reference['saved_output']
    ref = {
        'final_saved_time_s': _ref_float(saved, 'final_saved_time_s'),
        'final_saved_electron_density_m3': _ref_float(saved, 'final_saved_electron_density_m3'),
        'peak_saved_electron_density_m3': _ref_float(saved, 'peak_saved_electron_density_m3'),
        'peak_saved_electron_density_time_s': _ref_float(saved, 'peak_saved_electron_density_time_s'),
        'final_saved_Ar_star_density_m3': _ref_float(saved, 'final_saved_Ar_star_density_m3'),
        'final_saved_Ar_plus_density_m3': _ref_float(saved, 'final_saved_Ar_plus_density_m3'),
        'final_saved_Ar2_plus_density_m3': _ref_float(saved, 'final_saved_Ar2_plus_density_m3'),
        'final_saved_reduced_field_Td': _ref_float(saved, 'final_saved_reduced_field_Td'),
        'final_saved_voltage_V': _ref_float(saved, 'final_saved_voltage_V'),
        'final_saved_current_A': _ref_float(saved, 'final_saved_current_A'),
    }
    comparisons = {
        'final_electron_density_ratio_local_over_zdplaskin': _ratio(local['final_electron_density_m3'], ref['final_saved_electron_density_m3']),
        'peak_electron_density_ratio_local_over_zdplaskin': _ratio(local['peak_electron_density_m3'], ref['peak_saved_electron_density_m3']),
        'final_Ar_star_ratio_local_over_zdplaskin': _ratio(local['final_Ar_star_density_m3'], ref['final_saved_Ar_star_density_m3']),
        'final_Ar_plus_ratio_local_over_zdplaskin': _ratio(local['final_Ar_plus_density_m3'], ref['final_saved_Ar_plus_density_m3']),
        'final_Ar2_plus_ratio_local_over_zdplaskin': _ratio(local['final_Ar2_plus_density_m3'], ref['final_saved_Ar2_plus_density_m3']),
        'final_reduced_field_ratio_local_over_zdplaskin': _ratio(local['final_reduced_field_Td'], ref['final_saved_reduced_field_Td']),
    }
    return {
        'source_reference': {
            'repository': reference['source']['repository'],
            'example_path': reference['source']['example_path'],
            'committed_output_file': reference['source']['committed_output_file'],
            'note': reference['source']['note'],
        },
        'local_case': {
            'output_dir': str(output_dir),
            'power_model': 'internal dc_series_circuit',
            'electron_impact_rate_model': 'external output-derived E/N-rate table',
            'absorbed_power_W': local['final_absorbed_power_W'],
            'pressure_Pa': local['final_pressure_Pa'],
        },
        'local_result': local,
        'zdplaskin_reference': ref,
        'comparison': comparisons,
        'reduced_field_physical_comparison': build_eovern_report(output_dir, reference_path),
        'limitations': [
            'The local case uses this repository internal dc_series_circuit backend with the ZDPlaskin source, ballast, gap, and area settings.',
            'Electron-impact rates are read from a local external E/N-rate table reverse-engineered from the committed ZDPlaskin output, not from a live BOLSIG+ executable call.',
            'The Ar* de-excitation/superelastic channel is represented by the output-derived E/N-rate table.',
            'The benchmark uses a ZDPlaskin-style first-order diffusion loss setting; it is not a general wall-model validation.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='External-tool comparison for the ZDPlaskin example2-inspired DC-series case.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--write', type=Path, default=None, help='Output YAML path; defaults to <output-dir>/comparison_zdplaskin_example2.yaml')
    args = parser.parse_args()
    out_path = args.write or (args.output_dir / 'comparison_zdplaskin_example2.yaml')
    report = build_report(args.output_dir.resolve(), args.reference.resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
    print(out_path.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
