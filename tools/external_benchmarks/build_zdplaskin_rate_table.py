from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.chemistry.models import K_B
from plasma_global.eedf.base import EEDFRequest
from plasma_global.eedf.boltzmann_2term import Boltzmann2TermSwarmModel
from scripts.build_rate_table_h5 import build_rate_table_h5


DEFAULT_CHEMISTRY = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml'
DEFAULT_TABLE_DIR = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'tables' / 'zdplaskin_example2_wide_eovern'
DEFAULT_OUTPUT = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'tables' / 'zdplaskin_example2_wide_eovern_rates.h5'


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_zdplaskin_rate_table(
    chemistry_manifest: Path = DEFAULT_CHEMISTRY,
    table_dir: Path = DEFAULT_TABLE_DIR,
    output: Path = DEFAULT_OUTPUT,
    *,
    min_EoverN_Td: float = 0.5,
    max_EoverN_Td: float = 250.0,
    n_fields: int = 64,
) -> Path:
    mechanism = load_mechanism_bundle(chemistry_manifest)
    chemistry_dir = chemistry_manifest.resolve().parent
    swarm_cfg = _ns(
        closure='local_field',
        mixture_key_species=['Ar', 'Ar_star'],
        cache=_ns(max_entries=1, fraction_decimals=1),
        boltzmann_2term=_ns(
            energy_grid=_ns(min_eV=1.0e-3, max_eV=500.0, n=220),
            reduced_field_grid_Td=_ns(min=float(min_EoverN_Td), max=float(max_EoverN_Td), n=int(n_fields)),
            max_shape_iterations=24,
            max_field_iterations=20,
        ),
    )
    run_cfg = _ns(paths=_ns(chemistry_dir=str(chemistry_dir)))
    model = Boltzmann2TermSwarmModel()
    model.prepare(mechanism=mechanism, chamber=_ns(), run_config=run_cfg, swarm_config=swarm_cfg)

    pressure_Pa = 13332.0
    gas_temperature_K = 300.0
    n_ar = pressure_Pa / (K_B * gas_temperature_K)
    request = EEDFRequest(
        time_s=0.0,
        zone_id='plasma',
        composition={'Ar': n_ar, 'Ar_star': 0.0},
        electron_density_m3=1.0e16,
        mean_energy_eV=3.0,
        reduced_field_Td=None,
        gas_temperature_K=gas_temperature_K,
        pressure_Pa=pressure_Pa,
        metadata={},
    )
    profile = model._build_mixture_profile(request)
    table = model._solve_field_table(request, profile)

    rate_rows: list[dict[str, float]] = []
    transport_rows: list[dict[str, float]] = []
    for i, field in enumerate(table.field_grid_Td):
        rate_row = {'EoverN_Td': float(field)}
        for cs_id, values in table.k_by_field.items():
            rate_row[cs_id] = float(values[i])
        rate_rows.append(rate_row)
        transport_rows.append(
            {
                'EoverN_Td': float(field),
                'mean_energy_eV': float(table.mean_energy_by_field_eV[i]),
                'mobility_m2_V_s': float(table.mobility_by_field[i]),
                'diffusion_m2_s': float(table.diffusion_by_field[i]),
            }
        )

    table_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(table_dir / 'rates.csv', rate_rows)
    _write_csv(table_dir / 'transport.csv', transport_rows)
    (table_dir / 'metadata.yaml').write_text(
        '\n'.join(
            [
                'source: ZDPlaskin example2 bolsigdb.dat cross sections',
                'generator: tools/external_benchmarks/build_zdplaskin_rate_table.py',
                'grid: EoverN_Td',
                f'min_EoverN_Td: {float(min_EoverN_Td)}',
                f'max_EoverN_Td: {float(max_EoverN_Td)}',
                f'n_fields: {int(n_fields)}',
                'note: Diagnostic wide-range table generated from the repository internal two-term-like solver using the ZDPlaskin cross-section bundle; not a BOLSIG+ executable rerun and not the ZDPlaskin parity table.',
                '',
            ]
        ),
        encoding='utf-8',
    )
    return build_rate_table_h5(table_dir, output)


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a wide diagnostic ZDPlaskin example2 E/N-rate HDF5 table for stress benchmarks.')
    parser.add_argument('--chemistry', type=Path, default=DEFAULT_CHEMISTRY)
    parser.add_argument('--table-dir', type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--min-EoverN-Td', type=float, default=0.5)
    parser.add_argument('--max-EoverN-Td', type=float, default=250.0)
    parser.add_argument('--n-fields', type=int, default=64)
    args = parser.parse_args()
    print(
        build_zdplaskin_rate_table(
            args.chemistry.resolve(),
            args.table_dir.resolve(),
            args.output.resolve(),
            min_EoverN_Td=args.min_EoverN_Td,
            max_EoverN_Td=args.max_EoverN_Td,
            n_fields=args.n_fields,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
