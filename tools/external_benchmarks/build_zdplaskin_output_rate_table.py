from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_rate_table_h5 import build_rate_table_h5


SOURCE_OUTPUT_URL = (
    'https://raw.githubusercontent.com/Hemadityamalla/ZDPlaskin/master/'
    'example2/Ar2Reac_1000V_100kOhm_100t_0.4d_0.4r.dat'
)
DEFAULT_TABLE_DIR = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'tables' / 'zdplaskin_example2_eovern'
DEFAULT_OUTPUT = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'tables' / 'zdplaskin_example2_eovern_rates.h5'
DEFAULT_CIRCUIT_OUTPUT = ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv'
EV_TO_K = 11604.518121550082


def _read_source(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding='utf-8')
    with urlopen(SOURCE_OUTPUT_URL, timeout=30) as response:
        return response.read().decode('utf-8')


def _safe_rate(numerator_cm3_s: float, *densities_cm3: float) -> float:
    denom = 1.0
    for density in densities_cm3:
        denom *= max(float(density), 1.0e-300)
    return max(float(numerator_cm3_s), 0.0) / denom * 1.0e-6


def _infer_mean_energy_eV(row: dict[str, float]) -> float:
    ar_plus = max(row['AR^+'], 1.0e-300)
    diff_rate_s = max(row['11'], 0.0) / ar_plus
    gas_pressure_torr = 100.0
    gas_temperature_K = 300.0
    radius_cm = 0.4
    gap_cm = 0.4
    geom = (2.405 / radius_cm) ** 2 + (math.pi / gap_cm) ** 2
    coeff = 1.52 * (760.0 / gas_pressure_torr) * (gas_temperature_K / 273.16) * geom
    te_K = diff_rate_s * 11600.0 / max(coeff, 1.0e-300)
    return max(1.5 * te_K / EV_TO_K, 1.0e-6)


def _infer_mobility_m2_V_s(row: dict[str, float]) -> float:
    radius_cm = 0.4
    area_cm2 = math.pi * radius_cm * radius_cm
    gas_density_cm3 = max(row['AR'] + row['AR*'], 1.0e-300)
    electric_field_V_cm = max(row['E/N'] * gas_density_cm3 / 1.0e17, 1.0e-300)
    current_A = row['J']
    electron_density_cm3 = max(row['E'], 1.0e-300)
    mobility_cm2_V_s = current_A / (1.602176634e-19 * area_cm2 * electron_density_cm3 * electric_field_V_cm)
    return max(mobility_cm2_V_s * 1.0e-4, 0.0)


def parse_zdplaskin_output(text: str) -> list[dict[str, float]]:
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[0].split()
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        values = line.split()
        if len(values) != len(header):
            continue
        row = {key: float(value.replace('D', 'E').replace('d', 'e')) for key, value in zip(header, values)}
        if row['E/N'] <= 0.0 or row['E'] <= 0.0 or row['AR'] <= 0.0:
            continue
        rows.append(row)
    return rows


def _write_circuit_table(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'voltage_V', 'current_A', 'absorbed_power_W', 'reduced_field_Td'])
        for row in rows:
            writer.writerow([row['Time'], row['V'], row['J'], max(row['V'] * row['J'], 0.0), row['E/N']])


def build_zdplaskin_output_rate_table(source_output: Path | None, table_dir: Path, output: Path, circuit_output: Path | None = None) -> Path:
    rows = parse_zdplaskin_output(_read_source(source_output))
    if not rows:
        raise RuntimeError('No usable rows were found in the ZDPlaskin output file.')

    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / 'rates.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                'EoverN_Td',
                'xs_zdp_ar_effective',
                'xs_zdp_ar_excitation',
                'xs_zdp_ar_ionization',
                'xs_zdp_ar_star_deexcitation',
                'xs_zdp_ar_star_ionization',
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row['E/N'],
                    0.0,
                    _safe_rate(row['2'], row['E'], row['AR']),
                    _safe_rate(row['1'], row['E'], row['AR']),
                    _safe_rate(row['3'], row['E'], row['AR*']),
                    _safe_rate(row['4'], row['E'], row['AR*']),
                ]
            )

    with (table_dir / 'transport.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['EoverN_Td', 'mean_energy_eV', 'mobility_m2_V_s', 'diffusion_m2_s'])
        for row in rows:
            writer.writerow([row['E/N'], _infer_mean_energy_eV(row), _infer_mobility_m2_V_s(row), 0.0])

    (table_dir / 'metadata.yaml').write_text(
        '\n'.join(
            [
                'source: ZDPlaskin example2 committed output',
                'generator: tools/external_benchmarks/build_zdplaskin_output_rate_table.py',
                'grid_column: EoverN_Td',
                'note: Rates are reverse-engineered from saved ZDPlaskin reaction-rate columns, including the BOLSIG+/superelastic behavior present in that run.',
                '',
            ]
        ),
        encoding='utf-8',
    )
    if circuit_output is not None:
        _write_circuit_table(rows, circuit_output)
    return build_rate_table_h5(table_dir, output)


def main() -> int:
    parser = argparse.ArgumentParser(description='Build the ZDPlaskin example2 E/N-rate table from the committed ZDPlaskin output.')
    parser.add_argument('--source-output', type=Path, default=None, help='Optional local Ar2Reac_*.dat file; defaults to the GitHub raw file.')
    parser.add_argument('--table-dir', type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--circuit-output', type=Path, default=DEFAULT_CIRCUIT_OUTPUT)
    args = parser.parse_args()
    print(
        build_zdplaskin_output_rate_table(
            args.source_output,
            args.table_dir.resolve(),
            args.output.resolve(),
            args.circuit_output.resolve() if args.circuit_output else None,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
