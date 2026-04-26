from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import yaml


REACTION_FIELDS = [
    'reaction_id',
    'phase',
    'equation',
    'rate_model_key',
    'energy_model_key',
    'zone_filter',
    'surface_filter',
    'enabled',
    'notes',
]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', value.strip())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned.upper() or 'LOSS'


def _loss_rate_s_inv(item: dict[str, Any]) -> tuple[float, str]:
    model = str(item.get('model', 'first_order')).lower()
    if model in {'first_order', 'first_order_loss'}:
        rate = float(item['rate_s_inv'])
        return rate, 'rate_s_inv'
    if model in {'diffusion_length', 'diffusion'}:
        D = float(item['diffusion_coefficient_m2_s'])
        L = float(item['diffusion_length_m'])
        factor = float(item.get('geometry_factor', 1.0))
        return factor * D / max(L, 1.0e-30) ** 2, 'geometry_factor * D / diffusion_length_m^2'
    if model in {'wall_recombination', 'wall_loss'}:
        D = float(item['diffusion_coefficient_m2_s'])
        L = float(item['diffusion_length_m'])
        probability = float(item.get('wall_probability', 1.0))
        factor = float(item.get('geometry_factor', 1.0))
        return probability * factor * D / max(L, 1.0e-30) ** 2, 'wall_probability * geometry_factor * D / diffusion_length_m^2'
    raise ValueError(f'Unsupported loss model {model!r}')


def build_loss_tables(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    rate_models: dict[str, Any] = {'rate_models': {}}
    for item in config.get('loss_models', []) or []:
        species = str(item['species'])
        return_species = str(item['return_species'])
        zone = str(item.get('zone', item.get('zone_filter', '')))
        if not zone:
            raise ValueError(f'Loss model for {species} requires zone or zone_filter')
        reaction_id = str(item.get('reaction_id') or f'{_safe_id(species)}_LOSS')
        rate_model_key = str(item.get('rate_model_key') or f'RM_{_safe_id(reaction_id)}')
        rate_s_inv, formula = _loss_rate_s_inv(item)
        if rate_s_inv < 0.0:
            raise ValueError(f'Loss model for {species} produced a negative rate_s_inv')
        notes = str(item.get('notes') or f'generated species-specific loss: {formula}')
        rows.append(
            {
                'reaction_id': reaction_id,
                'phase': 'gas',
                'equation': f'{species} -> {return_species}',
                'rate_model_key': rate_model_key,
                'energy_model_key': '',
                'zone_filter': zone,
                'surface_filter': '',
                'enabled': str(bool(item.get('enabled', True))).lower(),
                'notes': notes,
            }
        )
        rate_models['rate_models'][rate_model_key] = {
            'backend': 'first_order_loss',
            'rate_s_inv': float(rate_s_inv),
            'metadata': {
                'generated_by': 'scripts/generate_species_losses.py',
                'source_model': str(item.get('model', 'first_order')),
                'formula': formula,
                'species': species,
                'return_species': return_species,
                'zone': zone,
            },
        }
    return rows, rate_models


def write_loss_tables(config_path: Path, output_dir: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    rows, rate_models = build_loss_tables(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    reactions_path = output_dir / str(config.get('gas_reactions_file', 'gas_reactions_loss.csv'))
    models_path = output_dir / str(config.get('gas_rate_models_file', 'gas_rate_models_loss.yaml'))
    with reactions_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=REACTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    models_path.write_text(yaml.safe_dump(rate_models, sort_keys=False), encoding='utf-8')
    return reactions_path, models_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate first_order_loss chemistry files from species-specific diffusion/loss settings.')
    parser.add_argument('config', type=Path, help='YAML file containing loss_models.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for generated gas_reactions_loss.csv and gas_rate_models_loss.yaml.')
    args = parser.parse_args()
    reactions_path, models_path = write_loss_tables(args.config.resolve(), args.output_dir.resolve())
    print(reactions_path)
    print(models_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
