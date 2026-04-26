from __future__ import annotations

import csv

import yaml

from scripts.generate_species_losses import write_loss_tables


def test_species_loss_generator_writes_first_order_loss_files(tmp_path) -> None:
    cfg = tmp_path / 'losses.yaml'
    cfg.write_text(
        yaml.safe_dump(
            {
                'loss_models': [
                    {
                        'species': 'Ar_star',
                        'return_species': 'Ar',
                        'zone': 'plasma',
                        'model': 'diffusion_length',
                        'diffusion_coefficient_m2_s': 0.05,
                        'diffusion_length_m': 0.01,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    reactions_path, models_path = write_loss_tables(cfg, tmp_path / 'generated')

    rows = list(csv.DictReader(reactions_path.open('r', encoding='utf-8', newline='')))
    models = yaml.safe_load(models_path.read_text(encoding='utf-8'))

    assert rows[0]['equation'] == 'Ar_star -> Ar'
    assert rows[0]['zone_filter'] == 'plasma'
    model = models['rate_models'][rows[0]['rate_model_key']]
    assert model['backend'] == 'first_order_loss'
    assert model['rate_s_inv'] == 500.0
