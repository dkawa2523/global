from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.rate_model_schema import COVERAGE_FACTOR_KINDS


def surface_coverage_factor(core: Any, cfg: dict[str, Any] | None, surface_id: str, y: np.ndarray) -> tuple[float, set[str]]:
    system = core.system
    if not cfg:
        return 1.0, set()
    kind = str(cfg.get('kind', 'constant')).lower()
    if kind not in COVERAGE_FACTOR_KINDS:
        raise ValueError(f'Unsupported coverage factor kind: {kind}')
    if kind in {'site_blocking', 'species_power'}:
        site_species = str(cfg.get('site_species') or cfg.get('species'))
        exponent = float(cfg.get('exponent', 1.0))
        idx = system.state_layout.surface_index[surface_id][site_species]
        theta = float(np.clip(y[idx], 0.0, 1.0))
        if theta <= 0.0 and exponent > 0.0:
            return 0.0, {site_species}
        return theta ** exponent, {site_species}
    return 1.0, set()
