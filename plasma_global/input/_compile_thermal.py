"""Bind the heavy-particle thermodynamics once from canonical species data."""

from __future__ import annotations

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.errors import CaseValidationError
from plasma_global.models.gas_energy import HeavyParticleThermodynamics


def heavy_thermodynamics(
    data: ChemistryData, chemistry: CompiledChemistry
) -> HeavyParticleThermodynamics:
    species = {item.id: item for item in data.species}
    missing = [
        name for name in chemistry.species_ids if species[name].cv_over_kb is None
    ]
    if missing:
        raise CaseValidationError(
            "evolved gas energy requires cv_over_kb for every heavy gas species; "
            f"missing {missing}"
        )
    return HeavyParticleThermodynamics(
        np.asarray(
            [species[name].cv_over_kb for name in chemistry.species_ids], dtype=float
        )
    )
