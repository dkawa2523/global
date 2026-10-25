from pathlib import Path

import numpy as np

from plasma_global.chemistry.data import (
    ChemistryData,
    CrossSectionData,
    RateModelData,
    ReactionData,
)
from plasma_global.input.migrate_v3 import table_chemistry


def test_shared_rate_retains_each_reactions_energy_override_and_origin() -> None:
    curve = CrossSectionData(
        "ionize",
        "ionization",
        "Ar",
        10.0,
        10.0,
        np.array([10.0, 100.0]),
        np.array([0.0, 1e-20]),
        metadata={"source": "measured"},
    )
    model = RateModelData(
        "shared", "electron_impact", {"cross_section": "ionize", "branching_yield": 0.5}
    )
    inherited = ReactionData(
        "inherited", {"e": 1.0, "Ar": 1.0}, {"e": 2.0, "Ar_plus": 1.0}, "shared", None
    )
    overridden = ReactionData(
        "overridden",
        inherited.reactants,
        inherited.products,
        "shared",
        None,
        electron_energy_transfer_eV=-2.0,
    )
    data = ChemistryData(
        Path("source.yaml"),
        (),
        (inherited, overridden),
        (),
        (),
        {"shared": model},
        {"ionize": curve},
    )
    migrated = table_chemistry(data)
    assert not migrated.cross_sections
    assert [
        reaction.electron_energy_transfer_eV for reaction in migrated.gas_reactions
    ] == [-10.0, -2.0]
    assert migrated.rate_models["shared"].parameters == {
        "rate_id": "ionize",
        "branching_yield": 0.5,
    }
    assert migrated.provenance["electron_energy_sources"]["inherited"]["source"] == {
        "source": "measured"
    }
