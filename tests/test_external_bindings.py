from pathlib import Path

import pytest

from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.kinetics import TabulatedElectronKinetics


def test_coefficient_columns_are_explicit_and_unselected_text_is_not_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "instrument.csv"
    source.write_text(
        "Energy,Rate,Note\n1,1e-16,first\n3,3e-16,last\n", encoding="utf-8"
    )
    table = TabulatedElectronKinetics.from_csv(
        source,
        lookup="mean_energy",
        columns={"mean_energy_eV": "Energy", "k:ionize": "Rate"},
    )
    assert table.evaluate(mean_energy_eV=2.0).rate_coefficients[
        "ionize"
    ] == pytest.approx(2e-16, abs=0.0)
    with pytest.raises(ValueError, match="missing source columns"):
        TabulatedElectronKinetics.from_csv(
            source, lookup="mean_energy", columns={"mean_energy_eV": "missing"}
        )


def test_same_file_with_different_physical_bindings_is_not_one_cached_table(
    tmp_path: Path,
) -> None:
    source = tmp_path / "power.csv"
    source.write_text("t,power_a,power_b,Note\n0,1,3,on\n1,2,4,off\n", encoding="utf-8")
    store = ExternalTableStore()
    first = store.get(source, columns={"time_s": "t", "electron_power_W": "power_a"})
    second = store.get(source, columns={"time_s": "t", "electron_power_W": "power_b"})
    assert first.electron_power_W.tolist() == [1.0, 2.0]
    assert second.electron_power_W.tolist() == [3.0, 4.0]
