from __future__ import annotations

from pathlib import Path

import pytest

from plasma_global.errors import ModelDomainError
from plasma_global.models.kinetics import TabulatedElectronKinetics


def test_energy_table_requires_only_consumed_quantities(tmp_path: Path) -> None:
    path = tmp_path / "rates.csv"
    path.write_text("mean_energy_eV,k:ionize\n1,1e-16\n3,3e-16\n", encoding="utf-8")
    table = TabulatedElectronKinetics.from_csv(
        path, lookup="mean_energy", required_rate_ids=("ionize",)
    )
    result = table.evaluate(mean_energy_eV=2.0)
    assert result.rate_coefficients["ionize"] == pytest.approx(2e-16, abs=0.0)
    assert result.mobility_m2_V_s is None
    assert result.effective_field_Td is None
    with pytest.raises(ValueError, match="requires reduced_field"):
        TabulatedElectronKinetics.from_csv(path, lookup="local_field")
    with pytest.raises(ValueError, match="requires electron mobility"):
        TabulatedElectronKinetics.from_csv(
            path, lookup="mean_energy", require_mobility=True
        )


def test_reduced_mobility_has_no_inferred_reference_density(tmp_path: Path) -> None:
    path = tmp_path / "rates.csv"
    path.write_text(
        "reduced_field_Td,mean_energy_eV,reduced_mobility_1_V_m_s,k:ionize\n1,2,1e20,1e-16\n3,6,6e19,3e-16\n",
        encoding="utf-8",
    )
    table = TabulatedElectronKinetics.from_csv(path, lookup="local_field")
    assert table.mobility_m2_V_s is None
    assert table.mobility_reference_neutral_density_m3 is None
    assert table.reduced_mobility_1_V_m_s is not None
    result = table.evaluate(reduced_field_Td=2.0, neutral_density_m3=2e20)
    assert result.mean_energy_eV == 4.0
    assert result.electron_temperature_eV == pytest.approx(8 / 3)
    assert result.mobility_m2_V_s == pytest.approx(0.4)
    with pytest.raises(ModelDomainError, match="requires current neutral_density"):
        table.evaluate(reduced_field_Td=2.0)


def test_absolute_mobility_requires_the_generation_density(tmp_path: Path) -> None:
    path = tmp_path / "rates.csv"
    path.write_text("mean_energy_eV,mobility_m2_V_s\n1,0.5\n3,0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="table generation"):
        TabulatedElectronKinetics.from_csv(path, lookup="mean_energy")
    table = TabulatedElectronKinetics.from_csv(
        path, lookup="mean_energy", mobility_reference_neutral_density_m3=2e20
    )
    assert table.evaluate(
        mean_energy_eV=2.0, neutral_density_m3=4e20
    ).mobility_m2_V_s == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("mean_energy_eV,k:x\n1,2\n1,3\n", "strictly increasing"),
        ("mean_energy_eV,k:x\n2,2\n1,3\n", "strictly increasing"),
        ("mean_energy_eV,k:x\n1,-2\n2,3\n", "negative rates"),
        ("mean_energy_eV,k:x\n1,nan\n2,3\n", "finite"),
        ("mean_energy_eV,k:x\n1,\n2,3\n", "numeric"),
        ("mean_energy_eV,k:x\n1,2,3\n2,3\n", "header"),
        ("mean_energy_eV,mean_energy_eV\n1,2\n2,3\n", "duplicate"),
        ("mean_energy_eV,k:x,typo\n1,2,0\n2,3,0\n", "unknown columns"),
        (
            "mean_energy_eV,k:momentum:Ar,momentum:Ar\n1,2,3\n2,3,4\n",
            "duplicate coefficient",
        ),
    ],
)
def test_csv_rejects_ambiguous_or_invalid_data(
    tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        TabulatedElectronKinetics.from_csv(path, lookup="mean_energy")
