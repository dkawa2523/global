from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.kinetics import TabulatedElectronKinetics


def _table(path: Path, *, unordered: bool = False) -> Path:
    axis = np.array([1.0, 3.0, 2.0]) if unordered else np.array([1.0, 2.0, 3.0])
    with h5py.File(path, "w") as handle:
        handle["effective_field_Td"] = axis
        handle["mean_energy_eV"] = axis * 2.0
        handle["mobility_m2_V_s"] = np.array([0.5, 0.4, 0.3])
        group = handle.create_group("rate_coefficients")
        group["ionize"] = np.array([1.0e-16, 2.0e-16, 3.0e-16])
    return path


def test_table_is_prepared_once_and_interpolates_selected_rates(tmp_path: Path) -> None:
    table = TabulatedElectronKinetics.from_hdf5(
        _table(tmp_path / "rates.h5"),
        lookup="local_field",
        required_rate_ids=("ionize",),
    )
    result = table.evaluate(reduced_field_Td=1.5)
    assert result.mean_energy_eV == 3.0
    assert result.electron_temperature_eV == 2.0
    assert result.rate_coefficients["ionize"] == pytest.approx(1.5e-16)


@pytest.mark.parametrize(
    ("field_Td", "mean_energy_eV", "mobility_m2_V_s", "rate_m3_s"),
    [
        (1.0, 2.0, 0.5, 1.0e-16),
        (2.0, 4.0, 0.4, 2.0e-16),
        (3.0, 6.0, 0.3, 3.0e-16),
    ],
)
def test_table_nodes_are_reproduced_to_1e_12(
    tmp_path: Path,
    field_Td: float,
    mean_energy_eV: float,
    mobility_m2_V_s: float,
    rate_m3_s: float,
) -> None:
    table = TabulatedElectronKinetics.from_hdf5(
        _table(tmp_path / "rates.h5"),
        lookup="local_field",
        required_rate_ids=("ionize",),
    )

    result = table.evaluate(reduced_field_Td=field_Td)

    for actual, expected in (
        (result.effective_field_Td, field_Td),
        (result.mean_energy_eV, mean_energy_eV),
        (result.mobility_m2_V_s, mobility_m2_V_s),
        (result.rate_coefficients["ionize"], rate_m3_s),
    ):
        assert abs(actual - expected) / abs(expected) <= 1.0e-12


def test_table_bounds_error_is_default(tmp_path: Path) -> None:
    table = TabulatedElectronKinetics.from_hdf5(
        _table(tmp_path / "rates.h5"), lookup="local_field"
    )
    with pytest.raises(ModelDomainError, match="outside"):
        table.evaluate(reduced_field_Td=0.0)


def test_table_rejects_nonmonotone_axis_instead_of_sorting(tmp_path: Path) -> None:
    with pytest.raises(CaseValidationError, match="strictly increasing"):
        TabulatedElectronKinetics.from_hdf5(
            _table(tmp_path / "rates.h5", unordered=True), lookup="local_field"
        )


def test_table_rejects_unused_or_unknown_transport_datasets(tmp_path: Path) -> None:
    path = _table(tmp_path / "rates.h5")
    with h5py.File(path, "a") as handle:
        handle["diffusion_m2_s"] = np.ones(3)

    with pytest.raises(CaseValidationError, match="unknown datasets"):
        TabulatedElectronKinetics.from_hdf5(path, lookup="local_field")
