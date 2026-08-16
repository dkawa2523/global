from __future__ import annotations

from collections.abc import Callable, Mapping
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


def test_table_mobility_obeys_reduced_mobility_similarity(tmp_path: Path) -> None:
    reference_density = 2.0e20
    table = TabulatedElectronKinetics.from_hdf5(
        _table(tmp_path / "rates.h5"),
        lookup="local_field",
        required_rate_ids=("ionize",),
        mobility_reference_neutral_density_m3=reference_density,
    )

    reference = table.evaluate(
        reduced_field_Td=2.0,
        neutral_density_m3=reference_density,
    )
    denser = table.evaluate(
        reduced_field_Td=2.0,
        neutral_density_m3=2.0 * reference_density,
    )

    assert denser.mobility_m2_V_s == pytest.approx(0.5 * reference.mobility_m2_V_s)
    assert denser.mean_energy_eV == reference.mean_energy_eV
    assert denser.rate_coefficients == reference.rate_coefficients
    assert table.zero_field_result(
        neutral_density_m3=2.0 * reference_density
    ).mobility_m2_V_s == pytest.approx(
        0.5
        * table.zero_field_result(neutral_density_m3=reference_density).mobility_m2_V_s
    )
    with pytest.raises(ModelDomainError, match="requires current neutral_density_m3"):
        table.evaluate(reduced_field_Td=2.0)


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


def test_hdf5_table_rejects_complex_values_instead_of_dropping_imaginary_part(
    tmp_path: Path,
) -> None:
    path = _table(tmp_path / "rates.h5")
    with h5py.File(path, "a") as handle:
        del handle["mean_energy_eV"]
        handle["mean_energy_eV"] = np.array([2.0 + 1.0j, 4.0, 6.0])

    with pytest.raises(CaseValidationError, match="must be numeric"):
        TabulatedElectronKinetics.from_hdf5(path, lookup="local_field")


def test_table_rejects_unused_or_unknown_transport_datasets(tmp_path: Path) -> None:
    path = _table(tmp_path / "rates.h5")
    with h5py.File(path, "a") as handle:
        handle["diffusion_m2_s"] = np.ones(3)

    with pytest.raises(CaseValidationError, match="unknown datasets"):
        TabulatedElectronKinetics.from_hdf5(path, lookup="local_field")


def _direct_table(
    *,
    lookup: str = "local_field",
    bounds: str = "error",
    axis: np.ndarray | None = None,
    mean_energy_eV: np.ndarray | None = None,
    rate_tables: Mapping[str, np.ndarray] | None = None,
) -> TabulatedElectronKinetics:
    return TabulatedElectronKinetics(
        source=Path("programmatic-electron-table"),
        lookup=lookup,
        bounds=bounds,
        axis=np.array([1.0, 2.0, 3.0]) if axis is None else axis,
        mean_energy_eV=(
            np.array([2.0, 4.0, 6.0]) if mean_energy_eV is None else mean_energy_eV
        ),
        mobility_m2_V_s=np.array([0.5, 0.4, 0.3]),
        effective_field_Td=np.array([1.0, 2.0, 3.0]),
        rate_tables=(
            {"ionize": np.array([1.0e-16, 2.0e-16, 3.0e-16])}
            if rate_tables is None
            else rate_tables
        ),
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: _direct_table(lookup="bad"), "lookup must"),
        (lambda: _direct_table(bounds="bad"), "bounds must"),
        (
            lambda: _direct_table(axis=np.array([1.0, 3.0, 2.0])),
            "strictly increasing",
        ),
        (
            lambda: _direct_table(mean_energy_eV=np.array([2.0, -1.0, 6.0])),
            "transport values",
        ),
        (
            lambda: _direct_table(rate_tables={"ionize": np.ones(2)}),
            "equal 1-D arrays",
        ),
        (
            lambda: _direct_table(axis=np.array(["1", "2", "3"])),
            "must be numeric",
        ),
        (
            lambda: _direct_table(mean_energy_eV=np.array([False, True, True])),
            "must be numeric",
        ),
        (
            lambda: _direct_table(mean_energy_eV=np.array([2.0 + 1.0j, 4.0, 6.0])),
            "must be numeric",
        ),
    ],
)
def test_programmatic_table_uses_the_canonical_array_contract(
    build: Callable[[], TabulatedElectronKinetics], message: str
) -> None:
    with pytest.raises(CaseValidationError, match=message):
        build()


def test_programmatic_table_owns_read_only_array_copies() -> None:
    axis = np.array([1.0, 2.0, 3.0])
    rate = np.array([1.0e-16, 2.0e-16, 3.0e-16])
    table = _direct_table(axis=axis, rate_tables={"ionize": rate})

    axis[0] = 99.0
    rate[0] = 99.0

    assert table.axis.tolist() == [1.0, 2.0, 3.0]
    assert table.rate_tables["ionize"].tolist() == [1.0e-16, 2.0e-16, 3.0e-16]
    assert not table.axis.flags.writeable
    assert not table.rate_tables["ionize"].flags.writeable
