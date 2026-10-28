from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest

from plasma_global.errors import CaseValidationError
from plasma_global.models.kinetics import TabulatedElectronKinetics


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
