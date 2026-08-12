from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from plasma_global.errors import CaseValidationError
from plasma_global.models.external_table import load_external_table


def _write_table(tmp_path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    path = tmp_path / "power.csv"
    path.write_text(content, encoding=encoding)
    return path


def test_load_external_table_preserves_numeric_array_contract(tmp_path: Path) -> None:
    path = _write_table(
        tmp_path,
        "time_s,absorbed_power_W,gas_power_W,reduced_field_Td,voltage_V,current_A\n"
        "0,5,1,10,-2,-2\n"
        "1,8,2,20,3,3\n",
        encoding="utf-8-sig",
    )

    table = load_external_table(path)

    assert table.source == path.resolve()
    np.testing.assert_array_equal(table.time_s, [0.0, 1.0])
    np.testing.assert_array_equal(table.electron_power_W, [5.0, 8.0])
    np.testing.assert_array_equal(table.gas_power_W, [1.0, 2.0])
    np.testing.assert_array_equal(table.reduced_field_Td, [10.0, 20.0])
    np.testing.assert_array_equal(table.voltage_V, [-2.0, 3.0])
    np.testing.assert_array_equal(table.current_A, [-2.0, 3.0])

    arrays = (
        table.time_s,
        table.electron_power_W,
        table.gas_power_W,
        table.reduced_field_Td,
        table.voltage_V,
        table.current_A,
    )
    for values in arrays:
        assert values is not None
        assert values.dtype == np.dtype(float)
        assert values.shape == (2,)
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0


def test_voltage_current_schema_derives_power_and_defaults_optional_columns(
    tmp_path: Path,
) -> None:
    path = _write_table(
        tmp_path,
        "time_s,voltage_V,current_A\n0,-2,-3\n1,4,5\n",
    )

    table = load_external_table(path)

    np.testing.assert_array_equal(table.electron_power_W, [6.0, 20.0])
    np.testing.assert_array_equal(table.gas_power_W, [0.0, 0.0])
    assert table.reduced_field_Td is None
    np.testing.assert_array_equal(table.voltage_V, [-2.0, 4.0])
    np.testing.assert_array_equal(table.current_A, [-3.0, 5.0])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("electron_power_W\n1\n2\n", "is missing time_s"),
        (
            "time_s,electron_power_W,absorbed_power_W\n0,1,1\n1,2,2\n",
            "has ambiguous power columns",
        ),
        (
            "time_s,voltage_V\n0,1\n1,2\n",
            (
                "needs electron_power_W, absorbed_power_W, or both voltage_V "
                "and current_A"
            ),
        ),
        ("time_s,electron_power_W\n0,1\n", "needs at least two rows"),
        (
            "time_s,electron_power_W\n0,not-a-number\n1,2\n",
            "column 'electron_power_W' must be numeric",
        ),
        (
            "time_s,electron_power_W\n0,nan\n1,2\n",
            "contains non-finite values",
        ),
        (
            "time_s,electron_power_W\n1,1\n0,2\n",
            "time_s must be strictly increasing",
        ),
        (
            "time_s,electron_power_W\n0,-1\n1,2\n",
            "contains negative power",
        ),
        (
            "time_s,electron_power_W,gas_power_W\n0,1,-1\n1,2,0\n",
            "contains negative power",
        ),
        (
            "time_s,electron_power_W,reduced_field_Td\n0,1,-1\n1,2,0\n",
            "contains negative reduced field",
        ),
    ],
)
def test_external_table_reports_schema_and_numeric_failures(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = _write_table(tmp_path, content)
    expected = (
        rf"external power table {re.escape(str(path.resolve()))} "
        rf"{re.escape(message)}$"
    )

    with pytest.raises(CaseValidationError, match=expected):
        load_external_table(path)


def test_external_table_wraps_file_and_csv_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    missing_prefix = (
        rf"cannot read external power table {re.escape(str(missing.resolve()))}:"
    )
    with pytest.raises(CaseValidationError, match=missing_prefix):
        load_external_table(missing)

    malformed = _write_table(
        tmp_path,
        'time_s,electron_power_W\n0,"unterminated\n',
    )
    malformed_prefix = (
        rf"external power table {re.escape(str(malformed.resolve()))} is malformed CSV:"
    )
    with pytest.raises(CaseValidationError, match=malformed_prefix):
        load_external_table(malformed)
