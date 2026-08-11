from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from plasma_global.core.exceptions import ModelConfigurationError, StateDomainError
from plasma_global.models.rates import ConstantRate, DensityView, RateContext


def _context(*, mobility: float | None = None) -> RateContext:
    return RateContext(
        time_s=0.0,
        zone_id="plasma",
        gas_temperature_K=300.0,
        pressure_Pa=5.0,
        electron_density_m3=1.0e15,
        mean_energy_eV=3.0,
        electron_temperature_eV=2.0,
        reduced_field_Td=50.0,
        electron_mobility_m2_V_s=mobility,
        rate_coefficients={},
        densities_m3={"Ar": 1.0e20},
    )


def test_density_view_is_a_live_read_only_mapping() -> None:
    density = np.array([1.0, 2.0], dtype=np.float64)
    view = DensityView(("Ar", "Ar_plus"), {"Ar": 0, "Ar_plus": 1}, density)

    assert tuple(view) == ("Ar", "Ar_plus")
    assert len(view) == 2
    assert view["Ar"] == 1.0
    assert tuple(view.values()) == (1.0, 2.0)
    with pytest.raises(KeyError, match="missing"):
        view["missing"]

    density[0] = 4.0
    assert view["Ar"] == 4.0


@pytest.mark.property
@given(st.floats(min_value=0.0, max_value=1.0e-6, allow_nan=False))
def test_constant_rate_round_trips_every_finite_nonnegative_value(
    coefficient: float,
) -> None:
    evaluator = ConstantRate(coefficient)

    assert evaluator(_context()) == coefficient


@pytest.mark.parametrize("coefficient", [-math.inf, -1.0, -5e-324, math.inf, math.nan])
def test_constant_rate_rejects_nonfinite_or_negative_coefficients(
    coefficient: float,
) -> None:
    with pytest.raises(ModelConfigurationError) as error:
        ConstantRate(coefficient)
    assert str(error.value) == (
        "Constant rate coefficient must be finite and non-negative"
    )


@pytest.mark.property
@given(
    st.one_of(
        st.none(),
        st.floats(min_value=5e-324, max_value=1.0e6, allow_nan=False),
    )
)
def test_rate_context_accepts_absent_or_positive_finite_mobility(
    mobility: float | None,
) -> None:
    assert _context(mobility=mobility).electron_mobility_m2_V_s == mobility


@pytest.mark.parametrize(
    "mobility", [-math.inf, -1.0, -5e-324, 0.0, math.inf, math.nan]
)
def test_rate_context_rejects_invalid_mobility(mobility: float) -> None:
    with pytest.raises(StateDomainError) as error:
        _context(mobility=mobility)
    assert str(error.value) == "Electron mobility must be finite and positive"
