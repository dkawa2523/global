from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from plasma_global.numerics.jacobian_check import check_jacobian
from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator


class DecaySystem:
    def rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        return -y

    def jacobian(self, _t: float, y: np.ndarray) -> csr_matrix:
        return csr_matrix([[-1.0]])

    def scipy_events(self) -> list:
        return []

    def state_labels(self) -> list[str]:
        return ['y']


def test_scipy_bdf_allows_default_max_step() -> None:
    result = SciPyBDFIntegrator(max_step=None, first_step=None).solve(
        system=DecaySystem(),
        y0=np.array([1.0]),
        t_span=(0.0, 1.0e-6),
        t_eval=np.array([0.0, 1.0e-6]),
    )
    assert result.success is True
    assert result.y.shape == (1, 2)


def test_jacobian_check_reports_small_error_for_decay_system() -> None:
    result = check_jacobian(DecaySystem(), 0.0, np.array([1.0]), top_n=1)
    assert result.max_relative_error < 1.0e-8
    assert result.mismatches[0].row_label == 'y'
