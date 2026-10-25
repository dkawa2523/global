"""Analytic Maxwellian averages of a piecewise-linear cross section."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc

from plasma_global.chemistry._contracts import normalized_cross_section_curve
from plasma_global.chemistry.data import CrossSectionData
from plasma_global.errors import ChemistryError, ModelDomainError

E_CHARGE = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
_RATE_PREFACTOR = (
    2.0 / math.sqrt(math.pi) * math.sqrt(2.0 * E_CHARGE / ELECTRON_MASS_KG)
)
_TAIL_FRACTION = 1.0e-6
_RELATIVE_TAIL_ENVELOPE_LIMIT = 1.0e-3


def _tail_cutoff() -> float:
    lower, upper = 0.0, 64.0
    for _ in range(64):
        middle = (lower + upper) / 2.0
        if (1.0 + middle) * math.exp(-middle) > _TAIL_FRACTION:
            lower = middle
        else:
            upper = middle
    return upper


_TAIL_CUTOFF = _tail_cutoff()


@dataclass(frozen=True, slots=True)
class MaxwellRate:
    """Prepared interval data; evaluation introduces no rate-table interpolation."""

    cross_section_id: str
    distribution_mean_energy_limit_eV: float
    maximum_energy_eV: float
    maximum_sigma_m2: float
    energy_low_eV: np.ndarray
    energy_width_eV: np.ndarray
    sigma_low_m2: np.ndarray
    sigma_high_m2: np.ndarray

    def __call__(self, mean_energy_eV: float) -> float:
        mean = float(mean_energy_eV)
        if (
            not math.isfinite(mean)
            or not 0.0 <= mean <= self.distribution_mean_energy_limit_eV
        ):
            raise ModelDomainError(
                f"mean_energy_eV={mean:g} exceeds the distribution-tail range "
                f"[0, {self.distribution_mean_energy_limit_eV:g}] "
                f"for {self.cross_section_id!r}"
            )
        temperature = (2.0 / 3.0) * mean
        if temperature == 0.0:
            return 0.0
        with np.errstate(over="ignore", under="ignore"):
            lower = self.energy_low_eV / temperature
            width = self.energy_width_eV / temperature
        active = lower < 746.0  # Remaining exponential weights underflow in float64.
        lower, width = lower[active], width[active]
        # Shift each interval to zero. Incomplete gamma moments avoid subtracting
        # nearly equal antiderivatives at narrow intervals and threshold jumps.
        first = gammainc(2.0, width)
        second = 2.0 * gammainc(3.0, width)
        total_weight = lower * (-np.expm1(-width)) + first
        high_weight = np.divide(
            lower * first + second,
            width,
            out=np.zeros_like(width),
            where=width > 0.0,
        )
        low_weight = np.maximum(total_weight - high_weight, 0.0)
        integral = np.sum(
            np.exp(-lower)
            * (
                self.sigma_low_m2[active] * low_weight
                + self.sigma_high_m2[active] * high_weight
            )
        )
        tail_energy = self.maximum_energy_eV / temperature
        tail_envelope = (
            self.maximum_sigma_m2 * math.exp(-tail_energy) * (1.0 + tail_energy)
            if tail_energy < 746.0
            else 0.0
        )
        if tail_envelope > _RELATIVE_TAIL_ENVELOPE_LIMIT * integral:
            raise ModelDomainError(
                f"cross section {self.cross_section_id!r} "
                "has insufficient energy support "
                f"at mean_energy_eV={mean:g}: the observed-maximum sigma tail envelope "
                f"exceeds {_RELATIVE_TAIL_ENVELOPE_LIMIT:g} "
                "of the measured-support rate; "
                f"extend the source curve above {self.maximum_energy_eV:g} eV"
            )
        return float(_RATE_PREFACTOR * math.sqrt(temperature) * integral)


def compile_maxwell_rate(cross_section: CrossSectionData) -> MaxwellRate:
    try:
        energy, sigma = normalized_cross_section_curve(cross_section)
    except ValueError as exc:
        raise ChemistryError(str(exc)) from exc
    arrays = (energy[:-1].copy(), np.diff(energy), sigma[:-1].copy(), sigma[1:].copy())
    for values in arrays:
        values.setflags(write=False)
    return MaxwellRate(
        cross_section.id,
        float(energy[-1]) * (1.5 / _TAIL_CUTOFF),
        float(energy[-1]),
        float(np.max(sigma)),
        *arrays,
    )


def maxwell_integration_policy() -> dict[str, object]:
    """Describe the finite-data support checks without asserting an unknown tail."""

    return {
        "integration": "piecewise_linear_measured_support",
        "distribution_tail_limit": _TAIL_FRACTION,
        "relative_tail_envelope_limit": _RELATIVE_TAIL_ENVELOPE_LIMIT,
        "tail_envelope": "maximum_tabulated_sigma_above_last_energy",
        "physical_tail_bound_established": False,
    }
