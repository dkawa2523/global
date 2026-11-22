"""Small coefficient tables and power ports for physical model tests."""

from pathlib import Path

import numpy as np

from plasma_global.models.external_table import ExternalTableBinding, ExternalTableData
from plasma_global.models.kinetics import TabulatedElectronKinetics
from plasma_global.models.power import ExternalTablePowerPort


def constant_field_port(zone_id: str, field_Td: float) -> ExternalTablePowerPort:
    """Supply a fixed E/N through the external command-table port."""
    return ExternalTablePowerPort(
        port_id=f"field_{zone_id}",
        zone_id=zone_id,
        default=ExternalTableBinding(
            data=ExternalTableData(
                source=Path("constant-field.csv"),
                time_s=np.array([0.0, 1.0]),
                electron_power_W=np.zeros(2),
                gas_power_W=np.zeros(2),
                reduced_field_Td=np.full(2, field_Td),
                voltage_V=None,
                current_A=None,
            ),
            interpolation="previous",
            bounds="hold",
            power_scale=1.0,
            voltage_scale=1.0,
            current_scale=1.0,
            gap_m=None,
            total_density_m3=None,
            plasma_potential_V=0.0,
        ),
    )


def constant_mean_energy_table(mean_energy_eV: float) -> TabulatedElectronKinetics:
    """Keep mean energy fixed at E/N >= 1 Td, with an explicit cold boundary."""
    return TabulatedElectronKinetics(
        source=Path("constant-mean-energy.csv"),
        lookup="local_field",
        bounds="error",
        axis=np.array([0.0, 1.0, 100.0]),
        mean_energy_eV=np.array([0.0, mean_energy_eV, mean_energy_eV]),
        mobility_m2_V_s=None,
        effective_field_Td=np.array([0.0, 1.0, 100.0]),
        rate_tables={},
    )
