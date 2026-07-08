"""Built-in electrical backend registry."""

from __future__ import annotations

from plasma_global.backends import BackendCatalog, BackendSpec
from plasma_global.electrical.ccp import CCPBackend
from plasma_global.electrical.dc_series import DCSeriesCircuitBackend, validate_dc_series_port_config
from plasma_global.electrical.direct_power import DirectPowerBackend
from plasma_global.electrical.external_table import ExternalCircuitTableBackend, validate_external_circuit_table_port_config
from plasma_global.electrical.icp import ICPBackend
from plasma_global.electrical.rf_envelope import RFEnvelopeBackend, validate_rf_envelope_port_config


ELECTRICAL_REGISTRY = BackendCatalog(
    {
        'direct_power': BackendSpec(
            lambda **kwargs: DirectPowerBackend(),
            'Direct absorbed-power prescription.',
            maturity='prescribed',
            intended_use='Cases where absorbed power is an input rather than a circuit prediction.',
            caveat='Reduced field, bias, and plasma-potential values are simple proxies.',
        ),
        'dc_series_circuit': BackendSpec(
            lambda **kwargs: DCSeriesCircuitBackend(),
            'Reduced voltage-source plus ballast-resistor circuit for DC and pulsed DC cases.',
            validate_dc_series_port_config,
            maturity='reduced',
            intended_use='Compact DC or pulsed-DC voltage-source studies with plasma conductivity feedback.',
            caveat='Not a full circuit DAE or sheath-resolved discharge model.',
        ),
        'external_circuit_table': BackendSpec(
            lambda **kwargs: ExternalCircuitTableBackend(),
            'One-way coupling from measured or SPICE-generated circuit waveform CSV data.',
            validate_external_circuit_table_port_config,
            maturity='data_driven',
            intended_use='Replay measured, SPICE, or external executable waveforms without online coupling.',
            caveat='One-way interpolation only; plasma state does not feed back into the external circuit data.',
        ),
        'rf_envelope': BackendSpec(
            lambda **kwargs: RFEnvelopeBackend(),
            'Cycle-averaged HF/LF RF power and bias envelope model.',
            validate_rf_envelope_port_config,
            maturity='reduced',
            intended_use='Calibrated envelope studies where cycle-resolved RF dynamics are outside scope.',
            caveat='Requires case-specific calibration for quantitative bias or sheath claims.',
        ),
        'ccp': BackendSpec(
            lambda **kwargs: CCPBackend(),
            'Reduced CCP / bias backend.',
            maturity='experimental',
            intended_use='Compact CCP/bias sensitivity studies and IED proxy generation.',
            caveat='Uses empirical lumped sheath and resistance proxies.',
        ),
        'icp': BackendSpec(
            lambda **kwargs: ICPBackend(),
            'Reduced ICP source-coupling backend.',
            maturity='experimental',
            intended_use='Compact ICP source-coupling and downstream power-deposition studies.',
            caveat='Uses empirical E/H-mode and coupling-efficiency proxies.',
        ),
    }
)


__all__ = ['ELECTRICAL_REGISTRY']
