from __future__ import annotations

ELECTRON_IMPACT_BACKEND_NAMES = {'electron_impact_xsec'}
GAS_RATE_BACKEND_NAMES = {
    'electron_impact_xsec',
    'arrhenius',
    'constant',
    'first_order_loss',
    'te_power_law',
    'electron_temperature_power_law',
    'tabulated_1d',
}
ENERGY_LOSS_BACKEND_NAMES = {'constant_event_loss'}
SURFACE_RATE_BACKEND_NAMES = {
    'sticking',
    'eley_rideal',
    'ion_assisted',
    'ion_yield_table',
    'desorption',
    'langmuir_hinshelwood',
}
COVERAGE_FACTOR_KINDS = {'constant', 'site_blocking', 'species_power'}

MODEL_FILE_BACKENDS = {
    'electron_impact': ELECTRON_IMPACT_BACKEND_NAMES,
    'gas_rate': GAS_RATE_BACKEND_NAMES - ELECTRON_IMPACT_BACKEND_NAMES,
    'energy_loss': ENERGY_LOSS_BACKEND_NAMES,
    'surface_rate': SURFACE_RATE_BACKEND_NAMES,
}
