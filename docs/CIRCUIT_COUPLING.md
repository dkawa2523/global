# Circuit Coupling

## Design Goal

Electrical circuits are intentionally separated from plasma chemistry,
transport, and surface models. The plasma solver talks to electrical models
through `PowerRequest` and `PowerResult`; circuit-specific equations live below
the electrical backend layer.

This keeps future circuit work local:

- internal DC, pulsed, RC, or RF-envelope models can be changed inside
  `plasma_global.electrical`,
- plasma chemistry does not need to know whether power came from a fixed power
  command, an internal circuit model, or an external SPICE waveform,
- future external-circuit coupling can reuse the same port metadata and
  `PowerResult` fields.

## Current Circuit Backends

Use `dc_series_circuit` for a reduced DC or pulsed-DC voltage source with a
ballast resistor and a conductive plasma load.

```yaml
physics:
  electrical_backend: dc_series_circuit
```

Each driven port should provide:

```yaml
power_ports:
  dc_drive:
    mode: voltage_source
    zone_id: plasma
    voltage:
      waveform: pulsed_square
      high_V: 1000.0
      low_V: 0.0
      duty_cycle: 0.5
      frequency_Hz: 1.0e4
    ballast_resistance_ohm: 1.0e5
    gap_m: 0.004
    electrode_area_m2: 5.0e-5
```

Required parameters:

| Key | Meaning |
|---|---|
| `source_voltage_V` | Open-circuit source voltage before the ballast resistor |
| `ballast_resistance_ohm` | Series resistor between source and plasma |
| `gap_m` | Electrode or effective plasma-load gap |
| `electrode_area_m2` | Effective current-carrying area |

`source_voltage_V` may be supplied as a flat legacy key, or through the nested
`voltage` block shown above. The nested form is preferred for pulsed-DC cases
because it keeps `high_V`, `low_V`, duty, and frequency together.

Optional parameters:

| Key | Default | Meaning |
|---|---:|---|
| `electron_mobility_m2_V_s` | pressure-scaled estimate | Constant electron mobility for the load model |
| `mobility_ref_m2_V_s` | `0.10` | Reference mobility used when no constant mobility is supplied |
| `mobility_ref_pressure_Pa` | `133.322` | Reference pressure for inverse-pressure mobility scaling |
| `power_absorption_fraction` | `1.0` | Fraction of conductive plasma power deposited into electron energy |
| `min_plasma_resistance_ohm` | `1.0e-6` | Lower clamp on plasma resistance |
| `max_plasma_resistance_ohm` | `1.0e12` | Upper clamp on plasma resistance |

Supported waveforms currently mirror the lightweight direct-power controls:

- `cw`
- `off`
- `pulsed_square`

The model computes:

```text
G_plasma = e * ne * mu_e * electrode_area_m2 / gap_m
R_plasma = 1 / G_plasma
I = V_source / (ballast_resistance_ohm + R_plasma)
V_gap = I * R_plasma
P_abs = power_absorption_fraction * V_gap * I
E/N = abs(V_gap / gap_m) / N
```

It does not solve sheath capacitance, RLC ringing, matching networks, or a
SPICE DAE. Treat it as a transparent reduced circuit model.

Use `external_circuit_table` when the circuit solution is supplied by a
measurement, ngspice, or another external tool and the plasma should be driven
one-way by that waveform.

```yaml
files:
  external_inputs:
    circuit_result_csv: ../external/circuit/ngspice_result.csv

physics:
  electrical_backend: external_circuit_table
```

```yaml
power_ports:
  circuit_waveform:
    zone_id: plasma
    file_key: circuit_result_csv
    power_column: absorbed_power_W
    voltage_column: voltage_V
    current_column: current_A
    reduced_field_column: reduced_field_Td
    interpolation: linear
    hold: edge
```

The CSV must contain `time_s` and either:

- a power column such as `absorbed_power_W`, `power_W`, or `plasma_power_W`, or
- both `voltage_V` and `current_A`, where `P = V * I` is used as the deposited
  power.

Optional `reduced_field_Td` drives local-field EEDF studies. This backend does
not update the external circuit from the evolving plasma impedance; it is a
loose one-way interface.

Use `rf_envelope` for cycle-averaged HF/LF settings where the global model
should see an RF-period-averaged absorbed power and optional bias proxy.

```yaml
physics:
  electrical_backend: rf_envelope
```

```yaml
power_ports:
  hf_source:
    role: hf_source
    zone_id: source
    frequency_Hz: 13.56e6
    value_W: 300.0
    coupling_efficiency: 0.65
    base_reduced_field_Td: 25.0
    reduced_field_per_sqrt_W_Td: 1.0
  lf_bias:
    role: lf_bias
    zone_id: process
    frequency_Hz: 400.0e3
    voltage_rms_V: 100.0
    effective_impedance_ohm: 50.0
    coupling_efficiency: 0.25
    self_bias_fraction: 0.35
```

Required RF-envelope parameters:

| Key | Meaning |
|---|---|
| `frequency_Hz` | Carrier frequency represented by the cycle-averaged envelope |
| `value_W` / `power_W` / `absorbed_power_W` | Power-driven command |
| `voltage_rms_V` / `voltage_V` / `value_V` | Voltage-driven command |
| `effective_impedance_ohm` | Required when only voltage is supplied |

Optional controls include `coupling_efficiency`, `waveform: pulsed_square`,
`self_bias_fraction`, `base_reduced_field_Td`,
`reduced_field_per_sqrt_W_Td`, and plasma-potential proxy coefficients. This
model intentionally does not resolve the RF period, sheath motion, matching
network, or harmonic content.
For coefficient ranges and a calibration sequence, see
`RF_ENVELOPE_CALIBRATION.md`.
That document also includes a backend choice table and a helper script for
estimating `rf_envelope` coefficients from measured RF quantities.

## Output Diagnostics

Backends write circuit details to `PowerResult.metadata["port_details"]`.
Depending on the backend these include:

- `source_voltage_V`
- `gap_voltage_V`
- `current_A`
- `absorbed_power_W`
- `delivered_power_W`
- `ballast_resistance_ohm`
- `plasma_resistance_ohm`
- `plasma_conductance_S`
- `electron_mobility_m2_V_s`
- `electric_field_V_m`
- `reduced_field_Td`
- `frequency_Hz`
- `voltage_rms_V`
- `current_rms_A`
- `self_bias_V`

Numeric port details are also flattened into `observables.csv` with the
pattern `port_<port_id>_<metric>`, for example
`port_dc_drive_gap_voltage_V` or `port_lf_bias_self_bias_V`.
The compact `summary.yaml` keeps final and per-step statistics for the main
port quantities only, so detailed backend-specific diagnostics remain in the
time-series CSV.

## Future External-Circuit Entry

The reserved configuration path for external circuit inputs is
`files.external_inputs`. Recommended future keys are:

```yaml
files:
  external_inputs:
    circuit_netlist: ../external/circuit/example.cir
    circuit_waveform_csv: ../external/circuit/waveform.csv
    circuit_result_csv: ../external/circuit/ngspice_result.csv
```

Recommended CSV columns for loose coupling:

```text
time_s, voltage_V, current_A, absorbed_power_W
```

Recommended implementation stages:

1. read measured or SPICE-generated waveform/result CSV and drive the plasma
   one-way (implemented as `external_circuit_table`),
2. add internal reduced circuit models such as RC/RLC or RF envelopes
   (`rf_envelope` is the current HF/LF envelope entry),
3. add loose SPICE coupling over macro time windows using plasma-derived
   equivalent load parameters,
4. only then consider tight plasma-ODE/circuit-DAE co-simulation.

This staged path keeps the default solver lightweight while preserving a clear
route to detailed external circuit models.
For now, this is intentionally the stopping point for external circuits: the
code has a one-way table interface and reserved paths, but does not execute
ngspice or solve a coupled circuit/plasma DAE.
