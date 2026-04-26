# RF Envelope Calibration

## Scope

`rf_envelope` is a cycle-averaged HF/LF input model. It is useful when the
global plasma model should see recipe-scale absorbed power, RMS voltage,
self-bias, and an optional reduced-field proxy without resolving the RF period.

External circuit coupling is intentionally held at the future-entry stage:
`external_circuit_table` can already read measured or SPICE-generated waveform
CSV files one-way. Full ngspice execution, plasma-to-circuit feedback, and
tight circuit/plasma DAE coupling should remain future work until calibrated
RF-envelope and loose-coupling workflows are stable.

## Calibration Parameters

### `coupling_efficiency`

Use this to convert generator or delivered RF power into plasma absorbed power.

```text
coupling_efficiency = P_absorbed / P_commanded
```

Initial ranges:

| Port role | Suggested starting range |
|---|---:|
| HF source / ICP-like source | `0.30` to `0.90` |
| LF bias / CCP-like bias | `0.05` to `0.60` |

Start from measured absorbed power if available. If only forward/reflected
power is available, use the net delivered power as the command and tune the
efficiency against electron density, mean energy, or total ionization balance.

### `effective_impedance_ohm`

Use this for voltage-driven ports:

```text
P_commanded = V_rms^2 / effective_impedance_ohm
I_rms = V_rms / effective_impedance_ohm
```

Recommended first estimate:

```text
effective_impedance_ohm = V_rms / I_rms
```

or, when current is unavailable:

```text
effective_impedance_ohm = V_rms^2 / P_delivered
```

Suggested starting ranges:

| Port role | Suggested starting range |
|---|---:|
| HF source / ICP-like source | `5` to `500` Ohm |
| LF bias / CCP-like bias | `10` to `1000` Ohm |

Values outside these ranges can be valid, but should be treated as a sign that
the port definition, RMS convention, or delivered-power estimate needs review.
Validation warns when a voltage is supplied without an explicit
`effective_impedance_ohm`, because the backend otherwise uses `50 Ohm` only as
a convenience default for current reporting.

### `self_bias_fraction`

For LF/bias ports the backend uses:

```text
self_bias_V = -self_bias_fraction * sqrt(2) * voltage_rms_V
```

Recommended estimate from a measured DC self-bias:

```text
self_bias_fraction = abs(Vdc_self_bias) / (sqrt(2) * voltage_rms_V)
```

A practical starting range is `0.10` to `0.80`. The default `0.35` is only a
placeholder and should be replaced for process interpretation.

### Reduced-Field Proxy

The reduced-field proxy is optional and mainly matters for local-field EEDF
studies:

```text
EoverN_Td = base_reduced_field_Td
          + reduced_field_per_sqrt_W_Td * sqrt(P_absorbed_W)
```

For the default global ODE path, electron-impact rates are tied to the evolved
mean electron energy. Treat this proxy as a diagnostic or local-field study
input, not as an independent substitute for electron energy balance.

## Recommended Calibration Sequence

1. Set the HF source `value_W` from generator or delivered power.
2. Choose `coupling_efficiency` to match absorbed power or a primary plasma
   target such as electron density.
3. For LF bias, compute `effective_impedance_ohm` from measured `V_rms/I_rms`
   or `V_rms^2/P_delivered`.
4. Compute `self_bias_fraction` from measured DC self-bias if available.
5. Only then adjust reduced-field proxy coefficients for local-field studies.
6. Check `summary.yaml` and `observables.csv` columns such as
   `port_source_rf_absorbed_power_W`, `port_wafer_bias_voltage_rms_V`,
   `port_wafer_bias_current_rms_A`, and `port_wafer_bias_self_bias_V`.

## Coefficient Estimation Helper

Use `scripts/calibrate_rf_envelope.py` to convert measured RF quantities into
recipe coefficients.

HF source example:

```bash
python scripts/calibrate_rf_envelope.py \
  --role hf_source \
  --frequency-Hz 13.56e6 \
  --forward-power-W 1000 \
  --reflected-power-W 100 \
  --absorbed-power-W 540
```

LF bias example:

```bash
python scripts/calibrate_rf_envelope.py \
  --role lf_bias \
  --frequency-Hz 2.0e6 \
  --commanded-power-W 200 \
  --absorbed-power-W 40 \
  --voltage-rms-V 100 \
  --current-rms-A 0.5 \
  --dc-self-bias-V -49.5
```

The script emits both estimated coefficients and a compact recipe-port
snippet. It does not decide whether the measurements are physically valid; use
the warnings and the ranges above as review prompts.

## Backend Choice

| Need | Prefer | Reason |
|---|---|---|
| Fixed absorbed power for smoke tests or fast sweeps | `direct_power` | Minimal assumptions; no circuit interpretation |
| DC or pulsed DC with voltage-current feedback | `dc_series_circuit` | Captures source voltage, ballast resistance, plasma resistance, and current |
| Recipe-scale HF/LF with calibrated power, voltage, current, and self-bias | `rf_envelope` | Cycle-averaged and explicit about calibration coefficients |
| ICP/CCP proxy with simple built-in source/bias heuristics | `icp` / `ccp` | Convenient exploratory proxy, but coefficients are less explicit |
| Measured or SPICE-generated waveform imposed on the plasma | `external_circuit_table` | One-way comparison input; no plasma-to-circuit feedback |

For production-style HF/LF studies, prefer `rf_envelope` when calibration data
are available. Prefer `icp` / `ccp` when exploring trends before calibration.

## Executable Example

The calibration-oriented pure-Ar example is:

```bash
plasma-global validate examples/configs/case_rf_envelope_calibration.yaml
plasma-global run examples/configs/case_rf_envelope_calibration.yaml
```

The case uses public Ar chemistry and explicit HF/LF parameters:

- `examples/configs/case_rf_envelope_calibration.yaml`
- `examples/configs/recipe_rf_envelope_calibration.yaml`

It is not a validated process recipe by itself. It is a reviewable starting
point for tuning RF-envelope coefficients against measured or trusted reference
power, voltage, current, self-bias, and plasma observables.
