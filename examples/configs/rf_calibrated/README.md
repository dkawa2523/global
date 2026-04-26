# RF Envelope Calibrated Case Library

This directory is reserved for RF-envelope cases that have explicit calibration
targets. A case placed here should state the measured or trusted reference used
to choose:

- `coupling_efficiency`
- `effective_impedance_ohm`
- `self_bias_fraction`
- reduced-field proxy coefficients

Use `../case_rf_envelope_calibration.yaml` as the current runnable starting
point. Promote a case into this directory only after its README or metadata
records the pressure, gas, frequency, power/voltage range, calibration target,
and known invalid regimes.
