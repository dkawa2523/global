# LoKI O2 DC Glow Digitization Scaffold

This directory is an external-only scaffold for digitizing the public
LoKI O2 DC glow benchmark paper.

Rules:
- Keep CSV headers unchanged.
- Put raw digitized points into the provided CSV files.
- Do not mix units inside one file.
- Keep 1D and 0D series separated with `series_id`.
- Run `py tools\external_benchmarks\loki_o2_dc_glow_digitization.py`
  after editing files to validate headers and readiness.

This scaffold does not depend on `plasma_global` core modules.
