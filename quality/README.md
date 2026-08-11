# Quality baselines

The tracked baselines separate accepted legacy debt from regressions. Ordinary
quality commands never rewrite these files.

Run `uv run quality-baseline` only after reviewing every new diagnostic. The
command first requires tests, architecture checks, dependency audit, and all
non-baselinable checks to pass. Pull-request CI also compares baseline changes
with the target branch and rejects weaker thresholds or newly accepted debt.

The initial repository measurement was 195 passing tests, 80.739627% branch
coverage, 735 raw Ruff findings, 56 Pyrefly errors, 47 functions over complexity
10, 25 Bandit findings, no vulnerable dependencies, one audited secret-scan
false positive, and no Vulture candidates at 80% confidence.
