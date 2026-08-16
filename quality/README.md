# Quality exceptions

Ordinary quality commands never rewrite tracked exception files. Ruff and
Vulture must remain clean, coverage.py's combined statement-and-branch metric
has a fixed 85% floor, changed production lines require 90% coverage, and new
code has a cyclomatic-complexity limit of 10.

`baseline.json` contains only the reviewed Bandit findings in the trusted
quality runner and the small set of existing complexity exceptions above 10.
The intentionally redundant boundary conversions are disabled once in the
Pyrefly configuration. Detect-secrets keeps its native reviewed baseline
because that format carries review identity.

The policy diff check is intentionally syntactic: it rejects newly added
suppression, secret-allowlist, and direct skip/xfail syntax. Test semantics are
enforced by execution, coverage, mutation checks, and review rather than a
home-grown AST implication engine.

Run repository gates through the single development runner:

```text
python -m tools.quality fast
python -m tools.quality pr
python -m tools.quality baseline
python -m tools.quality nightly --native
```

Run `baseline` only after reviewing every remaining exception. The command
first runs formatting, lint, import architecture, dependency audit, tests,
coverage, complexity, and dead-code detection. Pull-request CI rejects new
Bandit findings, new or worsened complexity exceptions, and a larger secret
baseline.
