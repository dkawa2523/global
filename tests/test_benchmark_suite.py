from __future__ import annotations

from scripts.run_benchmark_suite import evaluate_expectations


def test_benchmark_expectation_ranges_and_targets() -> None:
    report = evaluate_expectations(
        {'ne': 2.0, 'power': 10.0},
        {
            'ne': {'min': 1.0, 'max': 3.0},
            'power': {'target': 10.1, 'rtol': 0.02},
        },
    )

    assert report['passed'] is True
    assert report['checks']['ne']['passed'] is True
    assert report['checks']['power']['passed'] is True
