"""Reproduce warmed baseline/current timings and numerical deviations."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from scripts.rolling_kernel_audit.cases import CASES, Case, Output


def flatten(output: Output) -> np.ndarray:
    """Flatten scalar and multi-output kernel results without dropping NaNs."""
    return np.concatenate([item.ravel() for item in output]) if isinstance(output, tuple) else output.ravel()


def deviation(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    """Report absolute, relative, and non-near-zero relative differences."""
    finite = np.isfinite(actual) & np.isfinite(expected)
    actual_finite = actual[finite]
    expected_finite = expected[finite]
    errors = np.abs(actual_finite - expected_finite)
    scale = np.abs(expected_finite)
    nonzero = scale > 0
    threshold = float(np.median(scale)) * 0.01 if len(scale) else 0.0
    substantial = scale > threshold
    return {
        "max_absolute": float(np.max(errors, initial=0)),
        "max_relative": float(np.max(errors[nonzero] / scale[nonzero], initial=0)),
        "relative_above_1pct_median": float(np.max(errors[substantial] / scale[substantial], initial=0)),
        "nan_mask_mismatches": int(np.count_nonzero(np.isnan(actual) != np.isnan(expected))),
        "infinity_mismatches": int(np.count_nonzero(np.isinf(actual) != np.isinf(expected))),
        "finite_compared": int(finite.sum()),
    }


def measure(
    case: Case, values: np.ndarray, window: int, repetitions: int
) -> dict[str, float | int | list[list[float]] | dict[str, float | int]]:
    """Alternate baseline/current timing after compiling both on identical input."""
    args = case.arguments(values, window)
    baseline = case.kernel(True)
    current = case.kernel(False)
    expected = flatten(baseline(*args))
    actual = flatten(current(*args))
    timings: list[list[float]] = [[], []]
    for repeat in range(repetitions):
        for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
            function = baseline if index == 0 else current
            start = time.perf_counter_ns()
            function(*args)
            timings[index].append((time.perf_counter_ns() - start) / 1e6)
    old_ms, new_ms = [float(np.median(samples)) for samples in timings]
    return {
        "window": window,
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speed_improvement_percent": 100 * (1 - new_ms / old_ms),
        "speedup": old_ms / new_ms,
        "samples_ms": timings,
        "deviation": deviation(actual, expected),
    }


def main() -> None:
    """Write machine-readable evidence; no research run is started."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark.json"))
    args = parser.parse_args()
    values = 60648 + np.random.default_rng(20260919).normal(scale=100, size=args.rows).cumsum()
    windows = [40, 480, 960]
    records = []
    for case in CASES:
        measurements = [measure(case, values, window, args.repetitions) for window in windows]
        exponents = {
            label: float(np.polyfit(np.log(windows), np.log([row[label + "_ms"] for row in measurements]), 1)[0])
            for label in ["old", "new"]
        }
        records.append({"kernel": case.name, "measurements": measurements, "exponents": exponents})
        print(case.name, f"{measurements[-1]['speed_improvement_percent']:.3f}%", flush=True)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {name: version(name) for name in ["numpy", "polars", "numba", "scipy", "TA-Lib"]},
        "rows": args.rows,
        "seed": 20260919,
        "repetitions": args.repetitions,
        "kernels": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
