"""Expression-level parity and timing for callback and composition rewrites."""

import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

from scripts.rolling_kernel_audit.benchmark import deviation
from scripts.rolling_kernel_audit.reference import original

NAMES = [
    "alma",
    "fwma",
    "pwma",
    "sinwma",
    "swma",
    "kama",
    "fisher",
    "kdj",
    "increasing",
    "decreasing",
    "variance",
    "stdev",
    "sma",
]


def evaluate(name: str, frame: pl.DataFrame, window: int, baseline: bool, drift: int = 1) -> np.ndarray:
    """Evaluate the public API against frozen or current source."""
    category = (
        "momentum" if name in {"fisher", "kdj"} else "trend" if name in {"increasing", "decreasing"} else "overlap"
    )
    if name in {"variance", "stdev"}:
        category = "statistics"
    module = original(category + "." + name) if baseline else import_module("polars_ti." + category + "." + name)
    function = getattr(module, name)
    if name == "fisher":
        expression = function("high", "low", length=window)
    elif name == "kdj":
        expression = function("high", "low", "close", length=window)
    elif name in {"kama", "variance", "stdev", "sma"}:
        expression = function("close", length=window, talib=False)
    elif name in {"increasing", "decreasing"}:
        expression = function("close", length=window, strict=True, drift=drift)
    else:
        expression = function("close", length=window)
    return frame.select(expression).to_numpy().astype(float).ravel()


def main() -> None:
    """Record public-API timing including callback/Polars overhead."""
    values = 60648 + np.random.default_rng(20260919).normal(scale=100, size=8192).cumsum()
    frame = pl.DataFrame({"close": values, "high": values + 2, "low": values - 2})
    rows = []
    for name in NAMES:
        measurements = []
        for window in [40, 480, 960]:
            expected = evaluate(name, frame, window, True)
            actual = evaluate(name, frame, window, False)
            timings = [[], []]
            for repeat in range(5):
                for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    start = time.perf_counter_ns()
                    evaluate(name, frame, window, index == 0)
                    timings[index].append((time.perf_counter_ns() - start) / 1e6)
            old_ms, new_ms = [float(np.median(values)) for values in timings]
            measurements.append(
                {
                    "window": window,
                    "old_ms": old_ms,
                    "new_ms": new_ms,
                    "speed_improvement_percent": 100 * (1 - new_ms / old_ms),
                    "samples_ms": timings,
                    "deviation": deviation(actual, expected),
                }
            )
        rows.append(
            {
                "kernel": name,
                "measurements": measurements,
                "exponents": {
                    label: float(
                        np.polyfit(np.log([40, 480, 960]), np.log([row[label + "_ms"] for row in measurements]), 1)[0]
                    )
                    for label in ["old", "new"]
                },
            }
        )
        print(name, measurements[-1]["speed_improvement_percent"], flush=True)
    Path(__file__).with_name("expressions.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
