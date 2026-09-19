"""Reproduce FIR timing after the strict prefix-consistency correction."""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from scripts.rolling_kernel_audit.benchmark import deviation
from scripts.rolling_kernel_audit.expressions import evaluate


def main() -> None:
    """Compare public filters to the preserved pre-optimization source baseline."""
    values = 60648 + np.random.default_rng(20260919).normal(scale=100, size=8192).cumsum()
    frame = pl.DataFrame({"close": values})
    rows = []
    for name in ["alma", "fwma", "pwma", "sinwma", "swma"]:
        for window in [35, 128, 480, 960]:
            expected = evaluate(name, frame, window, True)
            actual = evaluate(name, frame, window, False)
            samples = [[], []]
            for repeat in range(5):
                for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    started = time.perf_counter_ns()
                    evaluate(name, frame, window, index == 0)
                    samples[index].append((time.perf_counter_ns() - started) / 1e6)
            old_ms, new_ms = [float(np.median(sample)) for sample in samples]
            rows.append(
                {
                    "indicator": name,
                    "window": window,
                    "old_ms": old_ms,
                    "new_ms": new_ms,
                    "time_reduction_percent": 100 * (1 - new_ms / old_ms),
                    "samples_ms": samples,
                    "deviation": deviation(actual, expected),
                }
            )
            print(name, window, rows[-1]["time_reduction_percent"], flush=True)
    Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
