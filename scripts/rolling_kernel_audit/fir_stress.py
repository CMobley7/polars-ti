"""Benchmark zero-centered and outlier FIR inputs against the preserved SWMA."""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from polars_ti.overlap.swma import swma
from scripts.rolling_kernel_audit.benchmark import deviation
from scripts.rolling_kernel_audit.reference import original


def main() -> None:
    """Save paired timings and deviations for the review-discovered edge cases."""
    size = 20_000
    records = []
    for kind in ["positive", "cancellation", "outlier"]:
        values = np.ones(size)
        if kind == "cancellation":
            values[::2] = -1
        elif kind == "outlier":
            values[0] = 1e12
        frame = pl.DataFrame({"x": values})
        for window in [128, 512, 1024, 2048]:
            old_expr = original("overlap.swma").swma("x", length=window)
            new_expr = swma("x", length=window)
            old = frame.select(old_expr).to_series().to_numpy()
            new = frame.select(new_expr).to_series().to_numpy()
            samples = [[], []]
            for repeat in range(5):
                for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    start = time.perf_counter_ns()
                    frame.select(old_expr if index == 0 else new_expr)
                    samples[index].append((time.perf_counter_ns() - start) / 1e6)
            old_ms, new_ms = [float(np.median(sample)) for sample in samples]
            records.append(
                {
                    "corpus": kind,
                    "rows": size,
                    "window": window,
                    "old_ms": old_ms,
                    "new_ms": new_ms,
                    "speed_improvement_percent": 100 * (1 - new_ms / old_ms),
                    "samples_ms": samples,
                    "deviation": deviation(new, old),
                }
            )
        print(kind, flush=True)
    Path(__file__).with_name("fir_stress.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
