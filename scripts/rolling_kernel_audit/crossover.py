"""Measure order-statistics crossover beyond the bounded CRSI scan."""

import json
from pathlib import Path

import numpy as np

from scripts.rolling_kernel_audit.benchmark import measure
from scripts.rolling_kernel_audit.cases import Case


def main() -> None:
    """Write paired large-window timings and unchanged-reference deviations."""
    values = 60648 + np.random.default_rng(20260919).normal(scale=100, size=32768).cumsum()
    records = []
    for case in [Case("momentum.crsi", "nb_percent_rank"), Case("statistics.mad", "nb_mad")]:
        records.append(
            {
                "kernel": case.name,
                "rows": len(values),
                "measurements": [measure(case, values, window, 5) for window in [1024, 2048, 8192]],
            }
        )
    Path(__file__).with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
