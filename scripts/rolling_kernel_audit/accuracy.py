"""Measure errors against independent math.fsum two-pass references."""

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from polars_ti.overlap.linreg import nb_linreg
from polars_ti.statistics.kurtosis import nb_kurtosis
from polars_ti.statistics.mad import nb_mad
from polars_ti.statistics.skew import nb_skew
from polars_ti.statistics.variance import variance
from polars_ti.statistics.zscore import nb_zscore
from scripts.rolling_kernel_audit.benchmark import deviation
from scripts.rolling_kernel_audit.reference import original


def exact_statistics(segment: np.ndarray) -> dict[str, float]:
    """Evaluate each definition with accurate summation, independent of rolling code."""
    count = len(segment)
    mean = math.fsum(segment) / count
    centered = [float(value) - mean for value in segment]
    second = math.fsum(value * value for value in centered) / count
    third = math.fsum(value**3 for value in centered) / count
    fourth = math.fsum(value**4 for value in centered) / count
    slope = math.fsum((j + 1 - (count + 1) / 2) * value for j, value in enumerate(centered)) / (
        count * (count**2 - 1) / 12
    )
    return {
        "variance": second,
        "zscore": centered[-1] / math.sqrt(second) if second > 0 else math.nan,
        "mad": math.fsum(abs(value) for value in centered) / count,
        "skew": math.sqrt(count * (count - 1)) / (count - 2) * third / second**1.5
        if count > 2 and second > 0
        else math.nan,
        "kurtosis": ((count + 1) * (fourth / second**2 - 3) + 6) * (count - 1) / ((count - 2) * (count - 3))
        if count > 3 and second > 0
        else math.nan,
        "linreg_slope": slope,
        "linreg": mean + slope * (count - (count + 1) / 2),
    }


def main() -> None:
    """Write per-class old/new errors, including a long-series drift experiment."""
    random = np.random.default_rng(808)
    corpora = {
        "prices": 60648 + random.normal(scale=100, size=4096).cumsum(),
        "tiny_spread": 1e9 + random.normal(scale=1e-4, size=4096),
        "changing_scale": np.concatenate((np.full(1200, 1e12), 60648 + random.normal(size=2896))),
        "long_series": 60648 + random.normal(scale=100, size=120000).cumsum(),
    }
    records = []
    for label, values in corpora.items():
        for window in [40, 480, 960, 23040] if label == "long_series" else [5, 40, 480, 960]:
            indices = np.unique(np.linspace(window - 1, len(values) - 1, 256, dtype=int))
            expected_rows = [exact_statistics(values[index - window + 1 : index + 1]) for index in indices]
            for name, current in [
                ("variance", None),
                ("zscore", nb_zscore),
                ("mad", nb_mad),
                ("skew", nb_skew),
                ("kurtosis", nb_kurtosis),
                ("linreg", nb_linreg),
                ("linreg_slope", nb_linreg),
            ]:
                expected = np.array([row[name] for row in expected_rows])
                if name == "variance":
                    frame = pl.DataFrame({"x": values})
                    old = (
                        frame.select(pl.col("x").rolling_var(window_size=window, min_samples=window, ddof=0))
                        .to_series()
                        .to_numpy()[indices]
                    )
                    new = frame.select(variance("x", window, talib=False)).to_series().to_numpy()[indices]
                elif name.startswith("linreg"):
                    flags = (False, False, False, False, name == "linreg_slope", False)
                    old = original("overlap.linreg").nb_linreg(values, window, *flags)[indices]
                    new = nb_linreg(values, window, *flags)[indices]
                else:
                    extras = (1.0,) if name == "zscore" else ()
                    # Long-window baseline comparisons need only the sampled windows.
                    baseline = getattr(original("statistics." + name), "nb_" + name)
                    old = np.array(
                        [baseline(values[index - window + 1 : index + 1], window, *extras)[-1] for index in indices]
                    )
                    new = current(values, window, *extras)[indices]
                records.append(
                    {
                        "kernel": name,
                        "corpus": label,
                        "rows": len(values),
                        "window": window,
                        "sampled_outputs": len(indices),
                        "old_error": deviation(old, expected),
                        "new_error": deviation(new, expected),
                    }
                )
            print(label, window, flush=True)
    Path(__file__).with_name("accuracy.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
