"""Shared reproducible corpus definitions for regression and timing runs."""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scripts.rolling_kernel_audit.reference import original

Array: TypeAlias = NDArray[np.float64]
Argument: TypeAlias = Array | int | float | bool
Output: TypeAlias = Array | tuple[Array, ...]
Kernel: TypeAlias = Callable[..., Output]


@dataclass(frozen=True)
class Case:
    """A production kernel and reproducible argument construction rule."""

    module: str
    function: str
    kind: str = "single"
    extras: tuple[int | float | bool, ...] = ()
    changed_nan: bool = False

    @property
    def name(self) -> str:
        """Return the unique kernel label."""
        return self.module + "." + self.function + ("." + str(self.extras) if self.extras else "")

    def kernel(self, baseline: bool = False) -> Kernel:
        """Load the current or frozen implementation."""
        module = original(self.module) if baseline else import_module("polars_ti." + self.module)
        return cast(Kernel, getattr(module, self.function))

    def arguments(self, values: Array, window: int) -> tuple[Argument, ...]:
        """Construct deterministic arguments with positive prices and volumes."""
        if self.kind == "ohlc":
            return values + 2, values - 2, values, window, *self.extras
        if self.kind == "aroon":
            return values + 2, values - 2, window, 100.0
        if self.kind == "mfi":
            return values + 2, values - 2, values, np.abs(values * 3), window
        if self.kind == "tmo":
            return values - 0.2, values, window, *self.extras
        if self.kind == "massi":
            return values, 3, window
        if self.kind == "avsl":
            return values, np.full(len(values), 1.2), np.full(len(values), 1.1), np.ones(len(values)), 2.0, window
        if self.kind == "mavp":
            return values, (np.arange(len(values)) % window + 1).astype(float), 1, window
        if self.kind == "zigzag":
            return values + 2, values - 2, window
        return values, window, *self.extras


CASES = [
    Case("statistics.quantile", "nb_quantile", extras=(0.37,), changed_nan=True),
    Case("trend.trendflex", "nb_trendflex", extras=(20, 0.04, 3.14159, 1.414)),
    Case("cycles.reflex", "np_reflex", extras=(20, 0.04, 3.14159, 1.414)),
    Case("cycles.msw", "nb_msw", changed_nan=True),
    Case("trend.trama", "_nb_rolling_max", changed_nan=True),
    Case("trend.trama", "_nb_rolling_min", changed_nan=True),
    Case("trend.aroon", "_nb_aroon", kind="aroon", changed_nan=True),
    Case("momentum.stoch", "_stoch_rawk", kind="ohlc", changed_nan=True),
    Case("momentum.stoch", "_stoch_core", kind="ohlc", extras=(3, 3), changed_nan=True),
    Case("momentum.stochf", "_stochf_core", kind="ohlc", extras=(3,), changed_nan=True),
    Case("momentum.stochrsi", "_stochrsi_raw_core", extras=(5,)),
    Case("momentum.stc", "nb_schaff_tc", extras=(0.5,), changed_nan=True),
    Case("statistics.entropy", "nb_entropy", extras=(2.0,)),
    Case("statistics.zscore", "nb_zscore", extras=(1.0,)),
    Case("statistics.mad", "nb_mad"),
    Case("statistics.skew", "nb_skew"),
    Case("statistics.kurtosis", "nb_kurtosis"),
    Case("volume.mfi", "_nb_mfi", kind="mfi"),
    Case("volatility.massi", "nb_massi_from_ema1", kind="massi"),
    Case("volatility.avsl", "nb_avsl_core_logic", kind="avsl"),
    Case("momentum.crsi", "nb_percent_rank"),
    Case("momentum.tmo", "_signed_rolling_deltas_numba", kind="tmo", extras=(True,)),
    Case("momentum.tmo", "_signed_rolling_deltas_numba", kind="tmo", extras=(False,)),
    Case("overlap.sma", "nb_sma"),
    Case("overlap.wma", "nb_wma", extras=(True, True)),
    Case("overlap.wma", "nb_wma", extras=(False, True)),
    Case("momentum.cg", "nb_cg"),
    Case("overlap.mavp", "_nb_mavp", kind="mavp"),
    Case("volatility.rvi", "_rolling_std", extras=(1,)),
    Case("trend.zigzag", "nb_rolling_hl", kind="zigzag"),
]
CASES += [
    Case("overlap.linreg", "nb_linreg", extras=flags)
    for flags in [
        (False, False, False, False, False, False),
        (False, False, False, False, True, False),
        (False, False, True, False, False, False),
        (False, False, False, True, False, False),
        (True, True, False, False, False, False),
        (False, False, False, False, False, True),
    ]
]
