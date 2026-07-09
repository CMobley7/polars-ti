# -*- coding: utf-8 -*-
# =============================================================================
# Polars Aroon Implementation
# =============================================================================
import numpy as np
from numba import njit
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_aroon(high: np.ndarray, low: np.ndarray, length: int, scalar: float):
    """Numba kernel for Aroon Up, Down, and Oscillator."""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    aroon_osc = np.full(n, np.nan)
    window = length + 1

    for i in range(window - 1, n):
        # Find periods since highest high and lowest low
        max_idx = 0
        min_idx = 0
        max_val = high[i - window + 1]
        min_val = low[i - window + 1]
        for j in range(1, window):
            idx = i - window + 1 + j
            if high[idx] >= max_val:
                max_val = high[idx]
                max_idx = j
            if low[idx] <= min_val:
                min_val = low[idx]
                min_idx = j

        periods_from_hh = (window - 1) - max_idx
        periods_from_ll = (window - 1) - min_idx

        aroon_up[i] = scalar * (1.0 - periods_from_hh / length)
        aroon_down[i] = scalar * (1.0 - periods_from_ll / length)
        aroon_osc[i] = aroon_up[i] - aroon_down[i]

    return aroon_up, aroon_down, aroon_osc


def aroon(
    high: IntoExpr,
    low: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Aroon & Aroon Oscillator

    Identifies if a security is trending and how strong.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        length: Period. Default: 14
        scalar: Magnification. Default: 100
        talib: If True and TA-Lib is installed, use ``talib.AROON``/``AROONOSC``.
            The native path matches TA-Lib to float noise. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with AROONU, AROOND, AROONOSC columns
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    _use_talib = Imports["talib"] and v_talib(talib)

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_high"].to_numpy().astype(np.float64)
        l_ = data["_low"].to_numpy().astype(np.float64)
        if _use_talib:
            from talib import AROON as _AROON, AROONOSC as _AROONOSC

            # TA-Lib AROON hardcodes scalar=100; rescale so ``scalar`` is honoured
            # (exact *1.0 no-op at the default). Returns (down, up).
            down, up = _AROON(h, l_, length)
            osc = _AROONOSC(h, l_, length)
            if scalar != 100.0:
                _f = scalar / 100.0
                up, down, osc = up * _f, down * _f, osc * _f
        else:
            up, down, osc = _nb_aroon(h, l_, length, scalar)

        if offset != 0:
            up = np.roll(up, offset)
            down = np.roll(down, offset)
            osc = np.roll(osc, offset)
            if offset > 0:
                up[:offset] = np.nan
                down[:offset] = np.nan
                osc[:offset] = np.nan
            else:
                up[offset:] = np.nan
                down[offset:] = np.nan
                osc[offset:] = np.nan

        return pl.Series(values=[{"AROONU": u, "AROOND": d, "AROONOSC": o} for u, d, o in zip(up, down, osc)])

    fields = [
        pl.Field("AROONU", pl.Float64),
        pl.Field("AROOND", pl.Float64),
        pl.Field("AROONOSC", pl.Float64),
    ]
    return (
        pl.struct(
            high_expr.alias("_high"),
            low_expr.alias("_low"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"AROON_{length}")
    )
