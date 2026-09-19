# -*- coding: utf-8 -*-
# =============================================================================
# Polars Aroon Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._rolling import rolling_extreme
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_aroon(high: np.ndarray, low: np.ndarray, length: int, scalar: float):
    """Calculate Aroon with newest ties and no indexing of invalid sentinels."""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    _, high_index = rolling_extreme(high, length + 1, True)
    _, low_index = rolling_extreme(low, length + 1, False)
    for i in range(length, n):
        if high_index[i] >= 0:
            aroon_up[i] = scalar * (1.0 - (i - high_index[i]) / length)
        if low_index[i] >= 0:
            aroon_down[i] = scalar * (1.0 - (i - low_index[i]) / length)
    return aroon_up, aroon_down, aroon_up - aroon_down


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
            from talib import AROON as _AROON
            from talib import AROONOSC as _AROONOSC

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
