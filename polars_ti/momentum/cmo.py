# -*- coding: utf-8 -*-
# =============================================================================
# Polars CMO Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils import v_pos_int
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_cmo(close, length, drift, scalar):
    """Wilder-smoothed Chande Momentum Oscillator, matching TA-Lib CMO.

    TA-Lib smooths the up/down moves with Wilder's running average (the same
    smoothing as RSI; ``CMO == 2*RSI - 100``) rather than a flat rolling sum.
    The seed at index ``drift + length - 1`` is the simple average of the first
    ``length`` moves; subsequent bars use ``avg[i] = avg[i-1]*(length-1)/length
    + move[i]/length``.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < length + drift:
        return out

    up = np.zeros(n)
    dn = np.zeros(n)
    for i in range(drift, n):
        change = close[i] - close[i - drift]
        if change > 0:
            up[i] = change
        elif change < 0:
            dn[i] = -change

    seed = drift + length - 1
    up_sum = 0.0
    dn_sum = 0.0
    for i in range(drift, seed + 1):
        up_sum += up[i]
        dn_sum += dn[i]
    avg_up = up_sum / length
    avg_dn = dn_sum / length
    total = avg_up + avg_dn
    if total != 0:
        out[seed] = scalar * (avg_up - avg_dn) / total

    alpha = 1.0 / length
    for i in range(seed + 1, n):
        avg_up = avg_up * (1 - alpha) + up[i] * alpha
        avg_dn = avg_dn * (1 - alpha) + dn[i] * alpha
        total = avg_up + avg_dn
        if total != 0:
            out[i] = scalar * (avg_up - avg_dn) / total

    return out


def cmo(
    close: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    talib: bool = True,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Chande Momentum Oscillator (CMO)

    Measures momentum with overbought at 50 and oversold at -50.
    CMO = scalar * (sum_gains - sum_losses) / (sum_gains + sum_losses)

    Sources:
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/chande-momentum-oscillator-cmo/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling period. Default: 14
        scalar: Multiplication factor. Default: 100
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        drift: Difference period for momentum. Default: 1
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CMO expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    length = v_pos_int(length, "length")
    drift = v_pos_int(drift, "drift")
    # TA-Lib CMO hardcodes drift=1; the native _nb_cmo kernel honors drift, so
    # fall through to it for a non-default drift (scalar is already rescaled below).
    _use_talib = Imports["talib"] and v_talib(talib) and drift == 1

    if _use_talib:
        _length = length
        _scalar = scalar

        def compute_cmo_talib(s: pl.Series) -> pl.Series:
            from talib import CMO as TALIB_CMO

            arr = s.to_numpy().astype(np.float64)
            # TA-Lib CMO hardcodes scalar=100; rescale so user ``scalar`` takes
            # effect. At scalar=100 this is an exact *1.0 no-op.
            result = TALIB_CMO(arr, timeperiod=_length) * (_scalar / 100.0)
            return pl.Series(result)

        cmo_expr = close_expr.map_batches(compute_cmo_talib, return_dtype=pl.Float64)
    else:
        # Native path: Wilder-smoothed CMO, matching TA-Lib (the OLD native path
        # used a flat rolling sum, which diverged from TA-Lib by tens of points).
        _length = length
        _scalar = scalar
        _drift = drift

        def compute_cmo_native(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            return pl.Series(_nb_cmo(arr, _length, _drift, _scalar))

        cmo_expr = close_expr.map_batches(compute_cmo_native, return_dtype=pl.Float64)

    if offset != 0:
        cmo_expr = cmo_expr.shift(offset)

    return cmo_expr.alias(f"CMO_{length}")
