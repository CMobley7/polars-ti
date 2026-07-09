# -*- coding: utf-8 -*-
# =============================================================================
# Polars MAVP (Moving Average with Variable Period) Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_mavp(
    close: np.ndarray,
    periods: np.ndarray,
    min_period: int,
    max_period: int,
) -> np.ndarray:
    """Numba kernel: per-bar Simple Moving Average with a variable period.

    Matches TA-Lib ``MAVP`` (matype=0): the per-bar period is the truncated,
    clamped value from ``periods``, and every output before ``max_period - 1``
    is NaN regardless of that bar's own (possibly small) period.

    NaN period handling mirrors TA-Lib: casting a NaN period to ``int`` yields a
    value below ``min_period``, which the clamp then pins to ``min_period``.

    Known divergence (deferred): TA-Lib evaluates each distinct period via a
    running-sum SMA, so a NaN in ``close`` poisons every later output that shares
    that period bucket. This kernel uses a fresh per-bar window instead, so a
    close-NaN only affects the windows that actually span it. The two agree on
    realistic (NaN-free) data; the divergence is a degenerate-input edge.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    for i in range(max_period - 1, n):
        p = periods[i]
        # TA-Lib truncates the fractional period, then clamps to [min, max]. A
        # NaN period casts to a value below min_period, so it clamps to min.
        if np.isnan(p):
            period = min_period
        else:
            period = int(p)
            if period < min_period:
                period = min_period
            elif period > max_period:
                period = max_period

        total = 0.0
        for j in range(i - period + 1, i + 1):
            total += close[j]
        out[i] = total / period

    return out


def mavp(
    close: IntoExpr,
    periods: IntoExpr,
    min_period: int = 2,
    max_period: int = 30,
    matype: int = 0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Moving Average with Variable Period (MAVP)

    A moving average whose window length varies per bar according to the
    ``periods`` input. The native implementation supports ``matype=0`` (Simple
    Moving Average) only; other ``matype`` values require TA-Lib.

    Args:
        close: Column name or pl.Expr for 'close' prices
        periods: Column name or pl.Expr giving the per-bar period
        min_period: Minimum period clamp. Default: 2
        max_period: Maximum period clamp. Default: 30
        matype: TA-Lib MA type. Default: 0 (SMA). Non-zero requires TA-Lib.
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MAVP expression aliased ``MAVP_{min_period}_{max_period}``

    Raises:
        ValueError: If ``matype != 0`` and TA-Lib is not available/enabled, as
            the native kernel only implements the Simple Moving Average.
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    periods_expr = v_expr(periods)
    if close_expr is None or periods_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _min = min_period
    _max = max_period
    _matype = matype

    if not _use_talib and matype != 0:
        raise ValueError(
            "mavp: native implementation supports matype=0 (SMA) only; enable TA-Lib for other matype values"
        )

    def compute_mavp(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        c = df["_c"].to_numpy().astype(np.float64)
        p = df["_p"].to_numpy().astype(np.float64)
        if _use_talib:
            from talib import MAVP as TALIB_MAVP

            return pl.Series(TALIB_MAVP(c, p, minperiod=_min, maxperiod=_max, matype=_matype))
        return pl.Series(_nb_mavp(c, p, _min, _max))

    struct_expr = pl.struct(close_expr.alias("_c"), periods_expr.alias("_p"))
    mavp_expr = struct_expr.map_batches(compute_mavp, return_dtype=pl.Float64)

    if offset != 0:
        mavp_expr = mavp_expr.shift(offset)

    return mavp_expr.alias(f"MAVP_{min_period}_{max_period}")
