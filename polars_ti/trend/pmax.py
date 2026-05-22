# -*- coding: utf-8 -*-
from numpy import full, nan, ones
from numba import njit


@njit(cache=True)
def nb_pmax(close, ub, lb):
    m = close.size
    dir_ = np.ones(m)  # 1 for Uptrend, -1 for Downtrend
    trend = np.full(m, np.nan)
    long = np.full(m, np.nan)
    short = np.full(m, np.nan)

    # State-dependent iteration
    for i in range(1, m):
        # Trend detection: price crosses bands
        if close[i] > lb[i - 1]:
            dir_[i] = 1  # Uptrend
        elif close[i] < ub[i - 1]:
            dir_[i] = -1  # Downtrend
        else:
            dir_[i] = dir_[i - 1]  # Maintain previous trend

            # Adjust bands to not move against trend
            if dir_[i] > 0 and ub[i] < ub[i - 1]:
                ub[i] = ub[i - 1]
            if dir_[i] < 0 and lb[i] > lb[i - 1]:
                lb[i] = lb[i - 1]

        # Set PMAX value based on trend direction
        if dir_[i] > 0:
            trend[i] = long[i] = ub[i]
        else:
            trend[i] = short[i] = lb[i]

    return trend, dir_, long, short


# =============================================================================
# Polars PMAX Implementation (reuses nb_pmax kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit as _njit_pmax

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@_njit_pmax(cache=True)
def _nb_pmax_atr(high, low, close, length):
    """Compute ATR array via RMA for PMAX (no pandas)."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    result = np.full(n, np.nan)
    s = 0.0
    for i in range(length):
        s += tr[i]
    result[length - 1] = s / length
    alpha = 1.0 / length
    for i in range(length, n):
        result[i] = alpha * tr[i] + (1 - alpha) * result[i - 1]
    return result


@_njit_pmax(cache=True)
def _nb_ema_raw(close, length):
    """Compute EMA array (no pandas)."""
    n = len(close)
    result = np.full(n, np.nan)
    s = 0.0
    for i in range(length):
        s += close[i]
    result[length - 1] = s / length
    alpha = 2.0 / (length + 1)
    for i in range(length, n):
        result[i] = alpha * close[i] + (1 - alpha) * result[i - 1]
    return result


@_njit_pmax(cache=True)
def _nb_sma_raw(close, length):
    """Compute SMA array (no pandas)."""
    n = len(close)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        s = 0.0
        for j in range(length):
            s += close[i - j]
        result[i] = s / length
    return result


def pl_pmax(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 10,
    multiplier: float = 3.0,
    mamode: str = "ema",
    offset: int = 0,
) -> PlExpr:
    """Polars: PMAX (Price Max)

    Combines ATR-based volatility bands with a moving average for
    adaptive trailing stops.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: ATR/MA period. Default: 10
        multiplier: ATR multiplier. Default: 3.0
        mamode: MA type ('ema' or 'sma'). Default: 'ema'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with PMAX, PMAXd, PMAXl, PMAXs columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_h"].to_numpy().astype(np.float64)
        l_ = data["_l"].to_numpy().astype(np.float64)
        c = data["_c"].to_numpy().astype(np.float64)

        atr_arr = _nb_pmax_atr(h, l_, c, length)
        if mamode.lower() == "sma":
            ma_arr = _nb_sma_raw(c, length)
        else:
            ma_arr = _nb_ema_raw(c, length)

        matr = multiplier * atr_arr
        ub = ma_arr - matr
        lb = ma_arr + matr

        trend, dir_, long_arr, short_arr = nb_pmax(c, ub, lb)
        dir_[:length] = np.nan

        if offset != 0:
            for a in [trend, dir_, long_arr, short_arr]:
                a[:] = np.roll(a, offset)
                if offset > 0:
                    a[:offset] = np.nan

        _props = f"_{length}_{multiplier}"
        n = len(h)
        return pl.Series(
            values=[
                {
                    f"PMAX{_props}": trend[i],
                    f"PMAXd{_props}": dir_[i],
                    f"PMAXl{_props}": long_arr[i],
                    f"PMAXs{_props}": short_arr[i],
                }
                for i in range(n)
            ]
        )

    _pprops = f"_{length}_{multiplier}"
    fields = [
        pl.Field(f"PMAX{_pprops}", pl.Float64),
        pl.Field(f"PMAXd{_pprops}", pl.Float64),
        pl.Field(f"PMAXl{_pprops}", pl.Float64),
        pl.Field(f"PMAXs{_pprops}", pl.Float64),
    ]
    return (
        pl.struct(
            high_expr.alias("_h"),
            low_expr.alias("_l"),
            close_expr.alias("_c"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"PMAX_{length}_{multiplier}")
    )
