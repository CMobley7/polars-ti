# -*- coding: utf-8 -*-
from numpy import isnan, nan, zeros_like
from numba import njit


@njit(cache=True)
def nb_alpha(low_atr, high_atr, momo_threshold):
    m = momo_threshold.size
    result = zeros_like(low_atr, dtype=low_atr.dtype)

    for i in range(1, m):
        if momo_threshold[i]:
            if low_atr[i] < result[i - 1]:
                result[i] = result[i - 1]
            else:
                result[i] = low_atr[i]
        else:
            if high_atr[i] > result[i - 1]:
                result[i] = result[i - 1]
            else:
                result[i] = high_atr[i]
    result[0] = nan

    return result


# =============================================================================
# Polars AlphaTrend Implementation (reuses nb_alpha kernel)
# =============================================================================
import numpy as np
import polars as pl
from numba import njit as _njit_at

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@_njit_at(cache=True)
def _nb_rma(arr, length):
    """RMA (Wilder's smoothing) for ATR/RSI computation."""
    n = len(arr)
    result = np.full(n, np.nan)
    # SMA seed
    s = 0.0
    for i in range(length):
        s += arr[i]
    result[length - 1] = s / length
    alpha = 1.0 / length
    for i in range(length, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


@_njit_at(cache=True)
def _nb_atr_raw(high, low, close, length):
    """Compute ATR array from raw OHLC data."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return _nb_rma(tr, length)


@_njit_at(cache=True)
def _nb_rsi_raw(close, length):
    """Compute RSI array from raw close data."""
    n = len(close)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = _nb_rma(gains, length)
    avg_loss = _nb_rma(losses, length)

    rsi = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(avg_gain[i]) and not np.isnan(avg_loss[i]):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def pl_alphatrend(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    multiplier: float = 1.0,
    threshold: float = 50.0,
    lag: int = 2,
    offset: int = 0,
) -> PlExpr:
    """Polars: Alpha Trend

    Filters out sideways conditions for more accurate trend signals.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        multiplier: ATR multiplier. Default: 1
        threshold: Momentum threshold. Default: 50
        lag: Lag period. Default: 2
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with ALPHAT and ALPHATl columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_h"].to_numpy().astype(np.float64)
        l_ = data["_l"].to_numpy().astype(np.float64)
        c = data["_c"].to_numpy().astype(np.float64)

        atr_arr = _nb_atr_raw(h, l_, c, length)
        rsi_arr = _nb_rsi_raw(c, length)

        lower_atr = l_ - atr_arr * multiplier
        upper_atr = h + atr_arr * multiplier
        momo = rsi_arr >= threshold

        at = nb_alpha(lower_atr, upper_atr, momo)
        atl = np.roll(at, lag)
        atl[:lag] = np.nan

        if offset != 0:
            at = np.roll(at, offset)
            atl = np.roll(atl, offset)
            if offset > 0:
                at[:offset] = np.nan
                atl[:offset] = np.nan

        _props = f"_{length}_{multiplier}_{threshold}"
        n = len(h)
        return pl.Series(values=[{f"ALPHAT{_props}": at[i], f"ALPHATl{_props}_{lag}": atl[i]} for i in range(n)])

    _aprops = f"_{length}_{multiplier}_{threshold}"
    fields = [
        pl.Field(f"ALPHAT{_aprops}", pl.Float64),
        pl.Field(f"ALPHATl{_aprops}_{lag}", pl.Float64),
    ]
    return (
        pl.struct(
            high_expr.alias("_h"),
            low_expr.alias("_l"),
            close_expr.alias("_c"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"ALPHAT{_aprops}")
    )
