# -*- coding: utf-8 -*-
# =============================================================================
# Polars AVWAP (Anchored Volume Weighted Average Price) Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_find_pivots(data: np.ndarray, left: int, right: int, is_high: bool) -> np.ndarray:
    """Find pivot points in a series using Numba."""
    n = len(data)
    pivots = np.zeros(n, dtype=np.bool_)

    for i in range(left, n - right):
        window = data[i - left : i + right + 1]
        if is_high:
            if data[i] == np.max(window):
                pivots[i] = True
        else:
            if data[i] == np.min(window):
                pivots[i] = True

    return pivots


@njit(cache=True)
def _nb_avwap(close: np.ndarray, volume: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    """Calculate Anchored VWAP from pivot points."""
    n = len(close)
    result = np.full(n, np.nan)
    last_pivot = 0

    for i in range(n):
        if pivots[i]:
            last_pivot = i

        if last_pivot <= i:
            # Calculate VWAP from last pivot to current
            vp_sum = 0.0
            v_sum = 0.0
            for j in range(last_pivot, i + 1):
                vp_sum += volume[j] * close[j]
                v_sum += volume[j]

            if v_sum > 0:
                result[i] = vp_sum / v_sum

    return result


def avwap(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    left_strength: int = 5,
    right_strength: int = 5,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Anchored Volume Weighted Average Price (AVWAP)

    Anchored VWAP is calculated from specific pivot points in the data,
    providing insights into price action around significant market events.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        left_strength: Bars back for pivot detection. Default: 5
        right_strength: Bars forward for pivot detection. Default: 5
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of expressions [AVWAP_HIGH, AVWAP_LOW]
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None

    _left = left_strength
    _right = right_strength
    _props = f"_{left_strength}_{right_strength}"

    def compute_avwap_high(df: pl.DataFrame) -> pl.Series:
        h = df["high"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)

        pivot_highs = _nb_find_pivots(h, _left, _right, True)
        avwap_high = _nb_avwap(c, v, pivot_highs)
        return pl.Series(f"AVWAPH{_props}", avwap_high)

    def compute_avwap_low(df: pl.DataFrame) -> pl.Series:
        l = df["low"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)

        pivot_lows = _nb_find_pivots(l, _left, _right, False)
        avwap_low = _nb_avwap(c, v, pivot_lows)
        return pl.Series(f"AVWAPL{_props}", avwap_low)

    avwap_h_expr = pl.struct(
        [
            high_expr.alias("high"),
            close_expr.alias("close"),
            volume_expr.alias("volume"),
        ]
    ).map_batches(lambda s: compute_avwap_high(s.struct.unnest()), return_dtype=pl.Float64)

    avwap_l_expr = pl.struct(
        [low_expr.alias("low"), close_expr.alias("close"), volume_expr.alias("volume")]
    ).map_batches(lambda s: compute_avwap_low(s.struct.unnest()), return_dtype=pl.Float64)

    if offset != 0:
        avwap_h_expr = avwap_h_expr.shift(offset)
        avwap_l_expr = avwap_l_expr.shift(offset)

    return [
        avwap_h_expr.alias(f"AVWAPH{_props}"),
        avwap_l_expr.alias(f"AVWAPL{_props}"),
    ]
