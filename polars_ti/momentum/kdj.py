# -*- coding: utf-8 -*-
# =============================================================================
# Polars KDJ Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_rma(x: np.ndarray, n: int) -> np.ndarray:
    """Numba-accelerated Wilder's Recursive Moving Average (RMA/SMMA)."""
    m = len(x)
    result = np.full(m, np.nan, dtype=np.float64)
    alpha = 1.0 / n

    # Find first non-nan entry and seed from there
    seed = -1
    for i in range(m):
        if not np.isnan(x[i]):
            seed = i
            break
    if seed < 0:
        return result

    result[seed] = x[seed]
    for i in range(seed + 1, m):
        if np.isnan(x[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * x[i] + (1.0 - alpha) * result[i - 1]
    return result


def pl_kdj(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 9,
    signal: int = 3,
    offset: int = 0,
) -> list[pl.Expr]:
    """Polars: KDJ

    The KDJ indicator is a derived form of the Slow Stochastic with an
    extra J line (divergence of %D from %K). J can go beyond [0, 100].

    Sources:
        https://www.prorealcode.com/prorealtime-indicators/kdj/
        https://docs.anychart.com/Stock_Charts/Technical_Indicators/Mathematical_Description#kdj

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Stochastic lookback. Default: 9
        signal: Smoothing period. Default: 3
        offset: Shift result. Default: 0

    Returns:
        list[pl.Expr]: [K, D, J] expressions
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    _length = length
    _signal = signal

    def compute(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        h_arr = df["high"].to_numpy().astype(np.float64)
        l_arr = df["low"].to_numpy().astype(np.float64)
        c_arr = df["close"].to_numpy().astype(np.float64)
        n = len(h_arr)

        # Rolling max/min with window = _length
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        for i in range(_length - 1, n):
            window_h = h_arr[i - _length + 1 : i + 1]
            window_l = l_arr[i - _length + 1 : i + 1]
            if not np.any(np.isnan(window_h)):
                highest_high[i] = np.max(window_h)
            if not np.any(np.isnan(window_l)):
                lowest_low[i] = np.min(window_l)

        # Fast %K
        denom = highest_high - lowest_low
        # Avoid divide-by-zero
        denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        fastk = 100.0 * (c_arr - lowest_low) / denom_safe

        # Smooth with RMA
        k_arr = _nb_rma(fastk, _signal)
        d_arr = _nb_rma(k_arr, _signal)
        j_arr = 3.0 * k_arr - 2.0 * d_arr

        return pl.Series([{"K": float(kv), "D": float(dv), "J": float(jv)} for kv, dv, jv in zip(k_arr, d_arr, j_arr)])

    struct_expr = pl.struct(high=high_expr, low=low_expr, close=close_expr)
    result = struct_expr.map_batches(
        compute,
        return_dtype=pl.Struct({"K": pl.Float64, "D": pl.Float64, "J": pl.Float64}),
    )

    _props = f"_{length}_{signal}"
    k_expr = result.struct.field("K")
    d_expr = result.struct.field("D")
    j_expr = result.struct.field("J")

    if offset != 0:
        k_expr = k_expr.shift(offset)
        d_expr = d_expr.shift(offset)
        j_expr = j_expr.shift(offset)

    return [
        k_expr.alias(f"K{_props}"),
        d_expr.alias(f"D{_props}"),
        j_expr.alias(f"J{_props}"),
    ]
