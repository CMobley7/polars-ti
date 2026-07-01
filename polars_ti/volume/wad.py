# -*- coding: utf-8 -*-
# =============================================================================
# Polars WAD (Williams Accumulation/Distribution) Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_wad(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Numba-optimized Williams Accumulation/Distribution."""
    n = len(close)
    result = np.full(n, np.nan)
    cumsum = 0.0

    for i in range(1, n):
        prev_close = close[i - 1]
        c = close[i]
        h = high[i]
        lo = low[i]

        # True Range High / True Range Low
        trh = max(h, prev_close)
        trl = min(lo, prev_close)

        if c > prev_close:
            ad_day = c - trl
        elif c < prev_close:
            ad_day = c - trh
        else:
            ad_day = 0.0

        cumsum += ad_day
        result[i] = cumsum

    return result


def wad(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    offset: int = 0,
) -> PlExpr:
    """Polars: Williams Accumulation/Distribution (WAD)

    Cumulative flow of money into/out of a security using True Range
    High and True Range Low to determine the daily A/D value.

    Formula:
        TRH     = max(prev_close, high)
        TRL     = min(prev_close, low)
        AD_day  = close - TRL  if close > prev_close
        AD_day  = close - TRH  if close < prev_close
        AD_day  = 0            if close == prev_close
        WAD     = cumsum(AD_day)

    Sources:
        Larry Williams, "How I Made One Million Dollars Last Year Trading Commodities"

    Args:
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: WAD expression.
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if any(e is None for e in [high_expr, low_expr, close_expr]):
        return None

    _offset = offset

    def _compute(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        h = df["_high"].to_numpy().astype(np.float64)
        lo = df["_low"].to_numpy().astype(np.float64)
        c = df["_close"].to_numpy().astype(np.float64)
        result = nb_wad(h, lo, c)
        return pl.Series(result)

    result_expr = pl.struct(
        [
            high_expr.alias("_high"),
            low_expr.alias("_low"),
            close_expr.alias("_close"),
        ]
    ).map_batches(_compute, return_dtype=pl.Float64)

    if _offset != 0:
        result_expr = result_expr.shift(_offset)

    return result_expr.alias("WAD")
