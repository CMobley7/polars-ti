# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_Z Implementation
# =============================================================================
import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def _expanding_zscore(values: NDArray[np.float64], ddof: int) -> NDArray[np.float64]:
    """Update Welford moments around the first finite value.

    Anchoring preserves small spreads around large price offsets. Processing
    only observations through the current row prevents future-data leakage.
    """
    result = np.full(len(values), np.nan)
    count = 0
    origin = np.longdouble(0)
    mean = np.longdouble(0)
    moment = np.longdouble(0)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if count == 0:
            origin = np.longdouble(value)
        centered = np.longdouble(value) - origin
        count += 1
        delta = centered - mean
        mean += delta / count
        moment += delta * (centered - mean)
        if count > ddof and moment > 0:
            result[index] = float((centered - mean) / np.sqrt(moment / (count - ddof)))
    return result


def zscore(col: IntoExpr, length: int = 30, ddof: int = 1, full: bool = False) -> PlExpr:
    """Return rolling z-scores, or causal expanding z-scores when full=True."""
    expr = v_expr(col)
    if full:
        if ddof < 0:
            raise ValueError("ddof must be nonnegative")
        return expr.map_batches(
            lambda series: pl.Series(_expanding_zscore(series.to_numpy().astype(np.float64), ddof)),
            return_dtype=pl.Float64,
        )
    else:
        mean = expr.rolling_mean(window_size=length, min_samples=length)
        std = expr.rolling_std(window_size=length, ddof=ddof, min_samples=length)
    return (expr - mean) / std


def cdl_z(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 30,
    full: bool = False,
    ddof: int = 1,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Candle Type Z - Rolling Z-Score normalized OHLC

    Normalizes OHLC Candles with a rolling Z Score.

    Source: Kevin Johnson

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        full: If True, use all finite observations up to the current row. Default: False
        ddof: Degrees of freedom for std. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of Z-score expressions for OHLC
    """
    props = f"_{length}_{ddof}"

    # Build base expressions
    exprs = [
        zscore(open_, length, ddof, full).alias(f"open_Z{props}"),
        zscore(high, length, ddof, full).alias(f"high_Z{props}"),
        zscore(low, length, ddof, full).alias(f"low_Z{props}"),
        zscore(close, length, ddof, full).alias(f"close_Z{props}"),
    ]

    # Apply offset if needed
    if offset != 0:
        exprs = [e.shift(offset) for e in exprs]

    return exprs
