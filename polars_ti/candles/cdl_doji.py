# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_DOJI Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._candles import high_low_range, real_body
from polars_ti.utils._validate import v_expr


def cdl_doji(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
    asint: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Candle Type - Doji

    A candle body is Doji when it's shorter than a percentage of
    the average of the previous candles' high-low range.

    Sources:
        TA-Lib: 96.56% Correlation

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: The averaging period. Default: 10
        factor: Doji threshold percentage. Default: 10.0 (means 10%)
        scalar: Result multiplier. Default: 100.0
        asint: Return integer (scaled) instead of boolean. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: CDL_DOJI expression (100 for doji, 0 otherwise)
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    # Calculate real body (absolute difference between close and open)
    body = (close_expr - open_expr).abs()

    # Calculate high-low range
    hl_range = (high_expr - low_expr).abs()

    # Calculate average high-low range over the period
    hl_range_avg = hl_range.rolling_mean(window_size=length, min_samples=length)

    # Doji: body < 0.01 * factor * average HL range
    threshold = 0.01 * factor * hl_range_avg

    if asint:
        doji = pl.when(body < threshold).then(scalar).otherwise(0.0).cast(pl.Int64)
    else:
        doji = body < threshold

    # Apply offset
    if offset != 0:
        doji = doji.shift(offset)

    return doji.alias(f"CDL_DOJI_{length}_{0.01 * factor}")
