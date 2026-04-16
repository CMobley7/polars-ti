# -*- coding: utf-8 -*-
# =============================================================================
# Polars RVGI Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.swma import pl_swma


def pl_rvgi(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    swma_length: int = 4,
    offset: int = 0,
) -> PlExpr:
    """Polars: Relative Vigor Index (RVGI)

    The Relative Vigor Index attempts to measure the strength of a trend
    relative to its closing price to its trading range. It is based on the
    belief that prices tend to close higher than they open in uptrends or 
    close lower than they open in downtrends.

    Sources:
        https://www.investopedia.com/terms/r/relative_vigor_index.asp

    Args:
        open_: Column name or pl.Expr for 'open' prices
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Rolling sum period. Default: 14
        swma_length: SWMA period. Default: 4
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with RVGI and RVGIs (signal) columns
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if any(x is None for x in [open_expr, high_expr, low_expr, close_expr]):
        return None

    # Calculate ranges
    close_open_range = close_expr - open_expr
    high_low_range = high_expr - low_expr
    
    # Apply SWMA to ranges, then rolling sum
    swma_close_open = pl_swma(close_open_range, length=swma_length)
    swma_high_low = pl_swma(high_low_range, length=swma_length)
    
    numerator = swma_close_open.rolling_sum(window_size=length)
    denominator = swma_high_low.rolling_sum(window_size=length)
    
    # Avoid division by zero
    rvgi_expr = numerator / pl.when(denominator == 0).then(None).otherwise(denominator)
    
    # Signal is SWMA of RVGI
    signal_expr = pl_swma(rvgi_expr, length=swma_length)
    
    # Apply offset
    if offset != 0:
        rvgi_expr = rvgi_expr.shift(offset)
        signal_expr = signal_expr.shift(offset)
    
    # Return as struct
    return pl.struct([
        rvgi_expr.alias(f"RVGI_{length}_{swma_length}"),
        signal_expr.alias(f"RVGIs_{length}_{swma_length}"),
    ]).alias("RVGI")
