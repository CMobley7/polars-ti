# -*- coding: utf-8 -*-
# =============================================================================
# Polars PSL (Psychological Line) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_psl(
    close: IntoExpr,
    open_: IntoExpr | None = None,
    length: int = 12,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Psychological Line (PSL)

    The Psychological Line measures the percentage of rising periods over a
    lookback window. It's used to gauge market sentiment and identify
    overbought/oversold conditions.

    Sources:
        https://iqoption.com/blog/psychological-line
        
    Calculation:
        If open_ provided:
            diff = sign(close - open_)
        Else:
            diff = sign(close.diff(drift))
        diff = diff.clip(lower=0)  # Only count positive
        PSL = scalar * rolling_sum(diff, length) / length

    Args:
        close: Column name or pl.Expr for 'close' prices
        open_: Column name or pl.Expr for 'open' prices (optional)
        length: Rolling window period. Default: 12
        scalar: Multiplication factor. Default: 100
        drift: Difference period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: PSL expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None
    
    # Calculate diff based on open or drift
    if open_ is not None:
        open_expr = v_expr(open_)
        if open_expr is None:
            return None
        # sign(close - open): positive = +1, negative/zero = 0 after clipping
        diff = (close_expr - open_expr).sign()
    else:
        # sign(close.diff(drift))
        diff = close_expr.diff(drift).sign()
    
    # Fill NaN with 0, then clip to count only positive (rising) periods
    diff = diff.fill_nan(0.0).fill_null(0.0)
    # When diff <= 0, set to 0; when diff > 0, keep as 1
    diff = pl.when(diff > 0).then(1.0).otherwise(0.0)
    
    # PSL = scalar * rolling_sum(diff) / length
    psl = scalar * diff.rolling_sum(length) / length
    
    # Apply offset
    if offset != 0:
        psl = psl.shift(offset)
    
    return psl.alias(f"PSL_{length}")
