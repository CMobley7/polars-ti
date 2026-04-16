# -*- coding: utf-8 -*-
# =============================================================================
# Polars XSignals Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_xsignals(
    signal: IntoExpr,
    xa: float,
    xb: float,
    above: bool = True,
    long: bool = True,
    asbool: bool = False,
    trade_offset: int = 0,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Cross Signals (XSIGNALS)

    Returns trend signals for signal crossings. Useful for RSI, ZSCORE etc.

    Args:
        signal: Column name or pl.Expr for the signal
        xa: First cross threshold
        xb: Second cross threshold
        above: Cross above xa first, then below xb. Default: True
        long: Use long trend. Default: True
        asbool: Return bool instead of int. Default: False
        trade_offset: Shift trade signals. Default: 0
        drift: Difference period. Default: 1
        offset: Shift all results. Default: 0

    Returns:
        pl.Expr: Struct with TS_Trends, TS_Trades, TS_Entries, TS_Exits
    """
    from polars_ti.trend.tsignals import pl_tsignals

    signal_expr = v_expr(signal)

    if above:
        cross_above = (signal_expr > xa) & (signal_expr.shift(1) <= xa)
        cross_below = (signal_expr < xb) & (signal_expr.shift(1) >= xb)
    else:
        cross_above = (signal_expr < xa) & (signal_expr.shift(1) >= xa)
        cross_below = (signal_expr > xb) & (signal_expr.shift(1) <= xb)

    entries = cross_above.cast(pl.Int64)
    exits = cross_below.cast(pl.Int64) * -1
    trades = entries + exits

    # Forward fill inside to create trend
    trends_raw = trades.replace(0, None).forward_fill().fill_null(0)
    trends = (trends_raw > 0).cast(pl.Int64)

    if not long:
        trends = pl.lit(1) - trends

    return pl_tsignals(
        trends, asbool=asbool, trade_offset=trade_offset,
        drift=drift, offset=offset,
    )
