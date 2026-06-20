# -*- coding: utf-8 -*-
# =============================================================================
# Polars TSignals Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def tsignals(
    trend: IntoExpr,
    asbool: bool = False,
    trade_offset: int = 0,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Trend Signals

    Given a Trend expression, returns Trends, Trades, Entries and Exits
    as a struct.

    Args:
        trend: Column name or pl.Expr for trend values (0/1 or bool)
        asbool: Return bool instead of int. Default: False
        trade_offset: Shift trade signals. Default: 0
        drift: Difference period. Default: 1
        offset: Shift all results. Default: 0

    Returns:
        pl.Expr: Struct with TS_Trends, TS_Trades, TS_Entries, TS_Exits
    """
    trend_expr = v_expr(trend)
    trends = trend_expr.cast(pl.Int64)
    trades = trends.diff(drift).shift(trade_offset).fill_null(0).cast(pl.Int64)
    entries = (trades > 0).cast(pl.Int64)
    exits = (trades < 0).cast(pl.Int64)

    if asbool:
        trends = trends.cast(pl.Boolean)
        entries = entries.cast(pl.Boolean)
        exits = exits.cast(pl.Boolean)

    if offset != 0:
        trends = trends.shift(offset)
        trades = trades.shift(offset)
        entries = entries.shift(offset)
        exits = exits.shift(offset)

    return pl.struct(
        trends.alias("TS_Trends"),
        trades.alias("TS_Trades"),
        entries.alias("TS_Entries"),
        exits.alias("TS_Exits"),
    ).alias("TS")
