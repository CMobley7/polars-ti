# -*- coding: utf-8 -*-
# =============================================================================
# Polars DRAWDOWN Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def drawdown(
    close: IntoExpr,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Drawdown (DD)

    Drawdown is a peak-to-trough decline during a specific period for an
    investment, trading account, or fund.

    Sources:
        https://www.investopedia.com/terms/d/drawdown.asp

    Args:
        close: Column name or pl.Expr for 'close' prices
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with DD, DD_PCT, DD_LOG columns
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Calculate using pure Polars expressions
    max_close = close_expr.cum_max()
    dd = max_close - close_expr
    dd_pct = 1 - (close_expr / max_close)
    dd_log = max_close.log() - close_expr.log()

    # Apply offset if needed
    if offset != 0:
        dd = dd.shift(offset)
        dd_pct = dd_pct.shift(offset)
        dd_log = dd_log.shift(offset)

    return pl.struct(
        [
            dd.alias("DD"),
            dd_pct.alias("DD_PCT"),
            dd_log.alias("DD_LOG"),
        ]
    ).alias("DRAWDOWN")
