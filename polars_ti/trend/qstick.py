# -*- coding: utf-8 -*-
# =============================================================================
# Polars QStick Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def qstick(
    open_: IntoExpr,
    close: IntoExpr,
    length: int = 10,
    mamode: str = "sma",
    offset: int = 0,
) -> PlExpr:
    """Polars: Q Stick

    Quantifies and identifies trends in candlestick charts by applying
    a moving average to the close-open difference.

    Formula: QS = MA(close - open, length)

    Args:
        open_: Column name or pl.Expr for 'open' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 10
        mamode: MA type ('sma', 'ema', etc.). Default: 'sma'
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: QStick expression
    """
    from polars_ti.ma import ma

    open_expr = v_expr(open_)
    close_expr = v_expr(close)

    diff_expr = close_expr - open_expr
    result = ma(mamode, diff_expr, length=length, talib=False)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"QS_{length}")
