# -*- coding: utf-8 -*-
# =============================================================================
# Polars FOSC (Forecast Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def fosc(
    close: IntoExpr,
    length: int = 14,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Forecast Oscillator (FOSC)

    Percentage difference between actual close price and the Time Series
    Forecast (linear regression projected one step ahead).

    Formula:
        TSF    = Linear Regression Value projected one bar ahead
        FOSC   = 100 * (Close - TSF) / Close

    Sources:
        Tushar Chande, "The New Technical Trader", 1994
        Tulip Indicators: https://tulipindicators.org/fosc

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Lookback period. Default: 14
        talib: If True and TA-Lib is installed, uses TA-Lib for the underlying
            TSF/linreg. Native and TA-Lib TSF agree exactly. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: FOSC expression.
    """
    from polars_ti.overlap.linreg import linreg

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _offset = offset

    # TSF = Time Series Forecast (linreg with tsf=True)
    tsf_expr = linreg(close_expr, length=_length, talib=talib, tsf=True)

    result = pl.lit(100.0) * (close_expr - tsf_expr) / close_expr

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias(f"FOSC_{length}")
