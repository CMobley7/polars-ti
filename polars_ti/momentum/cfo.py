# -*- coding: utf-8 -*-
# =============================================================================
# Polars CFO (Chande Forecast Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def cfo(
    close: IntoExpr,
    length: int = 9,
    scalar: float = 100.0,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Chande Forecast Oscillator (CFO)

    Calculates the percentage difference between actual price and
    the Time Series Forecast (endpoint of linear regression line).

    Formula: CFO = scalar * (close - TSF) / close

    Sources:
        https://www.fmlabs.com/reference/default.htm?url=ForecastOscillator.htm

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 9
        scalar: Magnification factor. Default: 100
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CFO expression
    """
    from polars_ti.overlap.linreg import linreg

    close_expr = v_expr(close)

    # TSF = Time Series Forecast from linear regression
    tsf = linreg(close_expr, length=length, talib=talib, tsf=True, offset=0)

    # CFO = scalar * (close - TSF) / close
    # Protect against divide-by-zero
    cfo_expr = pl.when(close_expr.abs() < 1e-10).then(None).otherwise(scalar * (close_expr - tsf) / close_expr)

    if offset != 0:
        cfo_expr = cfo_expr.shift(offset)

    return cfo_expr.alias(f"CFO_{length}")
