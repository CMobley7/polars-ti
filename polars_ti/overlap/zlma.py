# -*- coding: utf-8 -*-
# =============================================================================
# Polars ZLMA Implementation (uses pl_ma for all mamode types)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.ma import pl_ma


def pl_zlma(
    close: IntoExpr,
    length: int = 10,
    mamode: str = "ema",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Zero Lag Moving Average (ZLMA)

    Eliminates lag by using (2 * close - close.shift(lag)) as input to any MA.

    Supported mamodes: dema, ema, fwma, hma, linreg, midpoint, pwma, rma,
                       sinwma, sma, ssf, swma, t3, tema, trima, vidya, wma

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        mamode: MA type (ema, sma, wma, hma, dema, tema, etc.). Default: "ema"
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ZLMA expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Supported MAs (same as Pandas zlma)
    supported_mas = [
        "dema",
        "ema",
        "fwma",
        "hma",
        "linreg",
        "midpoint",
        "pwma",
        "rma",
        "sinwma",
        "sma",
        "ssf",
        "swma",
        "t3",
        "tema",
        "trima",
        "vidya",
        "wma",
    ]

    _mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    if _mamode not in supported_mas:
        _mamode = "ema"  # Default fallback

    # Calculate lag and zero-lag adjusted series
    lag = int(0.5 * (length - 1))
    close_zl = 2 * close_expr - close_expr.shift(lag)

    # Apply MA using pl_ma - handles ALL mamodes!
    result = pl_ma(name=_mamode, source=close_zl, length=length, talib=talib)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"ZL_{_mamode.upper()}_{length}")
