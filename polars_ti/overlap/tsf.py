# -*- coding: utf-8 -*-
# =============================================================================
# Polars TSF (Time Series Forecast) Implementation
# =============================================================================
from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.overlap.linreg import linreg
from polars_ti.utils._validate import v_expr


def tsf(
    close: IntoExpr,
    length: int = 14,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Time Series Forecast (TSF)

    The Time Series Forecast is the one-step-ahead projection of a rolling
    linear regression fitted over the last ``length`` closes. It is exactly the
    ``tsf=True`` branch of :func:`linreg`, which is verified against
    ``talib.TSF``.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 14
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: TSF expression aliased ``TSF_{length}``
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    result = linreg(close_expr, length=length, talib=talib, offset=offset, tsf=True)

    return result.alias(f"TSF_{length}")
