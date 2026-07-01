# -*- coding: utf-8 -*-
# =============================================================================
# Polars HVOL (Historical Volatility) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def hvol(
    close: IntoExpr,
    length: int = 20,
    annualization: float = 252.0,
    offset: int = 0,
) -> PlExpr:
    """Polars: Historical Volatility (HVOL)

    Annualized standard deviation of logarithmic daily returns over a
    given period.  Uses sample standard deviation (ddof=1).

    Formula:
        log_return = log(close / prev_close)
        HVOL = 100 * std(log_return, length, ddof=1) * sqrt(annualization)

    Sources:
        https://www.investopedia.com/terms/h/historicalvolatility.asp

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Lookback period for std dev. Default: 20
        annualization: Annualization factor. Default: 252 (trading days/year)
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HVOL expression (annualized %).
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _annualization = annualization
    _offset = offset

    log_ret = (close_expr / close_expr.shift(1)).log(base=np.e)
    result = pl.lit(100.0) * log_ret.rolling_std(window_size=_length, ddof=1) * np.sqrt(_annualization)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias(f"HVOL_{length}")
