# -*- coding: utf-8 -*-
# =============================================================================
# Polars AVOLUME (Annualised Historical Volatility) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def avolume(
    close: IntoExpr,
    length: int = 20,
    offset: int = 0,
) -> PlExpr:
    """Polars: Annualised Historical Volatility (AVOLUME / tulipy: VOLATILITY)

    Population standard deviation of log-returns annualised by sqrt(252).
    Values are fractions (e.g. 0.09 = 9% p.a.), unlike hvol which returns
    percentages.  Uses population standard deviation (ddof=0), matching
    tulipy's implementation.

    Formula:
        log_ret   = log(close / prev_close)
        AVOLUME   = std(log_ret, length, ddof=0) * sqrt(252)

    Sources:
        Tulip Indicators: https://tulipindicators.org/volatility

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Lookback period. Default: 20
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: AVOLUME expression (fraction).
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _offset = offset

    log_ret = (close_expr / close_expr.shift(1)).log(base=np.e)
    result = log_ret.rolling_std(window_size=_length, ddof=0) * np.sqrt(252.0)

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias(f"AVOLUME_{length}")
