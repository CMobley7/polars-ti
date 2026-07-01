# -*- coding: utf-8 -*-
# =============================================================================
# Polars MD (Mean Deviation) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def md(
    close: IntoExpr,
    length: int = 30,
    offset: int = 0,
) -> PlExpr:
    """Polars: Mean Deviation (MD) — tulipy name: MD

    Rolling mean of absolute deviations from the rolling mean.
    Equivalent to ta.mad.  tulipy name: MD.

    Formula:
        MD = mean(|x - mean(x)|) over rolling window

    Sources:
        Tulip Indicators: https://tulipindicators.org/md

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Rolling window period. Default: 30
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MD expression.
    """
    from polars_ti.statistics.mad import mad

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _offset = offset

    # MD is identical to MAD (mean absolute deviation) — reuse the kernel
    result = mad(close_expr, length=_length, offset=_offset)

    # Rename from MAD_n to MD_n
    return result.alias(f"MD_{length}")
