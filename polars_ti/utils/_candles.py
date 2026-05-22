# -*- coding: utf-8 -*-
from polars_ti.utils._core import non_zero_range


def candle_color(open_, close):
    """Candle Change

    Returns 1 or -1 if close >= open_ respectively.
    """
    color = close.copy().astype(int)
    color[close >= open_] = 1
    color[close < open_] = -1
    return color


def high_low_range(high, low):
    """High Low Range

    Returns high - low = epsilon > 0
    """
    return non_zero_range(high, low)


def real_body(open_, close):
    """Body Low Range

    Returns close - open_ = epsilon > 0
    """
    return non_zero_range(close, open_)


# =============================================================================
# Polars Candle Utilities
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import pl_non_zero_range
from polars_ti.utils._validate import v_expr


def pl_candle_color(open_: IntoExpr, close: IntoExpr) -> PlExpr:
    """Polars: Candle Change - Returns 1 (bullish) or -1 (bearish)."""
    open_expr = v_expr(open_)
    close_expr = v_expr(close)
    return pl.when(close_expr >= open_expr).then(1).otherwise(-1).alias("candle_color")


def pl_high_low_range(high: IntoExpr, low: IntoExpr) -> PlExpr:
    """Polars: High-Low Range with epsilon for zero values."""
    return pl_non_zero_range(high, low).alias("hl_range")


def pl_real_body(open_: IntoExpr, close: IntoExpr) -> PlExpr:
    """Polars: Real Body (close - open) with epsilon for zero values."""
    return pl_non_zero_range(close, open_).alias("real_body")
