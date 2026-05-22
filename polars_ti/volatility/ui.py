# -*- coding: utf-8 -*-
# =============================================================================
# Polars UI Implementation (Pure Composition)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def pl_ui(
    close: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    everget: bool = False,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Ulcer Index (UI)

    Pure composition using pl_sma and native rolling functions.

    The Ulcer Index measures downside volatility using the Quadratic Mean.

    Sources:
        https://library.tradingtechnologies.com/trade/chrt-ti-ulcer-index.html
        https://en.wikipedia.org/wiki/Ulcer_index

    Args:
        close: Column name or pl.Expr for 'close'
        length: The period. Default: 14
        scalar: Scale factor. Default: 100.0
        everget: Use TradingView's Everget SMA instead of SUM. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: UI expression
    """
    from polars_ti.overlap.sma import pl_sma

    close_expr = v_expr(close)

    if close_expr is None:
        return None

    # Highest close over rolling window
    highest_close = close_expr.rolling_max(window_size=length, min_samples=1)

    # Downside = scalar * (close - highest) / highest
    downside = pl.lit(scalar) * (close_expr - highest_close) / highest_close
    d2 = downside * downside

    # Everget uses SMA instead of SUM
    if everget:
        _ui = pl_sma(d2, length=length)
    else:
        _ui = d2.rolling_sum(window_size=length, min_samples=length)

    # UI = sqrt(_ui / length)
    result = (_ui / pl.lit(length)).sqrt()

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    _name = f"UI{'' if not everget else 'e'}_{length}"
    return result.alias(_name)
