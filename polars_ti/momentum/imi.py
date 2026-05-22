# -*- coding: utf-8 -*-
# =============================================================================
# Polars IMI (Intraday Momentum Index) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_imi(
    open_: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Intraday Momentum Index (IMI)

    Combines candlestick analysis with RSI to generate overbought/oversold signals.
    IMI = 100 * sum(gains) / (sum(gains) + sum(losses))

    Args:
        open_: Column name or pl.Expr for 'open' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period for sums. Default: 14
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: IMI expression (0-100)
    """
    open_expr = v_expr(open_)
    close_expr = v_expr(close)

    # Gains when close > open, else 0
    gains = pl.when(close_expr > open_expr).then(close_expr - open_expr).otherwise(0.0)
    # Losses when close < open, else 0
    losses = pl.when(close_expr < open_expr).then(open_expr - close_expr).otherwise(0.0)

    sum_gains = gains.rolling_sum(window_size=length)
    sum_losses = losses.rolling_sum(window_size=length)

    imi_expr = 100.0 * sum_gains / (sum_gains + sum_losses)

    if offset != 0:
        imi_expr = imi_expr.shift(offset)

    return imi_expr.alias(f"IMI_{length}")
