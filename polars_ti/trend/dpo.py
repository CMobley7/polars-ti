# -*- coding: utf-8 -*-
# =============================================================================
# Polars DPO (Detrend Price Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def dpo(
    close: IntoExpr,
    length: int = 20,
    centered: bool = True,
    lookahead: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Detrend Price Oscillator (DPO)

    Removes trend from price to identify cycles.

    WARNING: centered=True (default) leaks future data. Set lookahead=False
    to avoid data leakage in ML applications.

    Formula:
        t = int(0.5 * length) + 1
        centered: DPO = close.shift(t) - SMA(close, length), then shift(-t)
        non-centered: DPO = close - SMA(close, length).shift(t)

    Args:
        close: Column name or pl.Expr for input values
        length: Period. Default: 20
        centered: Shift DPO back. Default: True
        lookahead: If False, forces centered=False. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: DPO expression
    """
    from polars_ti.overlap.sma import sma

    close_expr = v_expr(close)
    t = int(0.5 * length) + 1

    ma_expr = sma(close_expr, length=length, talib=False, offset=0)

    _centered = centered and lookahead
    if _centered:
        dpo_expr = (close_expr.shift(t) - ma_expr).shift(-t)
    else:
        dpo_expr = close_expr - ma_expr.shift(t)

    if offset != 0:
        dpo_expr = dpo_expr.shift(offset)

    return dpo_expr.alias(f"DPO_{length}")
