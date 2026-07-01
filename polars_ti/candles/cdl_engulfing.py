# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_ENGULFING Implementation
# =============================================================================
"""Candle Pattern: Engulfing."""

from typing import Any

import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles._cdl_math import CandleArrays, run_pattern


def _is_bullish_engulf(ca: CandleArrays, i: int) -> bool:
    return (
        ca.color[i] == 1
        and ca.color[i - 1] == -1
        and (
            (ca.close[i] >= ca.open[i - 1] and ca.open[i] < ca.close[i - 1])
            or (ca.close[i] > ca.open[i - 1] and ca.open[i] <= ca.close[i - 1])
        )
    )


def _is_bearish_engulf(ca: CandleArrays, i: int) -> bool:
    return (
        ca.color[i] == -1
        and ca.color[i - 1] == 1
        and (
            (ca.open[i] >= ca.close[i - 1] and ca.close[i] < ca.open[i - 1])
            or (ca.open[i] > ca.close[i - 1] and ca.close[i] <= ca.open[i - 1])
        )
    )


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = 2
    if start_idx >= len(out):
        return

    for i in range(start_idx, len(out)):
        if _is_bullish_engulf(ca, i) or _is_bearish_engulf(ca, i):
            if ca.open[i] != ca.close[i - 1] and ca.close[i] != ca.open[i - 1]:
                out[i] = ca.color[i] * 100
            else:
                out[i] = ca.color[i] * 80


def cdl_engulfing(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Engulfing.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_ENGULFING expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_ENGULFING",
        "ENGULFING",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
