# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_STICKSANDWICH Implementation
# =============================================================================
"""Candle Pattern: Stick Sandwich."""

from typing import Any

import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles._cdl_math import (
    AVG_FACTOR,
    CandleArrays,
    CandleSetting,
    candle_avg_period,
    run_pattern,
)


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    equal_period = candle_avg_period(CandleSetting.Equal)
    lookback = equal_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - 2 - equal_period
    equal_total = float(arr_eq[equal_trail : start_idx - 2].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == -1
            and ca.color[i - 1] == 1
            and ca.color[i] == -1
            and ca.low[i - 1] > ca.close[i - 2]
            and ca.close[i] <= ca.close[i - 2] + AVG_FACTOR[CandleSetting.Equal] * equal_total
            and ca.close[i] >= ca.close[i - 2] - AVG_FACTOR[CandleSetting.Equal] * equal_total
        ):
            out[i] = 100

        equal_total += arr_eq[i - 2] - arr_eq[equal_trail]
        equal_trail += 1


def cdl_sticksandwich(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Stick Sandwich.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_STICKSANDWICH expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_STICKSANDWICH",
        "STICKSANDWICH",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
