# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_MATCHINGLOW Implementation
# =============================================================================
"""Candle Pattern: Matching Low."""

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
    lookback = equal_period + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - 1 - equal_period
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -1
            and ca.color[i] == -1
            and ca.close[i] <= ca.close[i - 1] + AVG_FACTOR[CandleSetting.Equal] * equal_total
            and ca.close[i] >= ca.close[i - 1] - AVG_FACTOR[CandleSetting.Equal] * equal_total
        ):
            out[i] = 100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        equal_trail += 1


def cdl_matchinglow(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Matching Low.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_MATCHINGLOW expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_MATCHINGLOW",
        "MATCHINGLOW",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
