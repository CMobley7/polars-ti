# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_COUNTERATTACK Implementation
# =============================================================================
"""Candle Pattern: Counterattack."""

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
    # Lookback: max(Equal, BodyLong) + 1
    equal_period = candle_avg_period(CandleSetting.Equal)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - equal_period
    body_long_trail = start_idx - body_long_period

    equal_total = 0.0
    i = equal_trail
    while i < start_idx:
        equal_total += arr_eq[i - 1]
        i += 1

    body_long_total_1 = 0.0
    body_long_total_0 = 0.0
    i = body_long_trail
    while i < start_idx:
        body_long_total_1 += arr_bl[i - 1]
        body_long_total_0 += arr_bl[i]
        i += 1

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -ca.color[i]  # opposite candles
            and ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total_1  # 1st long
            and ca.real_body[i] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total_0  # 2nd long
            and ca.close[i] <= ca.close[i - 1] + AVG_FACTOR[CandleSetting.Equal] * equal_total  # equal closes
            and ca.close[i] >= ca.close[i - 1] - AVG_FACTOR[CandleSetting.Equal] * equal_total
        ):
            out[i] = ca.color[i] * 100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail - 1]
        body_long_total_1 += arr_bl[i - 1] - arr_bl[body_long_trail - 1]
        body_long_total_0 += arr_bl[i] - arr_bl[body_long_trail]
        equal_trail += 1
        body_long_trail += 1


def cdl_counterattack(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Counterattack.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_COUNTERATTACK expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_COUNTERATTACK",
        "COUNTERATTACK",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
