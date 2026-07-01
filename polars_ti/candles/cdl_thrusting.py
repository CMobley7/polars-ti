# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_THRUSTING Implementation
# =============================================================================
"""Candle Pattern: Thrusting Pattern."""

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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(equal_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_eq = ca._ranges[CandleSetting.Equal]

    equal_trail = start_idx - 1 - equal_period
    body_long_trail = start_idx - 1 - body_long_period
    equal_total = float(arr_eq[equal_trail : start_idx - 1].sum())
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == -1
            and ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.color[i] == 1
            and ca.open[i] < ca.low[i - 1]
            and ca.close[i] > ca.close[i - 1] + AVG_FACTOR[CandleSetting.Equal] * equal_total
            and ca.close[i] <= ca.open[i - 1] - ca.real_body[i - 1] * 0.5
        ):
            out[i] = -100

        equal_total += arr_eq[i - 1] - arr_eq[equal_trail]
        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        equal_trail += 1
        body_long_trail += 1


def cdl_thrusting(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Thrusting.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_THRUSTING expression (-scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_THRUSTING",
        "THRUSTING",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
