# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_TASUKIGAP Implementation
# =============================================================================
"""Candle Pattern: Tasuki Gap."""

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
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_nr = ca._ranges[CandleSetting.Near]
    body_hi = ca.body_high
    body_lo = ca.body_low

    near_trail = start_idx - near_period
    near_total = float(arr_nr[near_trail - 1 : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            body_lo[i - 1] > body_hi[i - 2]
            and ca.color[i - 1] == 1
            and ca.color[i] == -1
            and ca.open[i] < ca.close[i - 1]
            and ca.open[i] > ca.open[i - 1]
            and ca.close[i] < ca.open[i - 1]
            and ca.close[i] > body_hi[i - 2]
            and abs(ca.real_body[i - 1] - ca.real_body[i]) < AVG_FACTOR[CandleSetting.Near] * near_total
        ) or (
            body_hi[i - 1] < body_lo[i - 2]
            and ca.color[i - 1] == -1
            and ca.color[i] == 1
            and ca.open[i] < ca.open[i - 1]
            and ca.open[i] > ca.close[i - 1]
            and ca.close[i] > ca.open[i - 1]
            and ca.close[i] < body_lo[i - 2]
            and abs(ca.real_body[i - 1] - ca.real_body[i]) < AVG_FACTOR[CandleSetting.Near] * near_total
        ):
            out[i] = ca.color[i - 1] * 100

        near_total += arr_nr[i - 1] - arr_nr[near_trail - 1]
        near_trail += 1


def cdl_tasukigap(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Tasuki Gap.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_TASUKIGAP expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TASUKIGAP",
        "TASUKIGAP",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
