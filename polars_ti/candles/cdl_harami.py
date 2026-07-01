# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_HARAMI Implementation
# =============================================================================
"""Candle Pattern: Harami."""

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
    # Lookback: max(BodyShort, BodyLong) + 1
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = max(body_short_period, body_long_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 1 - body_long_period
    body_short_trail = start_idx - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 1].sum())
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyShort] * body_short_total  # 1st: long
        ):  # 2nd: short
            hi_i = body_hi[i]
            lo_i = body_lo[i]
            hi_p = body_hi[i - 1]
            lo_p = body_lo[i - 1]
            if hi_i < hi_p and lo_i > lo_p:
                out[i] = -ca.color[i - 1] * 100
            elif hi_i <= hi_p and lo_i >= lo_p:
                out[i] = -ca.color[i - 1] * 80

        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_short_trail += 1


def cdl_harami(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Harami.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_HARAMI expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HARAMI",
        "HARAMI",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
