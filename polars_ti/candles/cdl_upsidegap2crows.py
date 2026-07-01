# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_UPSIDEGAP2CROWS Implementation
# =============================================================================
"""Candle Pattern: Upside Gap Two Crows."""

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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_long_period, body_short_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_long_trail = start_idx - 2 - body_long_period
    body_short_trail = start_idx - 1 - body_short_period
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())
    body_short_total = float(arr_bs[body_short_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == 1
            and ca.real_body[i - 2] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.color[i - 1] == -1
            and ca.real_body[i - 1] <= AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and body_lo[i - 1] > body_hi[i - 2]
            and ca.color[i] == -1
            and ca.open[i] > ca.open[i - 1]
            and ca.close[i] < ca.close[i - 1]
            and ca.close[i] > ca.close[i - 2]
        ):
            out[i] = -100

        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_short_total += arr_bs[i - 1] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_short_trail += 1


def cdl_upsidegap2crows(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Upside Gap Two Crows.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_UPSIDEGAP2CROWS expression (-scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_UPSIDEGAP2CROWS",
        "UPSIDEGAP2CROWS",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
