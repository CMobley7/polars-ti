# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_DARKCLOUDCOVER Implementation
# =============================================================================
"""Candle Pattern: Dark Cloud Cover."""

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
    penetration = kwargs.get("penetration", 0.5)
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 1
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]

    body_long_trail = start_idx - body_long_period

    body_long_total = 0.0
    i = body_long_trail
    while i < start_idx:
        body_long_total += arr_bl[i - 1]
        i += 1

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == 1  # 1st: white
            and ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total  # long
            and ca.color[i] == -1  # 2nd: black
            and ca.open[i] > ca.high[i - 1]  # open above prior high
            and ca.close[i] > ca.open[i - 1]  # close within prior body
            and ca.close[i] < ca.close[i - 1] - ca.real_body[i - 1] * penetration
        ):
            out[i] = -100

        body_long_total += arr_bl[i - 1] - arr_bl[body_long_trail - 1]
        body_long_trail += 1


def cdl_darkcloudcover(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    penetration: float = 0.5,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Dark Cloud Cover.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        penetration: Penetration within 1st candle's real body. Default: 0.5
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_DARKCLOUDCOVER expression (-scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_DARKCLOUDCOVER",
        "DARKCLOUDCOVER",
        penetration=penetration,
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
