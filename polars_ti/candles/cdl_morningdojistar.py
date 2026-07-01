# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_MORNINGDOJISTAR Implementation
# =============================================================================
"""Candle Pattern: Morning Doji Star."""

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
    penetration = kwargs.get("penetration", 0.3)

    # Lookback: max(BodyDoji, BodyLong, BodyShort) + 2
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    lookback = max(body_doji_period, body_long_period, body_short_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    body_hi = ca.body_high
    body_lo = ca.body_low

    # Trailing indices
    # BodyLong at offset i-2: trail = startIdx - 2 - period
    body_long_trail = start_idx - 2 - body_long_period
    # BodyDoji at offset i-1: trail = startIdx - 1 - period
    body_doji_trail = start_idx - 1 - body_doji_period
    # BodyShort at offset i: trail = startIdx - period
    body_short_trail = start_idx - body_short_period

    # Seed totals
    body_long_total = float(arr_bl[body_long_trail : start_idx - 2].sum())
    body_doji_total = float(arr_bd[body_doji_trail : start_idx - 1].sum())
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            # 1st: long black
            ca.real_body[i - 2] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.color[i - 2] == -1
            # 2nd: doji gapping down
            and ca.real_body[i - 1] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_doji_total
            and body_hi[i - 1] < body_lo[i - 2]
            # 3rd: longer than short, white, closing well within 1st rb
            and ca.real_body[i] > AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and ca.color[i] == 1
            and ca.close[i] > ca.close[i - 2] + ca.real_body[i - 2] * penetration
        ):
            out[i] = 100

        # Update trailing windows
        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail]
        body_doji_total += arr_bd[i - 1] - arr_bd[body_doji_trail]
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        body_long_trail += 1
        body_doji_trail += 1
        body_short_trail += 1


def cdl_morningdojistar(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    penetration: float = 0.3,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Morning Doji Star.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        penetration: Percentage of penetration within the first candle's
            real body. Default: 0.3
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_MORNINGDOJISTAR expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_MORNINGDOJISTAR",
        "MORNINGDOJISTAR",
        scalar=scalar,
        offset=offset,
        talib=talib,
        penetration=penetration,
    )
