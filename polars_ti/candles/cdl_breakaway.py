# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_BREAKAWAY Implementation
# =============================================================================
"""Candle Pattern: Breakaway."""

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
    # Lookback: TA_CANDLEAVGPERIOD(BodyLong) + 4
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    lookback = body_long_period + 4
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    body_hi = ca.body_high
    body_lo = ca.body_low

    # Trailing index for BodyLong setting applied to i-4
    body_long_trail = start_idx - body_long_period

    # Seed BodyLong total: sum candle_range(BodyLong, i-4)
    # for i from body_long_trail to start_idx-1
    body_long_total = float(arr_bl[body_long_trail - 4 : start_idx - 4].sum())
    O = ca.open
    H = ca.high
    L = ca.low
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st: long body
            ca.real_body[i - 4] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            # 1st, 2nd, 4th same color; 5th opposite
            and ca.color[i - 4] == ca.color[i - 3]
            and ca.color[i - 3] == ca.color[i - 1]
            and ca.color[i - 1] == -ca.color[i]
            and (
                (
                    # When 1st is black:
                    ca.color[i - 4] == -1
                    # 2nd gaps down
                    and body_hi[i - 3] < body_lo[i - 4]
                    # 3rd has lower high and low than 2nd
                    and H[i - 2] < H[i - 3]
                    and L[i - 2] < L[i - 3]
                    # 4th has lower high and low than 3rd
                    and H[i - 1] < H[i - 2]
                    and L[i - 1] < L[i - 2]
                    # 5th closes inside the gap
                    and C[i] > O[i - 3]
                    and C[i] < C[i - 4]
                )
                or (
                    # When 1st is white:
                    ca.color[i - 4] == 1
                    # 2nd gaps up
                    and body_lo[i - 3] > body_hi[i - 4]
                    # 3rd has higher high and low than 2nd
                    and H[i - 2] > H[i - 3]
                    and L[i - 2] > L[i - 3]
                    # 4th has higher high and low than 3rd
                    and H[i - 1] > H[i - 2]
                    and L[i - 1] > L[i - 2]
                    # 5th closes inside the gap
                    and C[i] < O[i - 3]
                    and C[i] > C[i - 4]
                )
            )
        ):
            out[i] = ca.color[i] * 100

        # Update: add current, subtract trailing (both reference i-4)
        body_long_total += arr_bl[i - 4] - arr_bl[body_long_trail - 4]
        body_long_trail += 1


def cdl_breakaway(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Breakaway.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_BREAKAWAY expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_BREAKAWAY",
        "BREAKAWAY",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
