# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_HIKKAKEMOD Implementation
# =============================================================================
"""Candle Pattern: Modified Hikkake."""

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


def _hikkakemod_is_setup(H, L, C, near_total, i):
    """Check if bars at index i form a modified Hikkake setup with near condition."""
    avg = AVG_FACTOR[CandleSetting.Near]
    return (
        H[i - 2] < H[i - 3]
        and L[i - 2] > L[i - 3]
        and H[i - 1] < H[i - 2]
        and L[i - 1] > L[i - 2]
        and (
            (H[i] < H[i - 1] and L[i] < L[i - 1] and C[i - 2] <= L[i - 2] + avg * near_total)
            or (H[i] > H[i - 1] and L[i] > L[i - 1] and C[i - 2] >= H[i - 2] - avg * near_total)
        )
    )


def _hikkakemod_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
    """Check if bar i confirms a previously detected modified Hikkake pattern."""
    return i <= pattern_idx + 3 and (
        (pattern_result > 0 and C[i] > H[pattern_idx - 1]) or (pattern_result < 0 and C[i] < L[pattern_idx - 1])
    )


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback: max(1, TA_CANDLEAVGPERIOD(Near)) + 5
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = max(1, near_period) + 5
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_nr = ca._ranges[CandleSetting.Near]

    H = ca.high
    L = ca.low
    C = ca.close

    # Near trailing: seeds for the Near setting applied at i-2
    # NearTrailingIdx = startIdx - 3 - near_period
    near_trail = start_idx - 3 - near_period

    # Seed Near total
    near_total = 0.0
    j = near_trail
    while j < start_idx - 3:
        near_total += arr_nr[j - 2]
        j += 1

    pattern_idx = 0
    pattern_result = 0

    # Warm-up: scan the 3 bars before start_idx
    for i in range(start_idx - 3, start_idx):
        if _hikkakemod_is_setup(H, L, C, near_total, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
        else:
            # Search for confirmation
            if _hikkakemod_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
                pattern_idx = 0

        near_total += arr_nr[i - 2] - arr_nr[near_trail - 2]
        near_trail += 1

    # Main loop
    for i in range(start_idx, len(out)):
        if _hikkakemod_is_setup(H, L, C, near_total, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif _hikkakemod_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
            out[i] = pattern_result + 100 * (1 if pattern_result > 0 else -1)
            pattern_idx = 0

        near_total += arr_nr[i - 2] - arr_nr[near_trail - 2]
        near_trail += 1


def cdl_hikkakemod(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Modified Hikkake.

    Like the standard Hikkake but adds a requirement that the second
    candle has a close near its low (bullish) or near its high (bearish),
    and requires two nested inside bars (bar 2 inside bar 1, bar 3 inside
    bar 2) before the breakout bar. The pattern bar outputs +/-100 and the
    confirmation bar outputs +/-200.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_HIKKAKEMOD expression (pattern signals / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HIKKAKEMOD",
        "HIKKAKEMOD",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
