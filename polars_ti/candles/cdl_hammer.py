# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_HAMMER Implementation
# =============================================================================
"""Candle Pattern: Hammer."""

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
    body_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)  # 0
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    near_period = candle_avg_period(CandleSetting.Near)

    # Lookback: max of all periods + 1 (extra bar for i-1 reference)
    lookback = max(body_period, shadow_long_period, shadow_vs_period, near_period) + 1
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_nr = ca._ranges[CandleSetting.Near]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]
    body_lo = ca.body_low

    # Trailing indices
    body_trail = start_idx - body_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vs_trail = start_idx - shadow_vs_period
    near_trail = start_idx - 1 - near_period  # Near uses i-1

    # Seed totals
    body_total = float(arr_bs[body_trail:start_idx].sum())
    shadow_long_total = float(arr_sl[shadow_long_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    near_total = float(arr_nr[near_trail : start_idx - 1].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_total
            and ca.lower_shadow[i] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i]
            and ca.upper_shadow[i] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
            and (body_lo[i] <= ca.low[i - 1] + AVG_FACTOR[CandleSetting.Near] * near_total)
        ):
            out[i] = 100

        # Update trailing windows AFTER pattern check
        body_total += arr_bs[i] - arr_bs[body_trail]
        shadow_long_total += arr_sl[i] - arr_sl[shadow_long_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        near_total += arr_nr[i - 1] - arr_nr[near_trail]

        body_trail += 1
        shadow_long_trail += 1
        shadow_vs_trail += 1
        near_trail += 1


def cdl_hammer(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Hammer.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_HAMMER expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HAMMER",
        "HAMMER",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
