# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_3STARSINSOUTH Implementation
# =============================================================================
"""Candle Pattern: Three Stars In The South."""

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
    # Settings and their avg periods
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    shadow_vshort_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)

    # Lookback: max(all avg periods) + 2
    lookback = (
        max(
            shadow_vshort_period,
            shadow_long_period,
            body_long_period,
            body_short_period,
        )
        + 2
    )
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    # Trailing indices
    body_long_trail = start_idx - body_long_period
    shadow_long_trail = start_idx - shadow_long_period
    shadow_vshort_trail = start_idx - shadow_vshort_period
    body_short_trail = start_idx - body_short_period

    # Seed totals
    # BodyLong: applied to i-2
    body_long_total = float(arr_bl[body_long_trail - 2 : start_idx - 2].sum())
    # ShadowLong: applied to i-2
    shadow_long_total = float(arr_sl[shadow_long_trail - 2 : start_idx - 2].sum())
    # ShadowVeryShort[1]: applied to i-1; ShadowVeryShort[0]: applied to i
    shadow_vshort_total_1 = float(arr_svs[shadow_vshort_trail - 1 : start_idx - 1].sum())
    shadow_vshort_total_0 = float(arr_svs[shadow_vshort_trail:start_idx].sum())
    # BodyShort: applied to i
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    O = ca.open
    H = ca.high
    L = ca.low
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # All three candles are black
            ca.color[i - 2] == -1
            and ca.color[i - 1] == -1
            and ca.color[i] == -1
            # 1st: long body
            and ca.real_body[i - 2] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            # 1st: long lower shadow
            and ca.lower_shadow[i - 2] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i - 2]
            # 2nd: smaller candle
            and ca.real_body[i - 1] < ca.real_body[i - 2]
            # 2nd: opens higher than 1st close but within 1st range
            and O[i - 1] > C[i - 2]
            and O[i - 1] <= H[i - 2]
            # 2nd: trades lower than 1st close
            and L[i - 1] < C[i - 2]
            # 2nd: but not lower than 1st low
            and L[i - 1] >= L[i - 2]
            # 2nd: has a lower shadow (not very short)
            and ca.lower_shadow[i - 1] > AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vshort_total_1
            # 3rd: small marubozu (short body)
            and ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            # 3rd: very short lower shadow
            and ca.lower_shadow[i] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vshort_total_0
            # 3rd: very short upper shadow
            and ca.upper_shadow[i] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vshort_total_0
            # 3rd: engulfed by 2nd candle's range
            and L[i] > L[i - 1]
            and H[i] < H[i - 1]
        ):
            out[i] = 100  # Always bullish

        # Update totals
        body_long_total += arr_bl[i - 2] - arr_bl[body_long_trail - 2]
        shadow_long_total += arr_sl[i - 2] - arr_sl[shadow_long_trail - 2]

        shadow_vshort_total_1 += arr_svs[i - 1] - arr_svs[shadow_vshort_trail - 1]
        shadow_vshort_total_0 += arr_svs[i] - arr_svs[shadow_vshort_trail]

        body_short_total += arr_bs[i] - arr_bs[body_short_trail]

        body_long_trail += 1
        shadow_long_trail += 1
        shadow_vshort_trail += 1
        body_short_trail += 1


def cdl_3starsinsouth(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Three Stars In The South.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_3STARSINSOUTH expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3STARSINSOUTH",
        "3STARSINSOUTH",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
