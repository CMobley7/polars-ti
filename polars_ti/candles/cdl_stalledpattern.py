# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_STALLEDPATTERN Implementation
# =============================================================================
"""Candle Pattern: Stalled Pattern."""

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
    # Lookback: max(max(BodyLong, BodyShort),
    #               max(ShadowVeryShort, Near)) + 2
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    near_period = candle_avg_period(CandleSetting.Near)

    lookback = (
        max(
            max(body_long_period, body_short_period),
            max(svs_period, near_period),
        )
        + 2
    )
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_nr = ca._ranges[CandleSetting.Near]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    body_long_trail = start_idx - body_long_period
    body_short_trail = start_idx - body_short_period
    svs_trail = start_idx - svs_period
    near_trail = start_idx - near_period

    # Seed BodyLong totals for i-2 and i-1 (indices 2, 1)
    body_long_total_2 = float(arr_bl[body_long_trail - 2 : start_idx - 2].sum())
    body_long_total_1 = float(arr_bl[body_long_trail - 1 : start_idx - 1].sum())
    # Seed BodyShort total for i (index 0 / current bar)
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    # Seed ShadowVeryShort total for i-1
    svs_total = float(arr_svs[svs_trail - 1 : start_idx - 1].sum())
    # Seed Near totals for i-2 and i-1 (indices 2, 1)
    near_total_2 = float(arr_nr[near_trail - 2 : start_idx - 2].sum())
    near_total_1 = float(arr_nr[near_trail - 1 : start_idx - 1].sum())
    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st white
            ca.color[i - 2] == 1
            # 2nd white
            and ca.color[i - 1] == 1
            # 3rd white
            and ca.color[i] == 1
            # Consecutive higher closes
            and C[i] > C[i - 1]
            and C[i - 1] > C[i - 2]
            # 1st: long real body
            and ca.real_body[i - 2] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total_2
            # 2nd: long real body
            and ca.real_body[i - 1] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total_1
            # 2nd: very short upper shadow
            and ca.upper_shadow[i - 1] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total
            # 2nd opens within/near 1st real body: opens above 1st open
            and O[i - 1] > O[i - 2]
            # 2nd opens at or below 1st close + Near average
            and O[i - 1] <= C[i - 2] + AVG_FACTOR[CandleSetting.Near] * near_total_2
            # 3rd: small real body
            and ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            # 3rd rides on the shoulder of 2nd real body
            and O[i] >= C[i - 1] - ca.real_body[i] - AVG_FACTOR[CandleSetting.Near] * near_total_1
        ):
            out[i] = -100

        # Update BodyLong and Near totals (indices 2 and 1)
        body_long_total_2 += arr_bl[i - 2] - arr_bl[body_long_trail - 2]
        body_long_total_1 += arr_bl[i - 1] - arr_bl[body_long_trail - 1]
        near_total_2 += arr_nr[i - 2] - arr_nr[near_trail - 2]
        near_total_1 += arr_nr[i - 1] - arr_nr[near_trail - 1]
        # Update BodyShort total (index 0 / current bar)
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        # Update ShadowVeryShort total (index 1 / bar i-1)
        svs_total += arr_svs[i - 1] - arr_svs[svs_trail - 1]

        body_long_trail += 1
        body_short_trail += 1
        svs_trail += 1
        near_trail += 1


def cdl_stalledpattern(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Stalled Pattern.

    Three white candlesticks with consecutively higher closes. The first
    two have long real bodies; the second has a very short upper shadow
    and opens within or near the first's real body. The third has a
    small real body that gaps away or rides on the shoulder of the
    second's body, signaling a potential reversal.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_STALLEDPATTERN expression (-scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_STALLEDPATTERN",
        "STALLEDPATTERN",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
