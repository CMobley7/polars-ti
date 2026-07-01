# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_3WHITESOLDIERS Implementation
# =============================================================================
"""Candle Pattern: Three Advancing White Soldiers."""

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
    svs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    far_period = candle_avg_period(CandleSetting.Far)
    near_period = candle_avg_period(CandleSetting.Near)

    # Lookback: max(all avg periods) + 2
    lookback = max(svs_period, body_short_period, far_period, near_period) + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_fr = ca._ranges[CandleSetting.Far]
    arr_nr = ca._ranges[CandleSetting.Near]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    # Trailing indices
    svs_trail = start_idx - svs_period
    near_trail = start_idx - near_period
    far_trail = start_idx - far_period
    body_short_trail = start_idx - body_short_period

    # Seed ShadowVeryShort totals [2], [1], [0]
    svs_total_2 = float(arr_svs[svs_trail - 2 : start_idx - 2].sum())
    svs_total_1 = float(arr_svs[svs_trail - 1 : start_idx - 1].sum())
    svs_total_0 = float(arr_svs[svs_trail:start_idx].sum())
    # Seed Near totals [2], [1] (not [0])
    near_total_2 = float(arr_nr[near_trail - 2 : start_idx - 2].sum())
    near_total_1 = float(arr_nr[near_trail - 1 : start_idx - 1].sum())
    # Seed Far totals [2], [1] (not [0])
    far_total_2 = float(arr_fr[far_trail - 2 : start_idx - 2].sum())
    far_total_1 = float(arr_fr[far_trail - 1 : start_idx - 1].sum())
    # Seed BodyShort total
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # 1st white
            ca.color[i - 2] == 1
            # 1st: very short upper shadow
            and ca.upper_shadow[i - 2] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total_2
            # 2nd white
            and ca.color[i - 1] == 1
            # 2nd: very short upper shadow
            and ca.upper_shadow[i - 1] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total_1
            # 3rd white
            and ca.color[i] == 1
            # 3rd: very short upper shadow
            and ca.upper_shadow[i] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * svs_total_0
            # Consecutive higher closes
            and C[i] > C[i - 1]
            and C[i - 1] > C[i - 2]
            # 2nd opens within/near 1st real body
            and O[i - 1] > O[i - 2]
            and O[i - 1] <= C[i - 2] + AVG_FACTOR[CandleSetting.Near] * near_total_2
            # 3rd opens within/near 2nd real body
            and O[i] > O[i - 1]
            and O[i] <= C[i - 1] + AVG_FACTOR[CandleSetting.Near] * near_total_1
            # 2nd not far shorter than 1st
            and ca.real_body[i - 1] > ca.real_body[i - 2] - AVG_FACTOR[CandleSetting.Far] * far_total_2
            # 3rd not far shorter than 2nd
            and ca.real_body[i] > ca.real_body[i - 1] - AVG_FACTOR[CandleSetting.Far] * far_total_1
            # 3rd: not short real body
            and ca.real_body[i] > AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
        ):
            out[i] = 100  # Always bullish

        # Update ShadowVeryShort totals [2], [1], [0]
        svs_total_2 += arr_svs[i - 2] - arr_svs[svs_trail - 2]
        svs_total_1 += arr_svs[i - 1] - arr_svs[svs_trail - 1]
        svs_total_0 += arr_svs[i] - arr_svs[svs_trail]

        # Update Far and Near totals [2], [1]
        far_total_2 += arr_fr[i - 2] - arr_fr[far_trail - 2]
        far_total_1 += arr_fr[i - 1] - arr_fr[far_trail - 1]
        near_total_2 += arr_nr[i - 2] - arr_nr[near_trail - 2]
        near_total_1 += arr_nr[i - 1] - arr_nr[near_trail - 1]

        # Update BodyShort total
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]

        svs_trail += 1
        near_trail += 1
        far_trail += 1
        body_short_trail += 1


def cdl_3whitesoldiers(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Three Advancing White Soldiers.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_3WHITESOLDIERS expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3WHITESOLDIERS",
        "3WHITESOLDIERS",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
