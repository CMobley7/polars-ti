# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_3LINESTRIKE Implementation
# =============================================================================
"""Candle Pattern: Three-Line Strike."""

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
    # Lookback: TA_CANDLEAVGPERIOD(Near) + 3
    near_period = candle_avg_period(CandleSetting.Near)
    lookback = near_period + 3
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_nr = ca._ranges[CandleSetting.Near]

    near_trail = start_idx - near_period

    # Seed Near totals for i-3 and i-2 (indices 3, 2)
    near_total_3 = float(arr_nr[near_trail - 3 : start_idx - 3].sum())
    near_total_2 = float(arr_nr[near_trail - 2 : start_idx - 2].sum())
    O = ca.open
    C = ca.close

    for i in range(start_idx, len(out)):
        if (
            # Three candles with same color
            ca.color[i - 3] == ca.color[i - 2]
            and ca.color[i - 2] == ca.color[i - 1]
            # 4th opposite color
            and ca.color[i] == -ca.color[i - 1]
            # 2nd opens within/near 1st real body
            and O[i - 2] >= min(O[i - 3], C[i - 3]) - AVG_FACTOR[CandleSetting.Near] * near_total_3
            and O[i - 2] <= max(O[i - 3], C[i - 3]) + AVG_FACTOR[CandleSetting.Near] * near_total_3
            # 3rd opens within/near 2nd real body
            and O[i - 1] >= min(O[i - 2], C[i - 2]) - AVG_FACTOR[CandleSetting.Near] * near_total_2
            and O[i - 1] <= max(O[i - 2], C[i - 2]) + AVG_FACTOR[CandleSetting.Near] * near_total_2
            and (
                (
                    # If three white
                    ca.color[i - 1] == 1
                    # Consecutive higher closes
                    and C[i - 1] > C[i - 2]
                    and C[i - 2] > C[i - 3]
                    # 4th opens above prior close
                    and O[i] > C[i - 1]
                    # 4th closes below 1st open
                    and C[i] < O[i - 3]
                )
                or (
                    # If three black
                    ca.color[i - 1] == -1
                    # Consecutive lower closes
                    and C[i - 1] < C[i - 2]
                    and C[i - 2] < C[i - 3]
                    # 4th opens below prior close
                    and O[i] < C[i - 1]
                    # 4th closes above 1st open
                    and C[i] > O[i - 3]
                )
            )
        ):
            out[i] = ca.color[i - 1] * 100

        # Update totals: add current range, subtract trailing range
        near_total_3 += arr_nr[i - 3] - arr_nr[near_trail - 3]
        near_total_2 += arr_nr[i - 2] - arr_nr[near_trail - 2]
        near_trail += 1


def cdl_3linestrike(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Three-Line Strike.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_3LINESTRIKE expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3LINESTRIKE",
        "3LINESTRIKE",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
