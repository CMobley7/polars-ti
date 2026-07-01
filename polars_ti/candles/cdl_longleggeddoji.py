# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_LONGLEGGEDDOJI Implementation
# =============================================================================
"""Candle Pattern: Long Legged Doji."""

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
    body_doji_period = candle_avg_period(CandleSetting.BodyDoji)
    shadow_long_period = candle_avg_period(CandleSetting.ShadowLong)
    lookback = max(body_doji_period, shadow_long_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    arr_sl = ca._ranges[CandleSetting.ShadowLong]

    body_doji_trail = start_idx - body_doji_period
    shadow_long_trail = start_idx - shadow_long_period
    body_doji_total = float(arr_bd[body_doji_trail:start_idx].sum())
    shadow_long_total = float(arr_sl[shadow_long_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_doji_total and (
            ca.lower_shadow[i] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i]
            or ca.upper_shadow[i] > AVG_FACTOR[CandleSetting.ShadowLong] * arr_sl[i]
        ):
            out[i] = 100

        # Update trailing windows
        body_doji_total += arr_bd[i] - arr_bd[body_doji_trail]
        shadow_long_total += arr_sl[i] - arr_sl[shadow_long_trail]
        body_doji_trail += 1
        shadow_long_trail += 1


def cdl_longleggeddoji(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Long Legged Doji.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_LONGLEGGEDDOJI expression (+scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_LONGLEGGEDDOJI",
        "LONGLEGGEDDOJI",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
