# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_SHORTLINE Implementation
# =============================================================================
"""Candle Pattern: Short Line Candle."""

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
    body_short_period = candle_avg_period(CandleSetting.BodyShort)
    shadow_short_period = candle_avg_period(CandleSetting.ShadowShort)
    lookback = max(body_short_period, shadow_short_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bs = ca._ranges[CandleSetting.BodyShort]
    arr_ss = ca._ranges[CandleSetting.ShadowShort]

    body_short_trail = start_idx - body_short_period
    shadow_short_trail = start_idx - shadow_short_period
    body_short_total = float(arr_bs[body_short_trail:start_idx].sum())
    shadow_short_total = float(arr_ss[shadow_short_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] < AVG_FACTOR[CandleSetting.BodyShort] * body_short_total
            and ca.upper_shadow[i] < AVG_FACTOR[CandleSetting.ShadowShort] * shadow_short_total
            and ca.lower_shadow[i] < AVG_FACTOR[CandleSetting.ShadowShort] * shadow_short_total
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_short_total += arr_bs[i] - arr_bs[body_short_trail]
        shadow_short_total += arr_ss[i] - arr_ss[shadow_short_trail]
        body_short_trail += 1
        shadow_short_trail += 1


def cdl_shortline(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Short Line Candle.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_SHORTLINE expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_SHORTLINE",
        "SHORTLINE",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
