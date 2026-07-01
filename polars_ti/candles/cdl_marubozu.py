# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_MARUBOZU Implementation
# =============================================================================
"""Candle Pattern: Marubozu."""

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
    body_long_period = candle_avg_period(CandleSetting.BodyLong)
    shadow_vs_period = candle_avg_period(CandleSetting.ShadowVeryShort)
    lookback = max(body_long_period, shadow_vs_period)
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bl = ca._ranges[CandleSetting.BodyLong]
    arr_svs = ca._ranges[CandleSetting.ShadowVeryShort]

    body_long_trail = start_idx - body_long_period
    shadow_vs_trail = start_idx - shadow_vs_period
    body_long_total = float(arr_bl[body_long_trail:start_idx].sum())
    shadow_vs_total = float(arr_svs[shadow_vs_trail:start_idx].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i] > AVG_FACTOR[CandleSetting.BodyLong] * body_long_total
            and ca.upper_shadow[i] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
            and ca.lower_shadow[i] < AVG_FACTOR[CandleSetting.ShadowVeryShort] * shadow_vs_total
        ):
            out[i] = ca.color[i] * 100

        # Update trailing windows
        body_long_total += arr_bl[i] - arr_bl[body_long_trail]
        shadow_vs_total += arr_svs[i] - arr_svs[shadow_vs_trail]
        body_long_trail += 1
        shadow_vs_trail += 1


def cdl_marubozu(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Marubozu.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_MARUBOZU expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_MARUBOZU",
        "MARUBOZU",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
