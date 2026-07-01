# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_TRISTAR Implementation
# =============================================================================
"""Candle Pattern: Tristar Pattern."""

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
    lookback = body_doji_period + 2
    start_idx = lookback
    if start_idx >= len(out):
        return

    arr_bd = ca._ranges[CandleSetting.BodyDoji]
    body_hi = ca.body_high
    body_lo = ca.body_low

    body_trail = start_idx - 2 - body_doji_period
    body_total = float(arr_bd[body_trail : start_idx - 2].sum())
    for i in range(start_idx, len(out)):
        if (
            ca.real_body[i - 2] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_total
            and ca.real_body[i - 1] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_total
            and ca.real_body[i] <= AVG_FACTOR[CandleSetting.BodyDoji] * body_total
        ):
            if body_lo[i - 1] > body_hi[i - 2] and body_hi[i] < max(ca.open[i - 1], ca.close[i - 1]):
                out[i] = -100
            if body_hi[i - 1] < body_lo[i - 2] and min(ca.open[i], ca.close[i]) > body_lo[i - 1]:
                out[i] = 100

        body_total += arr_bd[i - 2] - arr_bd[body_trail]
        body_trail += 1


def cdl_tristar(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Tristar.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_TRISTAR expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_TRISTAR",
        "TRISTAR",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
