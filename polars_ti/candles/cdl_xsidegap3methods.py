# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_XSIDEGAP3METHODS Implementation
# =============================================================================
"""Candle Pattern: Upside/Downside Gap Three Methods."""

from typing import Any

import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles._cdl_math import CandleArrays, run_pattern


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = 2
    if start_idx >= len(out):
        return

    body_hi = ca.body_high
    body_lo = ca.body_low

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 2] == ca.color[i - 1]
            and ca.color[i - 1] == -ca.color[i]
            and ca.open[i] < body_hi[i - 1]
            and ca.open[i] > body_lo[i - 1]
            and ca.close[i] < body_hi[i - 2]
            and ca.close[i] > body_lo[i - 2]
            and (
                (ca.color[i - 2] == 1 and body_lo[i - 1] > body_hi[i - 2])
                or (ca.color[i - 2] == -1 and body_hi[i - 1] < body_lo[i - 2])
            )
        ):
            out[i] = ca.color[i - 2] * 100


def cdl_xsidegap3methods(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Upside/Downside Gap Three Methods.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_XSIDEGAP3METHODS expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_XSIDEGAP3METHODS",
        "XSIDEGAP3METHODS",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
