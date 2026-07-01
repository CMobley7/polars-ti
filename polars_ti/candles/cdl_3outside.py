# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_3OUTSIDE Implementation
# =============================================================================
"""Candle Pattern: Three Outside Up/Down."""

from typing import Any

import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles._cdl_math import CandleArrays, run_pattern


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    start_idx = 3
    if start_idx >= len(out):
        return

    for i in range(start_idx, len(out)):
        if (
            ca.color[i - 1] == 1
            and ca.color[i - 2] == -1
            and ca.close[i - 1] > ca.open[i - 2]
            and ca.open[i - 1] < ca.close[i - 2]
            and ca.close[i] > ca.close[i - 1]
        ) or (
            ca.color[i - 1] == -1
            and ca.color[i - 2] == 1
            and ca.open[i - 1] > ca.close[i - 2]
            and ca.close[i - 1] < ca.open[i - 2]
            and ca.close[i] < ca.close[i - 1]
        ):
            out[i] = ca.color[i - 1] * 100


def cdl_3outside(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Three Outside Up/Down.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_3OUTSIDE expression (+scalar / -scalar / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_3OUTSIDE",
        "3OUTSIDE",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
