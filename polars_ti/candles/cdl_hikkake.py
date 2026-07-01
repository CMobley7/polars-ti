# -*- coding: utf-8 -*-
# =============================================================================
# Polars CDL_HIKKAKE Implementation
# =============================================================================
"""Candle Pattern: Hikkake."""

from typing import Any

import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.candles._cdl_math import CandleArrays, run_pattern


def _hikkake_is_setup(H, L, i):
    """Check if bars at index i form a Hikkake setup (inside bar + breakout direction)."""
    return (
        H[i - 1] < H[i - 2]
        and L[i - 1] > L[i - 2]
        and ((H[i] < H[i - 1] and L[i] < L[i - 1]) or (H[i] > H[i - 1] and L[i] > L[i - 1]))
    )


def _hikkake_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
    """Check if bar i confirms a previously detected Hikkake pattern."""
    return i <= pattern_idx + 3 and (
        (pattern_result > 0 and C[i] > H[pattern_idx - 1]) or (pattern_result < 0 and C[i] < L[pattern_idx - 1])
    )


def _detect(ca: CandleArrays, out: np.ndarray, **kwargs: Any) -> None:
    # Lookback = 5  (no candle settings used)
    lookback = 5
    start_idx = lookback
    if start_idx >= len(out):
        return

    H = ca.high
    L = ca.low
    C = ca.close

    pattern_idx = 0
    pattern_result = 0

    # Warm-up: scan the 3 bars before start_idx to initialize state
    # (i runs from start_idx-3 to start_idx-1)
    for i in range(start_idx - 3, start_idx):
        if _hikkake_is_setup(H, L, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
        else:
            # Search for confirmation
            if _hikkake_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
                pattern_idx = 0

    # Main loop
    for i in range(start_idx, len(out)):
        if _hikkake_is_setup(H, L, i):
            pattern_result = 100 * (1 if H[i] < H[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif _hikkake_is_confirmed(pattern_result, pattern_idx, C, H, L, i):
            out[i] = pattern_result + 100 * (1 if pattern_result > 0 else -1)
            pattern_idx = 0


def cdl_hikkake(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    scalar: float = 100.0,
    offset: int = 0,
    talib: bool = True,
) -> PlExpr:
    """Polars: Candle Pattern - Hikkake.

    A stateful pattern that detects an inside bar (bar 2 has lower high
    and higher low than bar 1) followed by a breakout bar (bar 3) that
    exceeds the inside bar's range. Confirmation occurs within the next
    3 bars when price closes beyond the inside bar's high/low. The pattern
    bar outputs +/-100 and the confirmation bar outputs +/-200.

    Args:
        open_: Column name or pl.Expr for 'open' prices.
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        close: Column name or pl.Expr for 'close' prices.
        scalar: Result multiplier. Default: 100.0
        offset: Shift result by N periods. Default: 0
        talib: Use TA-Lib when installed. Default: True

    Returns:
        pl.Expr: CDL_HIKKAKE expression (pattern signals / 0).
    """
    return run_pattern(
        open_,
        high,
        low,
        close,
        _detect,
        "CDL_HIKKAKE",
        "HIKKAKE",
        scalar=scalar,
        offset=offset,
        talib=talib,
    )
