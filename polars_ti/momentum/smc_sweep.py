# -*- coding: utf-8 -*-
# =============================================================================
# Polars SMC_SWEEP (Smart Money Concept Liquidity Sweep) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def smc_sweep(
    open_: IntoExpr = "open",
    high: IntoExpr = "high",
    low: IntoExpr = "low",
    close: IntoExpr = "close",
    length: int = 15,
    wick_mult: float = 1.5,
    offset: int = 0,
) -> PlExpr:
    """Polars: Smart Money Concept Liquidity Sweep (SMC_SWEEP)

    Identifies when price sweeps below a swing low (or above a swing high),
    violently rejects it, leaving a long wick and closing in the opposite
    direction.

    Sources:
        Smart Money Concept / ICT trading methodology

    Calculation:
        swing_low  = rolling min of low over length bars (shifted 1)
        swing_high = rolling max of high over length bars (shifted 1)
        body       = abs(close - open)
        lower_wick = min(open, close) - low
        upper_wick = high - max(open, close)
        Bull: low < swing_low AND close > swing_low AND close > open
              AND lower_wick > body * wick_mult -> +1
        Bear: high > swing_high AND close < swing_high AND close < open
              AND upper_wick > body * wick_mult -> -1

    Args:
        open_: Column name or pl.Expr for 'open' prices. Default: "open"
        high: Column name or pl.Expr for 'high' prices. Default: "high"
        low: Column name or pl.Expr for 'low' prices. Default: "low"
        close: Column name or pl.Expr for 'close' prices. Default: "close"
        length: Swing high/low lookback period. Default: 15
        wick_mult: Wick-to-body ratio multiplier. Default: 1.5
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: 1 (Bullish Sweep), -1 (Bearish Sweep), 0 (None).
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    if open_expr is None or high_expr is None or low_expr is None or close_expr is None:
        return None

    swing_low = low_expr.rolling_min(window_size=length).shift(1)
    swing_high = high_expr.rolling_max(window_size=length).shift(1)

    body = (close_expr - open_expr).abs()
    lower_wick = pl.min_horizontal(open_expr, close_expr) - low_expr
    upper_wick = high_expr - pl.max_horizontal(open_expr, close_expr)

    bull = (
        (low_expr < swing_low) & (close_expr > swing_low) & (close_expr > open_expr) & (lower_wick > body * wick_mult)
    )
    bear = (
        (high_expr > swing_high)
        & (close_expr < swing_high)
        & (close_expr < open_expr)
        & (upper_wick > body * wick_mult)
    )

    # bull contributes +1, bear contributes -1. Comparisons with NaN/null swing
    # values yield null -> treat as 0 (no sweep), matching numpy.where on NaN.
    sweep = bull.fill_null(False).cast(pl.Int64) - bear.fill_null(False).cast(pl.Int64)

    if offset != 0:
        sweep = sweep.shift(offset)

    _props = f"_{length}_{round(wick_mult, 4)}"
    return sweep.alias(f"SMC_SWEEP{_props}")
