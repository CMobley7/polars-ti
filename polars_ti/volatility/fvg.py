# -*- coding: utf-8 -*-
from numpy import float64, full, nan
from numba import njit

from polars_ti._typing import Array


@njit(cache=True)
def nb_fvg(
    np_open: Array,
    np_high: Array,
    np_low: Array,
    np_close: Array,
    min_gap: float64,
) -> tuple[Array, Array, Array]:
    """Numba-optimized Fair Value Gap calculation.

    Identifies Fair Value Gaps using a 3-candle pattern:
    - Bullish FVG: low[i+1] > high[i-1] during an up candle
    - Bearish FVG: high[i+1] < low[i-1] during a down candle
    """
    n = np_open.size
    fvg_high = full(n, nan)
    fvg_low = full(n, nan)
    fvg_type = full(n, nan)

    for i in range(1, n - 1):
        # Bullish candle (close > open) with upward gap
        if np_close[i] > np_open[i]:
            gap = np_low[i + 1] - np_high[i - 1]
            if gap > min_gap * np_close[i]:
                fvg_low[i] = np_high[i - 1]
                fvg_high[i] = np_low[i + 1]
                fvg_type[i] = 1.0  # bullish
        # Bearish candle (close < open) with downward gap
        elif np_close[i] < np_open[i]:
            gap = np_low[i - 1] - np_high[i + 1]
            if gap > min_gap * np_close[i]:
                fvg_low[i] = np_high[i + 1]
                fvg_high[i] = np_low[i - 1]
                fvg_type[i] = -1.0  # bearish

    return fvg_high, fvg_low, fvg_type


# =============================================================================
# Polars FVG Implementation (Numba @njit via map_batches)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def fvg(
    open_: IntoExpr,
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    min_gap: float = 0.0,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Fair Value Gap (FVG)

    Uses Numba @njit kernel via map_batches.

    An FVG occurs when a strong momentum candle creates an imbalance,
    forming a price inefficiency that the market may revisit.

    Sources:
        https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/Inversion-Fair-Value-Gaps
        https://capital.com/fair-value-gap-in-trading

    Args:
        open_: Column name or pl.Expr for 'open'
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        min_gap: Minimum percentage gap size. Default: 0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with FVGh, FVGl, FVGt columns
    """
    open_expr = v_expr(open_)
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if open_expr is None or high_expr is None or low_expr is None or close_expr is None:
        return None

    min_gap_pct = min_gap / 100.0
    _min_gap = min_gap_pct
    _offset = offset
    _min_gap_int = int(min_gap)

    def compute_fvg(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        np_open = df["_open"].to_numpy().astype(np.float64)
        np_high = df["_high"].to_numpy().astype(np.float64)
        np_low = df["_low"].to_numpy().astype(np.float64)
        np_close = df["_close"].to_numpy().astype(np.float64)

        fvg_high, fvg_low, fvg_type = nb_fvg(np_open, np_high, np_low, np_close, _min_gap)

        if _offset != 0:
            fvg_high = np.roll(fvg_high, _offset)
            fvg_low = np.roll(fvg_low, _offset)
            fvg_type = np.roll(fvg_type, _offset)
            if _offset > 0:
                fvg_high[:_offset] = np.nan
                fvg_low[:_offset] = np.nan
                fvg_type[:_offset] = np.nan

        return pl.DataFrame({"fvg_high": fvg_high, "fvg_low": fvg_low, "fvg_type": fvg_type}).to_struct("fvg")

    _props = f"_{_min_gap_int}"

    return (
        pl.struct(
            [
                open_expr.alias("_open"),
                high_expr.alias("_high"),
                low_expr.alias("_low"),
                close_expr.alias("_close"),
            ]
        )
        .map_batches(
            compute_fvg,
            return_dtype=pl.Struct({"fvg_high": pl.Float64, "fvg_low": pl.Float64, "fvg_type": pl.Float64}),
        )
        .alias(f"FVG{_props}")
    )
