# -*- coding: utf-8 -*-
# =============================================================================
# Polars MASSI Implementation (Composition: pl_ema + Numba for cascade)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils._math import pl_non_zero_range
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_massi_from_ema1(ema1: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """MASSI calculation from first EMA - handles second EMA, ratio, and rolling sum."""
    n = len(ema1)
    
    # Second EMA with presma=True behavior (SMA of first 'fast' valid values)
    ema2 = np.empty(n, dtype=np.float64)
    ema2[:] = np.nan
    
    # Find first valid index
    first_valid = -1
    for i in range(n):
        if not np.isnan(ema1[i]):
            first_valid = i
            break
    
    if first_valid >= 0 and first_valid + fast <= n:
        # SMA of first 'fast' valid values for presma=True behavior
        sma_start = first_valid + fast - 1
        sma_sum = 0.0
        for i in range(first_valid, first_valid + fast):
            sma_sum += ema1[i]
        ema2[sma_start] = sma_sum / fast
        
        # EMA for rest
        alpha = 2.0 / (fast + 1)
        for i in range(sma_start + 1, n):
            if np.isnan(ema1[i]):
                ema2[i] = ema2[i - 1]
            else:
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
    
    # Ratio
    ratio = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(ema1[i]) or np.isnan(ema2[i]) or ema2[i] == 0:
            ratio[i] = np.nan
        else:
            ratio[i] = ema1[i] / ema2[i]
    
    # Rolling sum
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    for i in range(slow - 1, n):
        window_sum = 0.0
        valid = True
        for j in range(slow):
            if np.isnan(ratio[i - j]):
                valid = False
                break
            window_sum += ratio[i - j]
        if valid:
            result[i] = window_sum
    
    return result


def pl_massi(
    high: IntoExpr,
    low: IntoExpr,
    fast: int = 9,
    slow: int = 25,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Mass Index (MASSI)

    Composition: pl_ema for first EMA, Numba for cascaded EMA + rolling sum.

    The Mass Index identifies trend reversals based on range expansions.

    Sources:
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        fast: The short period. Default: 9
        slow: The long period. Default: 25
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Mass Index expression
    """
    from polars_ti.overlap.ema import pl_ema
    
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    
    if high_expr is None or low_expr is None:
        return None
    
    # Swap if slow < fast (matches Pandas behavior)
    _fast = min(fast, slow)
    _slow = max(fast, slow)
    _offset = offset
    
    # High-Low range (non-zero protection like Pandas non_zero_range)
    hl_range = pl_non_zero_range(high_expr, low_expr)
    
    # First EMA using pl_ema composition
    ema1 = pl_ema(hl_range, length=_fast, talib=False, presma=True)
    
    def compute_massi(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        ema1_vals = df["_ema1"].to_numpy().astype(np.float64)
        
        result = nb_massi_from_ema1(ema1_vals, _fast, _slow)
        
        if _offset != 0:
            result = np.roll(result, _offset)
            if _offset > 0:
                result[:_offset] = np.nan
        
        return pl.Series(result)
    
    _props = f"_{_fast}_{_slow}"
    
    return pl.struct([
        ema1.alias("_ema1"),
    ]).map_batches(compute_massi, return_dtype=pl.Float64).alias(f"MASSI{_props}")


