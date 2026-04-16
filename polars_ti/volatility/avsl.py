# -*- coding: utf-8 -*-
# =============================================================================
# Polars AVSL Implementation (Composition: pl_vwma + pl_sma)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_avsl_core_logic(
    low_arr: np.ndarray,
    vpc_arr: np.ndarray,
    vpr_arr: np.ndarray,
    vm_arr: np.ndarray,
    scalar: float,
    slow: int
) -> np.ndarray:
    """Core AVSL calculation - only the non-expressible logic.
    
    Handles: VPC adjustment, division protections, inf handling, final smoothing.
    VWMA and SMA are computed externally via composition.
    """
    n = len(low_arr)
    
    # VPCI and deviation
    vpci = vpc_arr * vpr_arr * vm_arr
    deviation = scalar * vpci * vm_arr
    
    # Adjust VPC (clamp small values away from zero)
    vpc_adjusted = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = vpc_arr[i]
        if v > -1 and v < 0:
            vpc_adjusted[i] = -1
        elif v >= 0 and v < 1:
            vpc_adjusted[i] = 1
        else:
            vpc_adjusted[i] = v
    
    # Price function with division protection
    adjusted_price = np.empty(n, dtype=np.float64)
    for i in range(n):
        vpc_vpr = vpc_adjusted[i] * vpr_arr[i]
        if vpc_vpr == 0 or np.isnan(vpc_vpr):
            adjusted_price[i] = np.nan
        else:
            val = low_arr[i] / vpc_vpr
            if np.isinf(val):
                adjusted_price[i] = np.nan
            else:
                adjusted_price[i] = val
    
    # Rolling mean of adjusted_price / 100
    price_function = np.empty(n, dtype=np.float64)
    price_function[:slow - 1] = np.nan
    for i in range(slow - 1, n):
        win_sum = 0.0
        for j in range(slow):
            win_sum += adjusted_price[i - j]
        price_function[i] = (win_sum / slow) / 100
    
    # Final AVSL: rolling mean of (low - price_function + deviation)
    raw_avsl = low_arr - price_function + deviation
    
    avsl = np.empty(n, dtype=np.float64)
    avsl[:slow - 1] = np.nan
    for i in range(slow - 1, n):
        win_sum = 0.0
        for j in range(slow):
            win_sum += raw_avsl[i - j]
        avsl[i] = win_sum / slow
    
    return avsl


def pl_avsl(
    close: IntoExpr,
    low: IntoExpr,
    volume: IntoExpr,
    fast_period: int = 12,
    slow_period: int = 26,
    scalar: float = 2.0,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Anti-Volume Stop Loss (AVSL)

    Uses composition: pl_vwma + pl_sma for MAs, then Numba kernel
    for the remaining VPC adjustment and division protection logic.

    Sources:
        https://www.tradingview.com/script/lYWz5r9e-AVSL-Anti-Volume-Stop-Loss/
        Dormeier, Buff. "Investing with Volume Analysis"

    Args:
        close: Column name or pl.Expr for 'close'
        low: Column name or pl.Expr for 'low'
        volume: Column name or pl.Expr for 'volume'
        fast_period: Short period. Default: 12
        slow_period: Long period. Default: 26
        scalar: Band multiplier. Default: 2.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: AVSL stop-loss expression
    """
    from polars_ti.volume.vwma import pl_vwma
    from polars_ti.overlap.sma import pl_sma
    
    close_expr = v_expr(close)
    low_expr = v_expr(low)
    volume_expr = v_expr(volume)
    
    if close_expr is None or low_expr is None or volume_expr is None:
        return None
    
    _fast = fast_period
    _slow = slow_period
    _scalar = scalar
    _offset = offset
    
    # Use composition for VWMA and SMA (just like Pandas!)
    vwma_fast = pl_vwma(close_expr, volume_expr, length=fast_period)
    vwma_slow = pl_vwma(close_expr, volume_expr, length=slow_period)
    sma_fast = pl_sma(close_expr, length=fast_period)
    sma_slow = pl_sma(close_expr, length=slow_period)
    
    # VPC = vwma_slow - sma_slow
    vpc = vwma_slow - sma_slow
    
    # VPR = vwma_fast / sma_fast
    vpr = vwma_fast / sma_fast
    
    # VM = avg_vol_fast / avg_vol_slow
    vol_fast = pl_sma(volume_expr, length=fast_period)
    vol_slow = pl_sma(volume_expr, length=slow_period)
    vm = vol_fast / vol_slow
    
    def compute_avsl_final(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        low_arr = df["_low"].to_numpy().astype(np.float64)
        vpc_arr = df["_vpc"].to_numpy().astype(np.float64)
        vpr_arr = df["_vpr"].to_numpy().astype(np.float64)
        vm_arr = df["_vm"].to_numpy().astype(np.float64)
        
        result = nb_avsl_core_logic(low_arr, vpc_arr, vpr_arr, vm_arr, _scalar, _slow)
        
        if _offset != 0:
            result = np.roll(result, _offset)
            if _offset > 0:
                result[:_offset] = np.nan
        
        return pl.Series(result)
    
    return pl.struct([
        low_expr.alias("_low"),
        vpc.alias("_vpc"),
        vpr.alias("_vpr"),
        vm.alias("_vm"),
    ]).map_batches(
        compute_avsl_final, return_dtype=pl.Float64
    ).alias(f"AVSL_{fast_period}_{slow_period}")


