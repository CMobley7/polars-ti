# -*- coding: utf-8 -*-
# =============================================================================
# Polars SUPERTREND Implementation (pl_hl2 + pl_atr composition)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.hl2 import pl_hl2
from polars_ti.volatility.atr import pl_atr


@njit(cache=True)
def nb_supertrend_bands(close: np.ndarray, lb: np.ndarray, ub: np.ndarray, 
                         length: int) -> tuple:
    """Numba kernel for recursive Supertrend band preservation logic."""
    m = len(close)
    
    dir_ = np.ones(m, dtype=np.int64)
    trend = np.zeros(m, dtype=np.float64)
    long = np.empty(m, dtype=np.float64)
    short = np.empty(m, dtype=np.float64)
    long[:] = np.nan
    short[:] = np.nan
    
    # Make copies so we can modify in-place
    lb = lb.copy()
    ub = ub.copy()
    
    for i in range(1, m):
        if close[i] > ub[i - 1]:
            dir_[i] = 1
        elif close[i] < lb[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
        
        # Band preservation logic (TradingView style)
        if dir_[i] > 0 and lb[i] < lb[i - 1]:
            lb[i] = lb[i - 1]
        if dir_[i] < 0 and ub[i] > ub[i - 1]:
            ub[i] = ub[i - 1]
        
        if dir_[i] > 0:
            trend[i] = lb[i]
            long[i] = lb[i]
        else:
            trend[i] = ub[i]
            short[i] = ub[i]
    
    trend[0] = np.nan
    return trend, dir_, long, short


def pl_supertrend(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 7,
    multiplier: float = 3.0,
    mamode: str = "rma",
    offset: int = 0,
) -> pl.Expr:
    """Polars: Supertrend - uses pl_hl2 and pl_atr composition.
    
    Returns struct with SUPERT, SUPERTd, SUPERTl, SUPERTs.
    
    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        length: ATR period. Default: 7
        multiplier: Band distance multiplier. Default: 3.0
        mamode: MA type for ATR ('rma', 'sma', 'ema'). Default: 'rma'
        offset: Shift result by N periods. Default: 0
    
    Returns:
        pl.Expr: Struct with SUPERT, SUPERTd, SUPERTl, SUPERTs
    """
    _props = f"_{length}_{multiplier}"
    _length = length
    _multiplier = multiplier
    _offset = offset
    
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    
    # Use pl_hl2 and pl_atr composition!
    hl2_expr = pl_hl2(high_expr, low_expr)
    atr_expr = pl_atr(high_expr, low_expr, close_expr, length=length, mamode=mamode, talib=True)
    
    def compute_supertrend(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        close_arr = df["_close"].to_numpy().astype(np.float64)
        hl2_arr = df["_hl2"].to_numpy().astype(np.float64)
        atr_arr = df["_atr"].to_numpy().astype(np.float64)
        
        # Compute basic bands
        lb = hl2_arr - _multiplier * atr_arr
        ub = hl2_arr + _multiplier * atr_arr
        
        # Numba for recursive band logic only
        trend, dir_, long, short = nb_supertrend_bands(close_arr, lb, ub, _length)
        dir_f = dir_.astype(np.float64)
        dir_f[:_length] = np.nan
        
        if _offset != 0:
            for arr in [trend, dir_f, long, short]:
                arr[:] = np.roll(arr, _offset)
                if _offset > 0:
                    arr[:_offset] = np.nan
        
        n = len(close_arr)
        return pl.Series([{
            f"SUPERT{_props}": trend[i], f"SUPERTd{_props}": dir_f[i],
            f"SUPERTl{_props}": long[i], f"SUPERTs{_props}": short[i]
        } for i in range(n)])
    
    fields = [
        pl.Field(f"SUPERT{_props}", pl.Float64), 
        pl.Field(f"SUPERTd{_props}", pl.Float64),
        pl.Field(f"SUPERTl{_props}", pl.Float64), 
        pl.Field(f"SUPERTs{_props}", pl.Float64)
    ]
    
    return pl.struct([
        close_expr.alias("_close"),
        hl2_expr.alias("_hl2"),
        atr_expr.alias("_atr"),
    ]).map_batches(compute_supertrend, return_dtype=pl.Struct(fields)).alias(f"SUPERT{_props}")


