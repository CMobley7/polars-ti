# -*- coding: utf-8 -*-
# =============================================================================
# Polars RSI (Relative Strength Index) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_rsi(
    close: IntoExpr,
    length: int = 14,
    scalar: float = 100.0,
    mamode: str = "rma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Relative Strength Index (RSI)

    Popular momentum oscillator measuring velocity and magnitude
    of directional price movements.

    RSI = scalar * avg_gain / (avg_gain + |avg_loss|)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        scalar: Magnification (typically 100). Default: 100
        mamode: MA type for smoothing. Default: 'rma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: RSI expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    from polars_ti.ma import pl_ma
    
    close_expr = v_expr(close)
    
    if close_expr is None:
        return None
    
    _use_talib = Imports["talib"] and v_talib(talib)
    _length = length
    _scalar = scalar
    
    if _use_talib:
        def compute_rsi(s: pl.Series) -> pl.Series:
            from talib import RSI as TALIB_RSI
            arr = s.to_numpy().astype(np.float64)
            result = TALIB_RSI(arr, timeperiod=_length)
            return pl.Series(f"RSI_{_length}", result)
        
        rsi_expr = close_expr.map_batches(compute_rsi, return_dtype=pl.Float64)
    else:
        # Price change
        diff = close_expr.diff(1)
        
        # Separate gains and losses - MATCH PANDAS EXACTLY
        # Pandas: positive[positive < 0] = 0; negative[negative > 0] = 0
        # Keep losses as NEGATIVE values (matches Pandas), apply .abs() at end
        gains = pl.when(diff > 0).then(diff).otherwise(0.0)
        losses = pl.when(diff < 0).then(diff).otherwise(0.0)  # Keep negative
        
        # Apply MA for smoothing (default RMA/Wilder's)
        # Note: presma=False matches Pandas ewm(alpha=1/n, adjust=False) behavior
        avg_gain = pl_ma(name=mamode, source=gains, length=length, talib=False, presma=False)
        avg_loss = pl_ma(name=mamode, source=losses, length=length, talib=False, presma=False)
        
        # RSI calculation: scalar * avg_gain / (avg_gain + |avg_loss|)
        # Note: avg_loss is negative, so .abs() is required to match Pandas
        rsi_expr = _scalar * avg_gain / (avg_gain + avg_loss.abs())
    
    if offset != 0:
        rsi_expr = rsi_expr.shift(offset)
    
    return rsi_expr.alias(f"RSI_{length}")


