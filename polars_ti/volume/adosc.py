# -*- coding: utf-8 -*-
# =============================================================================
# Polars ADOSC (Accumulation/Distribution Oscillator) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_adosc(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    fast: int = 3,
    slow: int = 10,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Accumulation/Distribution Oscillator (ADOSC) / Chaikin Oscillator

    Accumulation/Distribution Oscillator indicator utilizes
    Accumulation/Distribution and treats it similarly to MACD.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        fast: Fast EMA period. Default: 3
        slow: Slow EMA period. Default: 10
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ADOSC expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)
    
    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None
    
    _use_talib = Imports["talib"] and v_talib(talib)
    _fast = fast
    _slow = slow
    
    if _use_talib:
        def compute_adosc(df: pl.DataFrame) -> pl.Series:
            from talib import ADOSC as TALIB_ADOSC
            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            v = df["volume"].to_numpy().astype(np.float64)
            result = TALIB_ADOSC(h, l, c, v, fastperiod=_fast, slowperiod=_slow)
            return pl.Series(f"ADOSC_{_fast}_{_slow}", result)
        
        adosc_expr = pl.struct([
            high_expr.alias("high"),
            low_expr.alias("low"),
            close_expr.alias("close"),
            volume_expr.alias("volume")
        ]).map_batches(
            lambda s: compute_adosc(s.struct.unnest()),
            return_dtype=pl.Float64
        )
    else:
        # Compose using pl_ad and pl_ma for code reuse
        # Forward talib param for consistent behavior
        from polars_ti.volume.ad import pl_ad
        from polars_ti.ma import pl_ma
        
        # Build AD expression (without offset) - use talib if available for AD
        ad_expr = pl_ad(high_expr, low_expr, close_expr, volume_expr, talib=talib, offset=0)
        
        # Apply EMA to AD for fast and slow - forward talib for TA-Lib EMA behavior
        fast_ma = pl_ma(name="ema", source=ad_expr, length=fast, talib=talib)
        slow_ma = pl_ma(name="ema", source=ad_expr, length=slow, talib=talib)
        
        # ADOSC = FastEMA(AD) - SlowEMA(AD)
        adosc_expr = fast_ma - slow_ma
    
    if offset != 0:
        adosc_expr = adosc_expr.shift(offset)
    
    return adosc_expr.alias(f"ADOSC_{fast}_{slow}")

