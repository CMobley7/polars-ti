# -*- coding: utf-8 -*-
# =============================================================================
# Polars CCI (Commodity Channel Index) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_cci(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    c: float = 0.015,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Commodity Channel Index (CCI)

    Momentum oscillator for overbought/oversold levels.
    CCI = (TP - SMA(TP)) / (c * MAD(TP))
    where TP = (high + low + close) / 3

    Sources:
        https://www.investopedia.com/terms/c/commoditychannelindex.asp

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        c: Scaling constant (Lambert's constant). Default: 0.015
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CCI expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    
    _use_talib = Imports["talib"] and v_talib(talib)
    
    if _use_talib:
        _length = length
        
        def compute_cci_talib(struct: pl.Series) -> pl.Series:
            from talib import CCI as TALIB_CCI
            df = struct.struct.unnest()
            h = df["_high"].to_numpy().astype(np.float64)
            l = df["_low"].to_numpy().astype(np.float64)
            cl = df["_close"].to_numpy().astype(np.float64)
            result = TALIB_CCI(h, l, cl, timeperiod=_length)
            return pl.Series(result)
        
        cci_expr = pl.struct([
            high_expr.alias("_high"),
            low_expr.alias("_low"),
            close_expr.alias("_close")
        ]).map_batches(compute_cci_talib, return_dtype=pl.Float64)
    else:
        # Clean composition matching Pandas approach
        from polars_ti.overlap.hlc3 import pl_hlc3
        from polars_ti.overlap.sma import pl_sma
        from polars_ti.statistics.mad import pl_mad
        
        # Typical Price, SMA, and MAD
        tp = pl_hlc3(high_expr, low_expr, close_expr, talib=False, offset=0)
        tp_sma = pl_sma(tp, length=length, talib=False, offset=0)
        tp_mad = pl_mad(tp, length=length, offset=0)
        
        # CCI = (TP - SMA(TP)) / (c * MAD(TP))
        # Protect against divide-by-zero when MAD is near zero
        cci_expr = pl.when(tp_mad < 1e-8).then(0.0).otherwise(
            (tp - tp_sma) / (c * tp_mad)
        )
    
    if offset != 0:
        cci_expr = cci_expr.shift(offset)
    
    return cci_expr.alias(f"CCI_{length}_{c}")


