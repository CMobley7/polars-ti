# -*- coding: utf-8 -*-
# =============================================================================
# Polars KVO (Klinger Volume Oscillator) Implementation
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def kvo(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    fast: int = 34,
    slow: int = 55,
    signal: int = 13,
    mamode: str = "ema",
    talib: bool = True,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Klinger Volume Oscillator (KVO)

    This indicator attempts to predict price reversals by comparing volume to price.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        fast: Fast MA period. Default: 34
        slow: Slow MA period. Default: 55
        signal: Signal MA period. Default: 13
        mamode: MA type. Default: 'ema'
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of expressions [KVO, KVO_signal]
    """
    from polars_ti.overlap.hlc3 import hlc3
    from polars_ti.ma import ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None

    _props = f"_{fast}_{slow}_{signal}"

    # Use pl_hlc3 for code reuse
    hlc3_expr = hlc3(high_expr, low_expr, close_expr)

    # signed_volume = volume * sign(hlc3.diff())
    # Note: diff() on first row is null, which propagates to signed_volume (matches Pandas NaN)
    hlc3_diff = hlc3_expr.diff()
    sign = pl.when(hlc3_diff > 0).then(1.0).when(hlc3_diff < 0).then(-1.0).otherwise(pl.lit(None))
    signed_volume = volume_expr * sign

    # KVO = MA(signed_volume, fast) - MA(signed_volume, slow)
    kvo_fast = ma(name=mamode, source=signed_volume, length=fast, talib=talib)
    kvo_slow = ma(name=mamode, source=signed_volume, length=slow, talib=talib)
    kvo_expr = kvo_fast - kvo_slow

    # Signal = MA(KVO, signal)
    kvo_signal_expr = ma(name=mamode, source=kvo_expr, length=signal, talib=talib)

    if offset != 0:
        kvo_expr = kvo_expr.shift(offset)
        kvo_signal_expr = kvo_signal_expr.shift(offset)

    return [
        kvo_expr.alias(f"KVO{_props}"),
        kvo_signal_expr.alias(f"KVOs{_props}"),
    ]
