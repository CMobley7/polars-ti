# -*- coding: utf-8 -*-
# =============================================================================
# Polars PVO (Percentage Volume Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_pvo(
    volume: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Percentage Volume Oscillator (PVO)

    Percentage Volume Oscillator is a Momentum Oscillator for Volume.

    Args:
        volume: Column name or pl.Expr for 'volume'
        fast: Fast EMA period. Default: 12
        slow: Slow EMA period. Default: 26
        signal: Signal EMA period. Default: 9
        scalar: Scalar to multiply result. Default: 100
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: List of expressions [PVO, histogram, signal]
    """
    from polars_ti.overlap.ema import pl_ema

    volume_expr = v_expr(volume)

    if volume_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _props = f"_{fast}_{slow}_{signal}"

    # PVO = scalar * (fastEMA - slowEMA) / slowEMA
    fast_ema = pl_ema(volume_expr, length=fast)
    slow_ema = pl_ema(volume_expr, length=slow)
    pvo_expr = scalar * (fast_ema - slow_ema) / slow_ema

    # Signal = EMA(PVO, signal)
    signal_ema = pl_ema(pvo_expr, length=signal)

    # Histogram = PVO - Signal
    histogram_expr = pvo_expr - signal_ema

    if offset != 0:
        pvo_expr = pvo_expr.shift(offset)
        histogram_expr = histogram_expr.shift(offset)
        signal_ema = signal_ema.shift(offset)

    return [
        pvo_expr.alias(f"PVO{_props}"),
        histogram_expr.alias(f"PVOh{_props}"),
        signal_ema.alias(f"PVOs{_props}"),
    ]
