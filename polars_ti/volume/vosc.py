# -*- coding: utf-8 -*-
# =============================================================================
# Polars VOSC (Volume Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def vosc(
    volume: IntoExpr,
    fast: int = 14,
    slow: int = 28,
    offset: int = 0,
) -> PlExpr:
    """Polars: Volume Oscillator (VOSC)

    Measures the difference between two volume SMAs (fast and slow) as a
    percentage of the slow SMA.

    Formula:
        VOSC = 100 * (SMA(volume, fast) - SMA(volume, slow)) / SMA(volume, slow)

    Sources:
        https://school.stockcharts.com/doku.php?id=technical_indicators:volume_oscillator_vo

    Args:
        volume: Column name or pl.Expr for volume.
        fast: Fast SMA period. Default: 14
        slow: Slow SMA period. Default: 28
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: VOSC expression.
    """
    from polars_ti.overlap.sma import sma

    volume_expr = v_expr(volume)
    if volume_expr is None:
        return None

    _fast, _slow = fast, slow
    if _fast > _slow:
        _fast, _slow = _slow, _fast

    _offset = offset

    fast_sma = sma(volume_expr, length=_fast, talib=False)
    slow_sma = sma(volume_expr, length=_slow, talib=False)

    result = pl.lit(100.0) * (fast_sma - slow_sma) / slow_sma

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias(f"VOSC_{_fast}_{_slow}")
