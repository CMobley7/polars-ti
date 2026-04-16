# -*- coding: utf-8 -*-
# =============================================================================
# Polars VWMACD Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_vwmacd(
    close: IntoExpr,
    volume: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    offset: int = 0,
) -> list[PlExpr]:
    """Polars: Volume Weighted MACD (VWMACD)

    Volume Weighted MACD is a variation of the traditional MACD that
    incorporates volume into the calculation using Volume Weighted Moving
    Averages (VWMA) instead of EMAs.

    Sources:
        https://www.tradingview.com/script/NUs1Y5V7-Volume-Weighted-MACD/

    Calculation:
        FastVWMA = VWMA(close, volume, fast)
        SlowVWMA = VWMA(close, volume, slow)
        VWMACD = FastVWMA - SlowVWMA
        Signal = VWMA(VWMACD, volume, signal)
        Histogram = VWMACD - Signal

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        fast: Fast VWMA period. Default: 12
        slow: Slow VWMA period. Default: 26
        signal: Signal VWMA period. Default: 9
        offset: Shift result by N periods. Default: 0

    Returns:
        list[pl.Expr]: [VWMACD, Histogram, Signal]
    """
    from polars_ti.volume.vwma import pl_vwma

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _props = f"_{fast}_{slow}_{signal}"

    # FastVWMA and SlowVWMA
    # Note: pl_vwma returns an aliased expression, we need the raw expr
    fast_vwma = pl_vwma(close_expr, volume_expr, length=fast, offset=0)
    slow_vwma = pl_vwma(close_expr, volume_expr, length=slow, offset=0)

    # VWMACD = FastVWMA - SlowVWMA
    vwmacd_expr = fast_vwma - slow_vwma

    # Signal = VWMA(VWMACD, volume, signal)
    # For signal calculation, we need the raw VWMACD values (without alias)
    # Since pl_vwma expects close and volume, and signal uses VWMACD instead of close,
    # we need to reconstruct the VWMA calculation inline
    from polars_ti.overlap.sma import pl_sma

    # VWMA formula: SMA(close * volume) / SMA(volume)
    # For signal: SMA(vwmacd * volume) / SMA(volume)
    vwmacd_pv = vwmacd_expr * volume_expr
    signal_expr = pl_sma(vwmacd_pv, length=signal) / pl_sma(volume_expr, length=signal)

    # Histogram = VWMACD - Signal
    histogram_expr = vwmacd_expr - signal_expr

    if offset != 0:
        vwmacd_expr = vwmacd_expr.shift(offset)
        signal_expr = signal_expr.shift(offset)
        histogram_expr = histogram_expr.shift(offset)

    return [
        vwmacd_expr.alias(f"VWMACD{_props}"),
        histogram_expr.alias(f"VWMACDh{_props}"),
        signal_expr.alias(f"VWMACDs{_props}"),
    ]
