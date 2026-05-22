# -*- coding: utf-8 -*-
# =============================================================================
# Polars Chandelier Exit Implementation (Pure Composition)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def pl_chandelier_exit(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    high_length: int = 22,
    low_length: int = 22,
    atr_length: int = 14,
    multiplier: float = 2.0,
    use_close: bool = False,
    drift: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Chandelier Exit (CHDLREXT)

    Pure composition: pl_atr for ATR, native Polars for rolling max/min
    and direction with forward_fill.

    Sources:
        https://school.stockcharts.com/doku.php?id=technical_indicators:chandelier_exit

    Args:
        high: Column name or pl.Expr for 'high'
        low: Column name or pl.Expr for 'low'
        close: Column name or pl.Expr for 'close'
        high_length: Highest high length. Default: 22
        low_length: Lowest low length. Default: 22
        atr_length: ATR length. Default: 14
        multiplier: ATR multiplier. Default: 2.0
        use_close: Use close for rolling max/min. Default: False
        drift: Shift period for direction. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with long, short, direction columns
    """
    from polars_ti.volatility.atr import pl_atr

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    # ATR using pl_atr composition
    atr_expr = pl_atr(high_expr, low_expr, close_expr, length=atr_length, mamode="rma", talib=False)
    atr_mult = atr_expr * pl.lit(multiplier)

    # Rolling max/min for long/short
    if use_close:
        roll_length = max(high_length, low_length)
        long_expr = close_expr.rolling_max(window_size=roll_length, min_samples=1) - atr_mult
        short_expr = close_expr.rolling_min(window_size=roll_length, min_samples=1) + atr_mult
    else:
        long_expr = high_expr.rolling_max(window_size=high_length, min_samples=1) - atr_mult
        short_expr = low_expr.rolling_min(window_size=low_length, min_samples=1) + atr_mult

    # Direction: uptrend (1) if close > long.shift, downtrend (-1) if close < short.shift
    # Replace 0 with null, ffill, default to 1
    uptrend = (close_expr > long_expr.shift(drift)).cast(pl.Int64)
    downtrend = (close_expr < short_expr.shift(drift)).cast(pl.Int64) * -1
    raw_dir = uptrend + downtrend
    direction = pl.when(raw_dir == 0).then(None).otherwise(raw_dir).forward_fill().fill_null(1).cast(pl.Float64)

    # Apply offset
    if offset != 0:
        long_expr = long_expr.shift(offset)
        short_expr = short_expr.shift(offset)
        direction = direction.shift(offset)

    _props = f"_{high_length}_{low_length}_{atr_length}_{multiplier}"
    if use_close:
        _props = f"_CLOSE{_props}"

    return pl.struct(
        [
            long_expr.alias("long"),
            short_expr.alias("short"),
            direction.alias("direction"),
        ]
    ).alias(f"CHDLREXT{_props}")
