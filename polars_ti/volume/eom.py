# -*- coding: utf-8 -*-
# =============================================================================
# Polars EOM (Ease of Movement) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import pl_non_zero_range
from polars_ti.utils._validate import v_expr


def pl_eom(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 14,
    divisor: float = 100_000_000,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Ease of Movement (EOM)

    Ease of Movement is a volume based oscillator that measures the relationship
    between price and volume fluctuating across a zero line.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: SMA smoothing period. Default: 14
        divisor: Volume divisor. Default: 100000000
        drift: The diff period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: EOM expression
    """
    from polars_ti.overlap.hl2 import pl_hl2
    from polars_ti.overlap.sma import pl_sma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None

    # EOM = SMA(distance / box_ratio, length)
    # distance = hl2 - hl2.shift(drift)
    # box_ratio = (volume / divisor) / (high - low)

    # Use pl_hl2 for code reuse
    hl2_expr = pl_hl2(high_expr, low_expr)
    hl2_shifted = pl_hl2(high_expr.shift(drift), low_expr.shift(drift))
    distance = hl2_expr - hl2_shifted

    hl_range_safe = pl_non_zero_range(high_expr, low_expr)
    box_ratio = (volume_expr / divisor) / hl_range_safe

    eom_raw = distance / box_ratio

    # Use pl_sma for code reuse
    eom_expr = pl_sma(eom_raw, length=length, talib=False, offset=0)

    if offset != 0:
        eom_expr = eom_expr.shift(offset)

    return eom_expr.alias(f"EOM_{length}_{int(divisor)}")
