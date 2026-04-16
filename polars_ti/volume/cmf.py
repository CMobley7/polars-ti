# -*- coding: utf-8 -*-
# =============================================================================
# Polars CMF (Chaikin Money Flow) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import pl_non_zero_range
from polars_ti.utils._validate import v_expr


def pl_cmf(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 20,
    offset: int = 0,
) -> PlExpr:
    """Polars: Chaikin Money Flow (CMF)

    Chaikin Money Flow measures the amount of money flow volume over a
    specific period in conjunction with Accumulation/Distribution.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: Rolling window period. Default: 20
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: CMF expression
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)
    
    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None
    
    # CMF = rolling_sum(CLV * volume) / rolling_sum(volume)
    # Note: Unlike AD which uses cumsum, CMF uses rolling_sum - cannot reuse pl_ad
    # CLV (Close Location Value) = (2*close - high - low) / (high - low)
    hl_range_safe = pl_non_zero_range(high_expr, low_expr)
    clv = (2 * close_expr - high_expr - low_expr) / hl_range_safe
    ad_component = clv * volume_expr
    
    cmf_expr = (
        ad_component.rolling_sum(window_size=length, min_samples=length)
        / volume_expr.rolling_sum(window_size=length, min_samples=length)
    )
    
    if offset != 0:
        cmf_expr = cmf_expr.shift(offset)
    
    return cmf_expr.alias(f"CMF_{length}")

