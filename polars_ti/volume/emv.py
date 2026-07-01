# -*- coding: utf-8 -*-
# =============================================================================
# Polars EMV (Ease of Movement — raw/Tulip variant) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._math import non_zero_range
from polars_ti.utils._validate import v_expr


def emv(
    high: IntoExpr,
    low: IntoExpr,
    volume: IntoExpr,
    divisor: float = 10000.0,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Ease of Movement (EMV) — raw/Tulip variant

    Raw (unsmoothed) Ease of Movement oscillator matching tulipy's EMV.
    Unlike ta.eom (which applies SMA smoothing and uses divisor=1e8),
    this is the unsmoothed version with divisor=10000.

    Formula:
        midpoint  = (High + Low) / 2
        distance  = midpoint - prev(midpoint)
        box_ratio = (Volume / divisor) / (High - Low)
        EMV       = distance / box_ratio

    Sources:
        Tulip Indicators: https://tulipindicators.org/emv

    Args:
        high: Column name or pl.Expr for 'high' prices.
        low: Column name or pl.Expr for 'low' prices.
        volume: Column name or pl.Expr for volume.
        divisor: Volume scaling divisor. Default: 10000
        drift: Difference period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: EMV expression.
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    volume_expr = v_expr(volume)
    if any(e is None for e in [high_expr, low_expr, volume_expr]):
        return None

    _divisor = divisor
    _drift = drift
    _offset = offset

    midpoint = (high_expr + low_expr) / pl.lit(2.0)
    distance = midpoint - midpoint.shift(_drift)
    hl_range_safe = non_zero_range(high_expr, low_expr)
    box_ratio = (volume_expr / pl.lit(_divisor)) / hl_range_safe

    result = distance / box_ratio

    if _offset != 0:
        result = result.shift(_offset)

    return result.alias("EMV")
