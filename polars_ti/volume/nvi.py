# -*- coding: utf-8 -*-
# =============================================================================
# Polars NVI (Negative Volume Index) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def nvi(
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 1,
    initial: float = 1000.0,
    offset: int = 0,
) -> PlExpr:
    """Polars: Negative Volume Index (NVI)

    The Negative Volume Index is a cumulative indicator that uses volume
    change in an attempt to identify where smart money is active.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: ROC period. Default: 1
        initial: Initial NVI value. Default: 1000
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: NVI expression
    """
    from polars_ti.momentum.roc import roc

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    # Pure Polars implementation - no Numba needed!
    # ROC calculated using pl_roc for code reuse
    roc_expr = roc(close_expr, length=length, scalar=100.0, talib=False, offset=0)

    # NVI: When volume decreases, add ROC; otherwise add 0
    # Then cumsum + initial
    vol_decreased = volume_expr.diff() < 0
    nvi_change = pl.when(vol_decreased).then(roc_expr).otherwise(0.0).fill_null(0.0)
    nvi_expr = nvi_change.cum_sum() + initial

    if offset != 0:
        nvi_expr = nvi_expr.shift(offset)

    return nvi_expr.alias(f"NVI_{length}")
