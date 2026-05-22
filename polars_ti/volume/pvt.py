# -*- coding: utf-8 -*-
# =============================================================================
# Polars PVT (Price-Volume Trend) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_pvt(
    close: IntoExpr,
    volume: IntoExpr,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Price-Volume Trend (PVT)

    Uses Rate of Change with volume and cumulative sum for money flow.

    Args:
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        drift: ROC period. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: PVT expression
    """
    from polars_ti.momentum.roc import pl_roc

    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if close_expr is None or volume_expr is None:
        return None

    # PVT = cumsum(ROC * volume)
    # Use pl_roc for code reuse
    roc_expr = pl_roc(close_expr, length=drift, scalar=100.0, talib=False, offset=0)

    pv = roc_expr * volume_expr
    pvt_expr = pv.cum_sum()

    if offset != 0:
        pvt_expr = pvt_expr.shift(offset)

    return pvt_expr.alias("PVT")
