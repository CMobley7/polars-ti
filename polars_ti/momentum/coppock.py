# -*- coding: utf-8 -*-
# =============================================================================
# Polars Coppock Curve Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def coppock(
    close: IntoExpr,
    length: int = 10,
    fast: int = 11,
    slow: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Coppock Curve (COPC)

    Momentum indicator designed for use on a monthly time scale.
    Formula: WMA(ROC(fast) + ROC(slow), length)

    Sources:
        https://en.wikipedia.org/wiki/Coppock_curve

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: WMA period. Default: 10
        fast: Fast ROC period. Default: 11
        slow: Slow ROC period. Default: 14
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Coppock Curve expression
    """
    from polars_ti.momentum.roc import roc
    from polars_ti.overlap.wma import wma
    from polars_ti.utils import v_pos_int

    close_expr = v_expr(close)

    length = v_pos_int(length, "length")
    fast = v_pos_int(fast, "fast")
    slow = v_pos_int(slow, "slow")

    # Total ROC = ROC(fast) + ROC(slow)
    roc_fast = roc(close_expr, length=fast, scalar=100.0, talib=False, offset=0)
    roc_slow = roc(close_expr, length=slow, scalar=100.0, talib=False, offset=0)
    total_roc = roc_fast + roc_slow

    # Coppock = WMA(total_roc, length)
    coppock_expr = wma(total_roc, length=length, offset=0)

    if offset != 0:
        coppock_expr = coppock_expr.shift(offset)

    return coppock_expr.alias(f"COPC_{fast}_{slow}_{length}")
