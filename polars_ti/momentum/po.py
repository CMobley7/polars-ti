# -*- coding: utf-8 -*-
# =============================================================================
# Polars PO (Projection Oscillator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.overlap.linreg import pl_linreg


def pl_po(
    close: IntoExpr,
    length: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Projection Oscillator (PO)

    The Projection Oscillator measures the percentage deviation of price from
    its linear regression trend line. It helps identify overbought and
    oversold conditions relative to the trend.

    Sources:
        https://www.tradingview.com/script/CDdh2vTz-Projection-Oscillator/
        Technical Analysis of Stock Trends by Edwards & Magee

    Calculation:
        LR = Linear Regression(close, length)
        PO = 100 * (close - LR) / LR

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 14
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: PO expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Calculate linear regression using pl_linreg
    lr = pl_linreg(close, length=length, talib=True)
    
    # PO = 100 * (close - LR) / LR with division protection
    # When LR is 0, result should be NaN
    po = pl.when(lr != 0).then(100.0 * (close_expr - lr) / lr).otherwise(None)
    
    # Apply offset
    if offset != 0:
        po = po.shift(offset)
    
    return po.alias(f"PO_{length}")
