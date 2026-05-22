# -*- coding: utf-8 -*-
# =============================================================================
# Polars CTI (Correlation Trend Indicator) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_cti(
    close: IntoExpr,
    length: int = 12,
    offset: int = 0,
) -> PlExpr:
    """Polars: Correlation Trend Indicator (CTI)

    Oscillator by John Ehler. Measures how closely prices follow a trend line.
    Values range from -1 to 1 (+1 = perfect uptrend, -1 = perfect downtrend).
    This is a wrapper for linreg(close, r=True).

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 12
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: CTI expression
    """
    from polars_ti.overlap.linreg import pl_linreg

    # CTI = linreg correlation coefficient (r=True)
    cti_expr = pl_linreg(close, length=length, r=True, offset=offset)

    return cti_expr.alias(f"CTI_{length}")
