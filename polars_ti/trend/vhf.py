# -*- coding: utf-8 -*-
# =============================================================================
# Polars VHF (Vertical Horizontal Filter) Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def vhf(
    close: IntoExpr,
    length: int = 28,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Vertical Horizontal Filter (VHF)

    Identifies trending vs ranging markets.

    Formula: VHF = |HCP - LCP| / sum(|diff(close, drift)|, length)

    Args:
        close: Column name or pl.Expr for input values
        length: Period. Default: 28
        drift: Difference period. Default: 1
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: VHF expression
    """
    close_expr = v_expr(close)

    hcp = close_expr.rolling_max(window_size=length)
    lcp = close_expr.rolling_min(window_size=length)
    diff_abs = close_expr.diff(drift).abs()
    diff_sum = diff_abs.rolling_sum(window_size=length)

    vhf_expr = (hcp - lcp).abs() / diff_sum

    if offset != 0:
        vhf_expr = vhf_expr.shift(offset)

    return vhf_expr.alias(f"VHF_{length}")
