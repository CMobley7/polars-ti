# -*- coding: utf-8 -*-
# =============================================================================
# Polars Vortex Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_vortex(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    length: int = 14,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Vortex Indicator

    Two oscillators capturing positive and negative trend movement.

    Formula:
        VMP = abs(high - low.shift(1))
        VMM = abs(low - high.shift(1))
        VIP = sum(VMP, length) / sum(TR, length)
        VIM = sum(VMM, length) / sum(TR, length)

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        length: Period. Default: 14
        drift: Difference period. Default: 1
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with VTXP and VTXM columns
    """
    from polars_ti.volatility.true_range import pl_true_range

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    tr = pl_true_range(high_expr, low_expr, close_expr)
    tr_sum = tr.rolling_sum(window_size=length)

    vmp = (high_expr - low_expr.shift(drift)).abs()
    vmm = (low_expr - high_expr.shift(drift)).abs()

    vip = vmp.rolling_sum(window_size=length) / tr_sum
    vim = vmm.rolling_sum(window_size=length) / tr_sum

    if offset != 0:
        vip = vip.shift(offset)
        vim = vim.shift(offset)

    return pl.struct(
        vip.alias(f"VTXP_{length}"),
        vim.alias(f"VTXM_{length}"),
    ).alias(f"VTX_{length}")
