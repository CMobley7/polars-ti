# -*- coding: utf-8 -*-
# =============================================================================
# Polars Decreasing Implementation (pure Polars)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def pl_decreasing(
    close: IntoExpr,
    length: int = 1,
    strict: bool = False,
    asint: bool = True,
    percent: float | None = None,
    drift: int = 1,
    offset: int = 0,
) -> PlExpr:
    """Polars: Decreasing

    Returns True/1 if the series is decreasing over a period, False/0 otherwise.
    If strict=True, checks if the series is continuously decreasing over the period.

    Args:
        close: Column name or pl.Expr for input values
        length: Period length. Default: 1
        strict: If True, check continuous decrease. Default: False
        asint: Returns as 1/0 instead of True/False. Default: True
        percent: Percent threshold. Default: None
        drift: Difference period (for strict mode). Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Decreasing expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    # Apply percent threshold if specified (opposite to increasing)
    if percent is not None and percent > 0:
        close_adj = (1 - 0.01 * percent) * close_expr
    else:
        close_adj = close_expr

    if strict:
        # Strict mode: check each step is decreasing
        result = close_expr < close_adj.shift(drift)
        for x in range(3, length + 1):
            result = result & (close_expr.shift(x - (drift + 1)) < close_adj.shift(x - drift))
        result = result.fill_null(False)
    else:
        # Non-strict: just check if diff over length is negative
        result = close_adj.diff(length) < 0

    if asint:
        result = result.cast(pl.Int64)

    if offset != 0:
        result = result.shift(offset)

    # Build name like Pandas
    _percent = f"_{0.01 * percent}" if percent else ""
    _props = f"{'S' if strict else ''}DEC{'p' if percent else ''}"
    return result.alias(f"{_props}_{length}{_percent}")
