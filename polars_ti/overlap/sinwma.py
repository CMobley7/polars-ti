# -*- coding: utf-8 -*-
# =============================================================================
# Polars SINWMA Implementation
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._fir import fir_series
from polars_ti.utils._validate import v_expr


def sinwma(
    close: IntoExpr,
    length: int = 14,
    offset: int = 0,
) -> PlExpr:
    """Polars: Sine Weighted Moving Average (SINWMA)

    A weighted average using sine cycles. The middle term(s) of the average
    have the highest weight(s).

    Source:
        https://www.tradingview.com/script/6MWFvnPO-Sine-Weighted-Moving-Average/

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 14
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SINWMA expression for lazy evaluation
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    import numpy as np

    def compute(series: pl.Series) -> pl.Series:
        """Apply the filter once per batch, preserving missing-window semantics."""
        if len(series) < length:
            return pl.Series([None] * len(series), dtype=pl.Float64)
        weights = np.array([np.sin((i + 1) * np.pi / (length + 1)) for i in range(length)])
        weights /= weights.sum()
        return fir_series(series, weights)

    sinwma_expr = close_expr.map_batches(compute, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        sinwma_expr = sinwma_expr.shift(offset)

    return sinwma_expr.alias(f"SINWMA_{length}")
