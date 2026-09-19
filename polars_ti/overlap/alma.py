import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars ALMA Implementation (Shared FIR kernel)
# =============================================================================
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._fir import fir_series
from polars_ti.utils._validate import v_expr


def alma(
    close: IntoExpr,
    length: int = 9,
    sigma: float = 6.0,
    dist_offset: float = 0.85,
    offset: int = 0,
) -> PlExpr:
    """Polars: Arnaud Legoux Moving Average (ALMA)

    Uses Gaussian distribution weighting for smoothing.
    Uses the shared FIR kernel with an accuracy-checked convolution path.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 9
        sigma: Smoothing value. Default: 6.0
        dist_offset: Distribution offset (0=smooth, 1=responsive). Default: 0.85
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: ALMA expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    def compute(series: pl.Series) -> pl.Series:
        """Apply the filter once per batch, preserving missing-window semantics."""
        # Reject undersized batches before allocating length-sized weights;
        # an oversized requested window must not allocate proportional memory.
        if len(series) < length:
            return pl.Series([None] * len(series), dtype=pl.Float64)
        x = np.arange(length, dtype=np.float64)
        k = np.floor(dist_offset * (length - 1))
        weights = np.exp(-0.5 * ((sigma / length) * (x - k)) ** 2)
        weights /= weights.sum()
        return fir_series(series, weights)

    alma_expr = close_expr.map_batches(compute, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        alma_expr = alma_expr.shift(offset)

    return alma_expr.alias(f"ALMA_{length}_{sigma}_{dist_offset}")
