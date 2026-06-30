# -*- coding: utf-8 -*-
# =============================================================================
# Polars ALMA Implementation (Pure rolling_map)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
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
    Pure Polars implementation using rolling_map.

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

    # Pre-compute Gaussian weights
    x = np.arange(length, dtype=np.float64)
    k = np.floor(dist_offset * (length - 1))
    weights = np.exp(-0.5 * ((sigma / length) * (x - k)) ** 2)
    weights = weights / weights.sum()
    weights_list = weights.tolist()

    _length = length
    _weights = weights_list

    def gaussian_weighted_mean(s: pl.Series) -> float:
        vals = s.to_numpy()
        if len(vals) < _length:
            return float("nan")
        # Check for NaN in window
        if np.isnan(vals).any():
            return float("nan")
        return (vals * _weights).sum()

    alma_expr = close_expr.rolling_map(function=gaussian_weighted_mean, window_size=length, min_samples=length)

    # Apply offset
    if offset != 0:
        alma_expr = alma_expr.shift(offset)

    return alma_expr.alias(f"ALMA_{length}_{sigma}_{dist_offset}")
