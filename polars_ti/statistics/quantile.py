import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars QUANTILE Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils import v_pos_int
from polars_ti.utils._rolling import rolling_quantile
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_quantile(close: np.ndarray, length: int, q: float) -> np.ndarray:
    """Return linearly interpolated quantiles; any NaN invalidates its window."""
    return rolling_quantile(close, length, q)


def quantile(
    close: IntoExpr,
    length: int = 30,
    q: float = 0.5,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Rolling Quantile

    Calculates the Quantile over a rolling period.
    Uses Numba @njit kernel for high performance.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 30
        q: Quantile value (0 to 1). Default: 0.5
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Quantile expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _q = float(q) if isinstance(q, (int, float)) and 0 < q < 1 else 0.5
    length = v_pos_int(length, "length")
    _length = length

    def compute_quantile(s: pl.Series) -> pl.Series:
        """Compute quantile using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_quantile(arr, _length, _q)
        return pl.Series(result)

    result = close_expr.map_batches(compute_quantile, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"QTL_{length}_{_q}")
