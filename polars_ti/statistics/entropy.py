import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars ENTROPY Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.utils._rolling import rolling_sum
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_entropy(close: np.ndarray, length: int, base: float) -> np.ndarray:
    """Compute the two full-window entropy sums in linear time."""
    total = rolling_sum(close, length)
    term = np.full(len(close), np.nan)
    for i in range(length - 1, len(close)):
        if total[i] > 0:
            probability = close[i] / total[i]
            if probability > 0:
                term[i] = -probability * np.log(probability) / np.log(base)
    return rolling_sum(term, length)


def entropy(
    close: IntoExpr,
    length: int = 10,
    base: float = 2.0,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Entropy (ENTP)

    Shannon entropy measures the unpredictability of data.
    Uses Numba @njit kernel for high performance.

    Sources:
        https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 10
        base: Logarithmic base. Default: 2.0
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Entropy expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _base = base

    def compute_entropy(s: pl.Series) -> pl.Series:
        """Compute entropy using Numba kernel."""
        arr = s.to_numpy().astype(np.float64)
        result = nb_entropy(arr, _length, _base)
        return pl.Series(result)

    result = close_expr.map_batches(compute_entropy, return_dtype=pl.Float64)

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"ENTP_{length}")
