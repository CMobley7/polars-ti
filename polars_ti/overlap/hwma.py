# -*- coding: utf-8 -*-
# =============================================================================
# Polars HWMA Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._prefix import run_after_prefix
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _hwma_numba(close: np.ndarray, na: float, nb: float, nc: float) -> np.ndarray:
    """Numba-optimized HWMA calculation."""
    m = len(close)
    result = np.empty(m, dtype=np.float64)

    last_a = 0.0
    last_v = 0.0
    last_f = close[0]

    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result[i] = F + V + 0.5 * A
        last_a, last_f, last_v = A, F, V

    return result


def hwma(
    close: IntoExpr,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
) -> PlExpr:
    """Polars: HWMA (Holt-Winter Moving Average)

    Args:
        close: Column name or pl.Expr for 'close' prices
        na: Smoothed series parameter (0 to 1). Default: 0.2
        nb: Trend parameter (0 to 1). Default: 0.1
        nc: Seasonality parameter (0 to 1). Default: 0.1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: HWMA expression
    """
    close_expr = v_expr(close)

    def compute_hwma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        result = run_after_prefix((arr,), lambda arrays: (_hwma_numba(arrays[0], na, nb, nc),))[0]
        if offset != 0:
            result = np.roll(result, offset)
            if offset > 0:
                result[:offset] = np.nan
            else:
                result[offset:] = np.nan
        return pl.Series(result)

    return close_expr.map_batches(compute_hwma, return_dtype=pl.Float64).alias(f"HWMA_{na}_{nb}_{nc}")
