# -*- coding: utf-8 -*-
# =============================================================================
# Polars SMMA Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.ma import pl_ma


@njit(cache=True)
def _smma_numba(close: np.ndarray, length: int, initial: float) -> np.ndarray:
    """Numba-optimized SMMA calculation."""
    m = len(close)
    result = np.empty(m, dtype=np.float64)
    result[: length - 1] = np.nan
    result[length - 1] = initial

    for i in range(length, m):
        result[i] = ((length - 1) * result[i - 1] + close[i]) / length

    return result


def pl_smma(
    close: IntoExpr,
    length: int = 7,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: SMoothed Moving Average (SMMA)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 7
        mamode: MA type for initial value. Default: "sma"
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: SMMA expression
    """
    close_expr = v_expr(close)

    def compute_smma(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        # Get initial value from MA
        temp_df = pl.DataFrame({"c": arr[:length]})
        initial = temp_df.select(pl_ma(mamode, "c", length=length, talib=talib)).item(length - 1, 0)

        result = _smma_numba(arr, length, initial)
        if offset != 0:
            result = np.roll(result, offset)
            if offset > 0:
                result[:offset] = np.nan
        return pl.Series(result)

    return close_expr.map_batches(compute_smma, return_dtype=pl.Float64).alias(f"SMMA_{length}")
