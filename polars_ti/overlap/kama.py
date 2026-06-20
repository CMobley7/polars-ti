# -*- coding: utf-8 -*-
# =============================================================================
# Polars KAMA Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr
from polars_ti.maps import Imports


def kama(
    close: IntoExpr,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Kaufman's Adaptive Moving Average (KAMA)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Er period. Default: 10
        fast: Fast MA period. Default: 2
        slow: Slow MA period. Default: 30
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: KAMA expression
    """
    close_expr = v_expr(close)

    if Imports["talib"] and talib:

        def compute_kama(s: pl.Series) -> pl.Series:
            from talib import KAMA

            arr = s.to_numpy().astype(np.float64)
            result = KAMA(arr, timeperiod=length)
            if offset != 0:
                result = np.roll(result, offset)
                if offset > 0:
                    result[:offset] = np.nan
            return pl.Series(result)

        return close_expr.map_batches(compute_kama, return_dtype=pl.Float64).alias(f"KAMA_{length}_{fast}_{slow}")
    else:

        @njit(cache=True)
        def nb_kama(close_arr: np.ndarray, length: int, fast: int, slow: int) -> np.ndarray:
            """Numba-optimized KAMA calculation."""
            m = len(close_arr)
            result = np.empty(m, dtype=np.float64)
            result[: length - 1] = np.nan

            fr = 2.0 / (fast + 1)
            sr = 2.0 / (slow + 1)

            # Initial value - SMA of first `length` values
            result[length - 1] = np.mean(close_arr[:length])

            for i in range(length, m):
                # Change in price
                change = abs(close_arr[i] - close_arr[i - length])

                # Volatility (sum of absolute differences)
                volatility = 0.0
                for j in range(i - length + 1, i + 1):
                    volatility += abs(close_arr[j] - close_arr[j - 1])

                # Efficiency Ratio
                if volatility > 1e-10:
                    er = change / volatility
                else:
                    er = 0.0

                # Smoothing Constant
                sc = (er * (fr - sr) + sr) ** 2

                # KAMA value
                result[i] = sc * close_arr[i] + (1 - sc) * result[i - 1]

            return result

        def compute_kama(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            result = nb_kama(arr, length, fast, slow)
            if offset != 0:
                result = np.roll(result, offset)
                if offset > 0:
                    result[:offset] = np.nan
            return pl.Series(result)

        return close_expr.map_batches(compute_kama, return_dtype=pl.Float64).alias(f"KAMA_{length}_{fast}_{slow}")
