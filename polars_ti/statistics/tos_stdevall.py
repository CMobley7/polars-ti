# -*- coding: utf-8 -*-
# =============================================================================
# Polars TOS_STDEVALL Implementation (Numba @njit kernel)
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_linreg(arr: np.ndarray) -> tuple:
    """Numba-optimized simple linear regression.

    Returns (slope, intercept) for y = slope * x + intercept
    where x = 0, 1, 2, ..., n-1
    """
    n = len(arr)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    for i in range(n):
        sum_x += i
        sum_y += arr[i]
        sum_xy += i * arr[i]
        sum_x2 += i * i

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0, sum_y / n

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


@njit(cache=True)
def nb_std(arr: np.ndarray, ddof: int) -> float:
    """Numba-optimized standard deviation."""
    n = len(arr)
    if n <= ddof:
        return 0.0

    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n

    var = 0.0
    for i in range(n):
        diff = arr[i] - mean
        var += diff * diff
    var /= n - ddof

    return np.sqrt(var)


def tos_stdevall(
    close: IntoExpr,
    length: int | None = None,
    stds: list | None = None,
    ddof: int = 1,
    offset: int = 0,
) -> pl.Expr:
    """Polars: TOS Standard Deviation All

    TD Ameritrade's Think or Swim Standard Deviation All indicator.
    Uses Numba @njit kernel for high performance.

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Bars from current bar. Default: uses all data
        stds: List of standard deviations. Default: [1, 2, 3]
        ddof: Delta Degrees of Freedom. Default: 1
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct with LR line and std deviation bands
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _stds = stds if stds is not None else [1, 2, 3]
    _ddof = ddof
    _length = length

    def compute_stdevall(s: pl.Series) -> pl.Series:
        """Compute TOS_STDEVALL using Numba kernels."""
        arr = s.to_numpy().astype(np.float64)
        n = len(arr)

        # Use length or full series
        if _length is not None and _length < n:
            arr = arr[-_length:]
            n = _length

        # Linear regression using Numba
        slope, intercept = nb_linreg(arr)
        lr = np.empty(n, dtype=np.float64)
        for i in range(n):
            lr[i] = slope * i + intercept

        stdev = nb_std(arr, _ddof)

        # Build result columns
        result_dict = {"LR": lr}
        for i in _stds:
            result_dict[f"L_{i}"] = lr - i * stdev
            result_dict[f"U_{i}"] = lr + i * stdev

        # Return as DataFrame columns via struct
        return pl.DataFrame(result_dict).to_struct("TOS_STDEVALL")

    # Build return dtype dynamically using pl.Field
    struct_fields = [pl.Field("LR", pl.Float64)]
    for i in _stds:
        struct_fields.append(pl.Field(f"L_{i}", pl.Float64))
        struct_fields.append(pl.Field(f"U_{i}", pl.Float64))

    result = close_expr.map_batches(compute_stdevall, return_dtype=pl.Struct(struct_fields))

    if offset != 0:
        result = result.shift(offset)

    return result.alias("TOS_STDEVALL")
