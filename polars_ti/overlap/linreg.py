# -*- coding: utf-8 -*-
# =============================================================================
# Polars LINREG Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def nb_linreg(
    close: np.ndarray,
    length: int,
    angle: bool,
    degrees: bool,
    intercept: bool,
    r: bool,
    slope: bool,
    tsf: bool,
) -> np.ndarray:
    """Numba-optimized rolling linear regression calculation."""
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[: length - 1] = np.nan

    # Precompute constants
    x_sum = 0.5 * length * (length + 1)
    x2_sum = x_sum * (2 * length + 1) / 3
    divisor = length * x2_sum - x_sum * x_sum

    for i in range(length - 1, n):
        # Get window
        window = close[i - length + 1 : i + 1]

        # Compute sums
        y_sum = 0.0
        xy_sum = 0.0
        y2_sum = 0.0
        for j in range(length):
            y_sum += window[j]
            xy_sum += (j + 1) * window[j]
            y2_sum += window[j] * window[j]

        # Slope (m)
        m = (length * xy_sum - x_sum * y_sum) / divisor

        if slope:
            result[i] = m
            continue

        # Intercept (b)
        b = (y_sum * x2_sum - x_sum * xy_sum) / divisor

        if intercept:
            result[i] = b
            continue

        if angle:
            theta = np.arctan(m)
            if degrees:
                theta *= 180.0 / np.pi
            result[i] = theta
            continue

        if r:
            rn = length * xy_sum - x_sum * y_sum
            rd_sq = divisor * (length * y2_sum - y_sum * y_sum)
            if rd_sq > 0:
                rd = np.sqrt(rd_sq)
                result[i] = rn / rd
            else:
                result[i] = 0.0
            continue

        # Default: LINREG value or TSF
        if tsf:
            result[i] = m * (length - 1) + b
        else:
            result[i] = m * length + b

    return result


def linreg(
    close: IntoExpr,
    length: int = 14,
    talib: bool = True,
    offset: int = 0,
    angle: bool = False,
    degrees: bool = False,
    intercept: bool = False,
    r: bool = False,
    slope: bool = False,
    tsf: bool = False,
) -> PlExpr:
    """Polars: Linear Regression Moving Average (LINREG)

    Linear Regression Moving Average (LINREG). This is a simplified version
    of a Standard Linear Regression.

    Source: TA Lib

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Rolling window period. Default: 14
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0
        angle: If True, returns the slope angle in radians. Default: False
        degrees: If True, returns angle in degrees. Default: False
        intercept: If True, returns the intercept. Default: False
        r: If True, returns correlation 'r'. Default: False
        slope: If True, returns the slope. Default: False
        tsf: If True, returns Time Series Forecast. Default: False

    Returns:
        pl.Expr: LINREG expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib
    import numpy as np

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = length

    # LINREG is complex with multiple modes - use map_batches
    def compute_linreg(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib and not (angle or degrees or intercept or r or slope or tsf):
            # Basic LINREG - can use TA-Lib
            from talib import LINEARREG

            result = LINEARREG(arr, timeperiod=_length)
        elif _use_talib and slope and not (angle or degrees or intercept or r):
            from talib import LINEARREG_SLOPE

            result = LINEARREG_SLOPE(arr, timeperiod=_length)
        elif _use_talib and intercept and not (angle or degrees or r or slope):
            from talib import LINEARREG_INTERCEPT

            result = LINEARREG_INTERCEPT(arr, timeperiod=_length)
        elif _use_talib and angle and not (degrees or intercept or r or slope):
            from talib import LINEARREG_ANGLE

            result = LINEARREG_ANGLE(arr, timeperiod=_length)
        elif _use_talib and tsf and not (angle or degrees or intercept or r or slope):
            from talib import TSF

            result = TSF(arr, timeperiod=_length)
        else:
            # Complex modes or no TA-Lib - use Numba
            result = nb_linreg(arr, _length, angle, degrees, intercept, r, slope, tsf)

        return pl.Series(result)

    result = close_expr.map_batches(compute_linreg, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        result = result.shift(offset)

    # Build name
    name = "LINREG"
    if slope:
        name += "m"
    if intercept:
        name += "b"
    if angle:
        name += "a"
    if r:
        name += "r"
    name += f"_{length}"

    return result.alias(name)
