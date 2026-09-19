import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars LINREG Implementation
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._rolling import (
    compensated_add,
    needs_scaling,
    rolling_extreme,
    rolling_linear,
    rolling_moments,
    scaled_window_moments,
)
from polars_ti.utils._validate import v_expr, v_pos_int


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
    """Compute stable rolling regression without subtracting large raw moments."""
    n = len(close)
    result = np.full(n, np.nan)
    if length < 2:
        return result
    _, _, covariance = rolling_linear(close, length)
    mean, variance, _, _ = rolling_moments(close, length)
    # Native coordinates are x=1..length. LINREG evaluates at length; TSF
    # projects one step farther at length+1, so their intercepts share this mean.
    x_mean = (length + 1) / 2.0
    x_variation = length * (length * length - 1) / 12.0
    magnitude = rolling_extreme(np.abs(close), length, True)[0]
    for i in range(length - 1, n):
        gradient = covariance[i] / x_variation
        scaled_correlation = np.nan
        if needs_scaling(magnitude[i]):
            _, normalized2, _, _, scale, _ = scaled_window_moments(close, i - length + 1, i + 1)
            anchor = close[i - length + 1] / scale
            total = correction = 0.0
            for j in range(length):
                value = (j + 1 - x_mean) * (close[i - length + 1 + j] / scale - anchor)
                total, correction = compensated_add(total, correction, value)
            centered_covariance = total + correction
            gradient = (centered_covariance / x_variation) * scale
            denominator = x_variation * length * normalized2
            scaled_correlation = centered_covariance / np.sqrt(denominator) if denominator > 0 else 0.0
        if slope:
            result[i] = gradient
        elif intercept:
            result[i] = mean[i] - gradient * x_mean
        elif angle:
            result[i] = np.arctan(gradient)
            if degrees:
                result[i] *= 180.0 / np.pi
        elif r:
            if needs_scaling(magnitude[i]):
                result[i] = scaled_correlation
                continue
            denominator = x_variation * length * variance[i]
            result[i] = covariance[i] / np.sqrt(denominator) if denominator > 0 else 0.0
        else:
            endpoint = length + 1 if tsf else length
            result[i] = mean[i] + gradient * (endpoint - x_mean)
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
    import numpy as np

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _use_talib = Imports["talib"] and v_talib(talib) and length > 1
    _length = v_pos_int(length, "length")

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
