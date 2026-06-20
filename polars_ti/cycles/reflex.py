# -*- coding: utf-8 -*-
from numpy import cos, exp, nan, sqrt, zeros_like
from numba import njit


@njit(cache=True)
def np_reflex(x, n, k, alpha, pi, sqrt2):
    m, ratio = x.size, 2 * sqrt2 / k
    a = exp(-pi * ratio)
    b = 2 * a * cos(180 * ratio)
    c = a * a - b + 1

    _f = zeros_like(x, dtype=x.dtype)
    _ms = zeros_like(x, dtype=x.dtype)
    result = zeros_like(x, dtype=x.dtype)

    for i in range(2, m):
        _f[i] = 0.5 * c * (x[i] + x[i - 1]) + b * _f[i - 1] - a * a * _f[i - 2]

    for i in range(n, m):
        slope = (_f[i - n] - _f[i]) / n

        _sum = 0
        for j in range(1, n):
            _sum += _f[i] - _f[i - j] + j * slope
        _sum /= n

        _ms[i] = alpha * _sum * _sum + (1 - alpha) * _ms[i - 1]
        if _ms[i] != 0.0:
            result[i] = _sum / sqrt(_ms[i])

    return result


# =============================================================================
# Polars REFLEX Implementation
# =============================================================================
import polars as pl
from numpy import nan

from polars_ti._typing import IntoExpr


def reflex(
    close: str = "close",
    length: int = 20,
    smooth: int = 20,
    alpha: float = 0.04,
    pi: float = 3.14159,
    sqrt2: float = 1.414,
    offset: int = 0,
) -> callable:
    """Polars: Reflex Indicator

    John F. Ehlers' lag-reduced cycle indicator from TASC Feb 2020.
    Oscillator focused on cycle detection.

    This function returns a compute function due to the recursive nature.

    Sources:
        http://traders.com/Documentation/FEEDbk_docs/2020/02/TradersTips.html

    Args:
        close: Column name for 'close' prices. Default: "close"
        length: Period. Default: 20
        smooth: Period of internal SuperSmoother. Default: 20
        alpha: Alpha weight. Default: 0.04
        pi: PI value. Default: 3.14159
        sqrt2: sqrt(2) value. Default: 1.414
        offset: Shift result by N periods. Default: 0

    Returns:
        callable: Function to apply to DataFrame
    """
    _offset = offset  # Capture for closure

    def compute_reflex(df: pl.DataFrame) -> pl.DataFrame:
        np_close = df[close].to_numpy()
        result = np_reflex(np_close, length, smooth, alpha, pi, sqrt2)
        result[:length] = nan
        result_df = pl.DataFrame({f"REFLEX_{length}_{smooth}_{alpha}": result})

        # Apply offset if needed
        if _offset != 0:
            result_df = result_df.select([pl.all().shift(_offset)])

        return result_df

    return compute_reflex


def reflex_apply(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    """Apply Reflex to a DataFrame.

    Args:
        df: Polars DataFrame with close column
        **kwargs: Parameters (close, length, smooth, alpha, pi, sqrt2)

    Returns:
        pl.DataFrame: Original DataFrame with REFLEX column added
    """
    close = kwargs.get("close", "close")
    length = kwargs.get("length", 20)
    smooth = kwargs.get("smooth", 20)
    alpha = kwargs.get("alpha", 0.04)
    pi_val = kwargs.get("pi", 3.14159)
    sqrt2 = kwargs.get("sqrt2", 1.414)

    compute_fn = reflex(close, length, smooth, alpha, pi_val, sqrt2)
    reflex_df = compute_fn(df)
    return df.hstack(reflex_df)
