# -*- coding: utf-8 -*-
# =============================================================================
# Polars MACD Implementation
# =============================================================================
import polars as pl
from numba import njit
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _ema_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-accelerated EMA calculation with SMA initialization (TA-Lib style)."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    # Find first valid index (skip leading NaNs)
    first_valid = 0
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break
    else:
        return result  # All NaN

    # Check if we have enough values after first valid
    if n - first_valid < length:
        return result

    alpha = 2.0 / (length + 1.0)

    # Initialize with SMA of first 'length' valid values
    sma = 0.0
    for i in range(first_valid, first_valid + length):
        sma += values[i]

    sma_idx = first_valid + length - 1
    result[sma_idx] = sma / length

    # Continue with EMA
    for i in range(sma_idx + 1, n):
        if not np.isnan(values[i]):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result


def _macd_calc(
    close_arr: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
    as_mode: bool,
) -> np.ndarray:
    """Calculate MACD, Signal, and Histogram using Numba EMA."""
    n = len(close_arr)

    # Calculate fast and slow EMAs
    fast_ema = _ema_numba(close_arr, fast)
    slow_ema = _ema_numba(close_arr, slow)

    # MACD = Fast EMA - Slow EMA
    macd = fast_ema - slow_ema

    # Signal = EMA of MACD (starting from first valid MACD)
    signal_ema = _ema_numba(macd, signal)

    # Histogram = MACD - Signal
    histogram = macd - signal_ema

    # AS Mode: Apply transformation
    if as_mode:
        macd = macd - signal_ema
        signal_ema = _ema_numba(macd, signal)
        histogram = macd - signal_ema

    # Stack into 2D array: [macd, signal, histogram]
    return np.column_stack([macd, signal_ema, histogram])


def pl_macd(
    close: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    as_mode: bool = False,
) -> PlExpr:
    """Polars: Moving Average Convergence Divergence (MACD)

    The MACD is a popular indicator used to identify a security's trend.
    It calculates the difference between fast and slow EMAs, along with
    a signal line (EMA of MACD) and histogram (MACD - Signal).

    Sources:
        https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)
        https://www.investopedia.com/terms/m/macd.asp

    Args:
        close: Column name or pl.Expr for 'close' prices
        fast: Short period EMA. Default: 12
        slow: Long period EMA. Default: 26
        signal: Signal line EMA period. Default: 9
        as_mode: Enable AS (Alternative Signal) mode. Default: False

    Returns:
        pl.Expr: Struct expression with columns:
            - MACD_{fast}_{slow}_{signal}: MACD line
            - MACDs_{fast}_{slow}_{signal}: Signal line
            - MACDh_{fast}_{slow}_{signal}: Histogram
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _as = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"

    return close_expr.map_batches(
        lambda s: pl.DataFrame(
            _macd_calc(s.to_numpy(), fast, slow, signal, as_mode),
            schema=[
                f"MACD{_as}{_props}",
                f"MACD{_as}s{_props}",
                f"MACD{_as}h{_props}",
            ],
        ).to_struct("MACD"),
        return_dtype=pl.Struct(
            [
                pl.Field(f"MACD{_as}{_props}", pl.Float64),
                pl.Field(f"MACD{_as}s{_props}", pl.Float64),
                pl.Field(f"MACD{_as}h{_props}", pl.Float64),
            ]
        ),
    )
