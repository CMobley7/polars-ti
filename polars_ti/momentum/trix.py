# -*- coding: utf-8 -*-
# =============================================================================
# Polars TRIX Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _ema_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba EMA with presma initialization."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1 or n - first_valid < length:
        return result

    sma_sum = 0.0
    for i in range(first_valid, first_valid + length):
        sma_sum += values[i]
    sma_val = sma_sum / length

    result[first_valid + length - 1] = sma_val

    alpha = 2.0 / (length + 1)
    for i in range(first_valid + length, n):
        if not np.isnan(values[i]):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result


@njit(cache=True)
def _sma_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba SMA with NaN handling."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1 or n - first_valid < length:
        return result

    window_sum = 0.0
    for i in range(first_valid, first_valid + length):
        window_sum += values[i]
    result[first_valid + length - 1] = window_sum / length

    for i in range(first_valid + length, n):
        if not np.isnan(values[i]):
            window_sum = window_sum - values[i - length] + values[i]
            result[i] = window_sum / length

    return result


@njit(cache=True)
def _trix_core(
    close_arr: np.ndarray,
    length: int,
    signal: int,
    scalar: float,
    drift: int,
) -> tuple:
    """Numba kernel for TRIX calculation."""
    # Triple EMA
    ema1 = _ema_numba(close_arr, length)
    ema2 = _ema_numba(ema1, length)
    ema3 = _ema_numba(ema2, length)

    n = len(close_arr)

    # TRIX = scalar * pct_change(ema3, drift)
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(drift, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - drift]) and ema3[i - drift] != 0:
            trix[i] = scalar * (ema3[i] - ema3[i - drift]) / ema3[i - drift]

    # Signal = SMA of TRIX
    trix_signal = _sma_numba(trix, signal)

    return trix, trix_signal


def trix(
    close: IntoExpr,
    length: int = 30,
    signal: int = 9,
    scalar: float = 100.0,
    drift: int = 1,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: TRIX (Triple Exponential Average Rate of Change)

    TRIX is a momentum oscillator that displays the percent rate of change
    of a triple exponentially smoothed moving average. It helps identify
    divergences and filter out market noise.

    Sources:
        https://www.tradingview.com/wiki/TRIX

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: EMA period (applied 3 times). Default: 30
        signal: Signal SMA period. Default: 9
        scalar: Multiplication factor. Default: 100
        drift: Periods for pct_change. Default: 1
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            - TRIX_{length}_{signal}: TRIX line
            - TRIXs_{length}_{signal}: Signal line
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if length < signal:
        length, signal = signal, length

    _length = length
    _signal = signal
    _scalar = scalar
    _drift = drift
    _props = f"_{length}_{signal}"
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:

        def compute_trix_talib(s: pl.Series) -> pl.Series:
            from talib import TRIX as TALIB_TRIX

            arr = s.to_numpy().astype(np.float64)

            trix = TALIB_TRIX(arr, timeperiod=_length)
            trix_signal = _sma_numba(trix, _signal)

            return pl.DataFrame(
                {
                    f"TRIX{_props}": trix,
                    f"TRIXs{_props}": trix_signal,
                }
            ).to_struct("TRIX")

        result_expr = close_expr.map_batches(
            compute_trix_talib,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"TRIX{_props}", pl.Float64),
                    pl.Field(f"TRIXs{_props}", pl.Float64),
                ]
            ),
        )
    else:

        def compute_trix_numba(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            trix, trix_signal = _trix_core(arr, _length, _signal, _scalar, _drift)

            return pl.DataFrame(
                {
                    f"TRIX{_props}": trix,
                    f"TRIXs{_props}": trix_signal,
                }
            ).to_struct("TRIX")

        result_expr = close_expr.map_batches(
            compute_trix_numba,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"TRIX{_props}", pl.Float64),
                    pl.Field(f"TRIXs{_props}", pl.Float64),
                ]
            ),
        )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("TRIX")
