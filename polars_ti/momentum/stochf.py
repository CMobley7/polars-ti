# -*- coding: utf-8 -*-
# =============================================================================
# Polars Fast Stochastic Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _sma_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized Simple Moving Average handling NaNs."""
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
def _stochf_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: int,
    d: int,
) -> tuple:
    """Numba kernel for Fast Stochastic Oscillator calculation."""
    n = len(close)

    # Calculate lowest low and highest high over k periods
    lowest_low = np.full(n, np.nan, dtype=np.float64)
    highest_high = np.full(n, np.nan, dtype=np.float64)

    for i in range(k - 1, n):
        ll = low[i]
        hh = high[i]
        for j in range(i - k + 1, i):
            if low[j] < ll:
                ll = low[j]
            if high[j] > hh:
                hh = high[j]
        lowest_low[i] = ll
        highest_high[i] = hh

    # Fast %K = 100 * (close - ll) / (hh - ll) - NO smoothing
    stochf_k = np.full(n, np.nan, dtype=np.float64)
    for i in range(k - 1, n):
        range_val = highest_high[i] - lowest_low[i]
        if range_val != 0:
            stochf_k[i] = 100.0 * (close[i] - lowest_low[i]) / range_val

    # %D = SMA(Fast %K, d)
    stochf_d = _sma_numba(stochf_k, d)

    return stochf_k, stochf_d


def pl_stochf(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    k: int = 14,
    d: int = 3,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Fast Stochastic Oscillator (STOCHF)

    The Fast Stochastic Oscillator was developed by George Lane in the 1950's.
    Unlike the Slow Stochastic (STOCH), STOCHF does not smooth %K, making it
    more volatile and responsive to price changes.

    Sources:
        https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=333&Name=KD_-_Fast
        https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/fast-stochastic-indicator/

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        k: Fast %K period (lookback for high/low range). Default: 14
        d: %D period (MA of Fast %K). Default: 3
        mamode: MA type for %D smoothing ('sma', 'ema', etc.). Default: 'sma'
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            - STOCHFk_{k}_{d}: Fast %K line (unsmoothed)
            - STOCHFd_{k}_{d}: %D signal line (MA of Fast %K)
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib, tal_ma

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    if high_expr is None or low_expr is None or close_expr is None:
        return None

    _k = k
    _d = d
    _mamode = mamode.lower() if isinstance(mamode, str) else "sma"
    _props = f"_{k}_{d}"
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:
        # TA-Lib path: use STOCHF function
        def compute_stochf_talib(s: pl.Series) -> pl.Series:
            from talib import STOCHF as TALIB_STOCHF

            high_arr = s.struct.field("high").to_numpy().astype(np.float64)
            low_arr = s.struct.field("low").to_numpy().astype(np.float64)
            close_arr = s.struct.field("close").to_numpy().astype(np.float64)

            stochf_k, stochf_d = TALIB_STOCHF(
                high_arr,
                low_arr,
                close_arr,
                fastk_period=_k,
                fastd_period=_d,
                fastd_matype=tal_ma(_mamode),
            )

            return pl.DataFrame(
                {
                    f"STOCHFk{_props}": stochf_k,
                    f"STOCHFd{_props}": stochf_d,
                }
            ).to_struct("STOCHF")

        result_expr = pl.struct(
            high=high_expr,
            low=low_expr,
            close=close_expr,
        ).map_batches(
            compute_stochf_talib,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHFk{_props}", pl.Float64),
                    pl.Field(f"STOCHFd{_props}", pl.Float64),
                ]
            ),
        )
    else:
        # Pure Polars + Numba path
        def compute_stochf_numba(s: pl.Series) -> pl.Series:
            high_arr = s.struct.field("high").to_numpy().astype(np.float64)
            low_arr = s.struct.field("low").to_numpy().astype(np.float64)
            close_arr = s.struct.field("close").to_numpy().astype(np.float64)

            stochf_k, stochf_d = _stochf_core(high_arr, low_arr, close_arr, _k, _d)

            return pl.DataFrame(
                {
                    f"STOCHFk{_props}": stochf_k,
                    f"STOCHFd{_props}": stochf_d,
                }
            ).to_struct("STOCHF")

        result_expr = pl.struct(
            high=high_expr,
            low=low_expr,
            close=close_expr,
        ).map_batches(
            compute_stochf_numba,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHFk{_props}", pl.Float64),
                    pl.Field(f"STOCHFd{_props}", pl.Float64),
                ]
            ),
        )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("STOCHF")
