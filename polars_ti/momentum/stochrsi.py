# -*- coding: utf-8 -*-
# =============================================================================
# Polars StochRSI Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _rsi_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba RSI with Wilder's smoothing."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length + 1:
        return result

    alpha = 1.0 / length

    deltas = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        deltas[i] = values[i] - values[i - 1]

    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, length + 1):
        if deltas[i] > 0:
            gain_sum += deltas[i]
        else:
            loss_sum += abs(deltas[i])

    avg_gain = gain_sum / length
    avg_loss = loss_sum / length

    if avg_loss == 0:
        result[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[length] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(length + 1, n):
        delta = deltas[i]
        if delta > 0:
            gain = delta
            loss = 0.0
        else:
            gain = 0.0
            loss = abs(delta)

        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


@njit(cache=True)
def _sma_numba(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-optimized SMA with rolling-mean (min_periods=length) semantics.

    A window is only defined when every one of its ``length`` values is finite,
    matching pandas ``Series.rolling(length).mean()`` (and the OLD pandas-ta
    ``nb_sma`` convolution). Each maximal contiguous run of finite values is
    accumulated independently with a sliding sum, so an interior NaN cleanly
    breaks the window and the mean recovers on the next full valid window
    (rather than desyncing the running sum or poisoning the whole tail). For the
    common single-run case (leading-NaN warmup then all-valid) the sliding sum
    is identical to the previous implementation, so output stays byte-identical.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    i = 0
    while i < n:
        if np.isnan(values[i]):
            i += 1
            continue

        # Extent of the current contiguous run of finite values: [i, run_end).
        run_end = i
        while run_end < n and not np.isnan(values[run_end]):
            run_end += 1

        if run_end - i >= length:
            window_sum = 0.0
            for j in range(i, i + length):
                window_sum += values[j]
            result[i + length - 1] = window_sum / length
            for j in range(i + length, run_end):
                window_sum = window_sum - values[j - length] + values[j]
                result[j] = window_sum / length

        i = run_end

    return result


@njit(cache=True)
def _stochrsi_raw_core(
    close: np.ndarray,
    length: int,
    rsi_length: int,
) -> np.ndarray:
    """Numba kernel for the raw StochRSI (pre %K/%D smoothing)."""
    n = len(close)

    # 1. Calculate RSI
    rsi = _rsi_numba(close, rsi_length)

    # 2. Calculate rolling min/max of RSI over 'length' periods
    lowest_rsi = np.full(n, np.nan, dtype=np.float64)
    highest_rsi = np.full(n, np.nan, dtype=np.float64)

    for i in range(rsi_length + length - 1, n):
        window_start = i - length + 1
        # Check if all values in window are valid
        all_valid = True
        for j in range(window_start, i + 1):
            if np.isnan(rsi[j]):
                all_valid = False
                break

        if all_valid:
            min_val = rsi[window_start]
            max_val = rsi[window_start]
            for j in range(window_start + 1, i + 1):
                if rsi[j] < min_val:
                    min_val = rsi[j]
                if rsi[j] > max_val:
                    max_val = rsi[j]
            lowest_rsi[i] = min_val
            highest_rsi[i] = max_val

    # 3. Calculate raw StochRSI
    stochrsi_raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(lowest_rsi[i]) and not np.isnan(highest_rsi[i]):
            range_val = highest_rsi[i] - lowest_rsi[i]
            if range_val != 0:
                stochrsi_raw[i] = 100.0 * (rsi[i] - lowest_rsi[i]) / range_val

    return stochrsi_raw


def _stochrsi_smooth(
    stochrsi_raw: np.ndarray,
    k: int,
    d: int,
    mamode: str,
    use_talib: bool,
) -> tuple:
    """Smooth the raw StochRSI into %K/%D honouring ``mamode``.

    The default "sma" keeps the local NaN-tolerant ``_sma_numba`` kernel so the
    output is byte-identical to the prior implementation (golden-safe). Any
    other ``mamode`` is routed through the shared ``ma()`` dispatcher.
    """
    if mamode == "sma":
        stochrsi_k = _sma_numba(stochrsi_raw, k)
        stochrsi_d = _sma_numba(stochrsi_k, d)
        return stochrsi_k, stochrsi_d

    from polars_ti.ma import ma

    def _smooth(values: np.ndarray, length: int) -> np.ndarray:
        frame = pl.DataFrame({"_v": values.astype(np.float64)})
        return frame.select(ma(mamode, "_v", length=length, talib=use_talib)).to_series().to_numpy()

    stochrsi_k = _smooth(stochrsi_raw, k)
    stochrsi_d = _smooth(stochrsi_k, d)
    return stochrsi_k, stochrsi_d


def stochrsi(
    close: IntoExpr,
    length: int = 14,
    rsi_length: int = 14,
    k: int = 3,
    d: int = 3,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Stochastic RSI (STOCHRSI)

    Created by Tushar Chande and Stanley Kroll. It applies the Stochastic
    formula to RSI values instead of price data, resulting in an indicator
    that ranges from 0 to 100.

    Sources:
        https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI)
        Stock & Commodities V.11:5 (189-199)

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: StochRSI lookback period. Default: 14
        rsi_length: RSI calculation period. Default: 14
        k: Fast %K smoothing period. Default: 3
        d: Slow %D smoothing period. Default: 3
        mamode: MA type for smoothing. Default: 'sma'
        talib: If True and TA-Lib installed, use TA-Lib RSI. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with columns:
            - STOCHRSIk_{length}_{rsi_length}_{k}_{d}: %K line
            - STOCHRSId_{length}_{rsi_length}_{k}_{d}: %D line
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _rsi_length = rsi_length
    _k = k
    _d = d
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    _mamode = mamode
    _use_talib = Imports["talib"] and v_talib(talib)

    if _use_talib:
        # Use TA-Lib RSI, then apply Stochastic formula
        def compute_stochrsi_talib(s: pl.Series) -> pl.Series:
            from talib import RSI as TALIB_RSI

            arr = s.to_numpy().astype(np.float64)

            # Get RSI from TA-Lib
            rsi = TALIB_RSI(arr, timeperiod=_rsi_length)

            # Apply Stochastic to RSI
            n = len(rsi)
            lowest_rsi = np.full(n, np.nan, dtype=np.float64)
            highest_rsi = np.full(n, np.nan, dtype=np.float64)

            for i in range(_length - 1, n):
                window = rsi[i - _length + 1 : i + 1]
                if not np.any(np.isnan(window)):
                    lowest_rsi[i] = np.min(window)
                    highest_rsi[i] = np.max(window)

            # Raw StochRSI
            range_val = highest_rsi - lowest_rsi
            range_val = np.where(range_val == 0, np.nan, range_val)
            stochrsi_raw = 100.0 * (rsi - lowest_rsi) / range_val

            # %K and %D smoothing (honour mamode; default "sma")
            stochrsi_k, stochrsi_d = _stochrsi_smooth(stochrsi_raw, _k, _d, _mamode, _use_talib)

            return pl.DataFrame(
                {
                    f"STOCHRSIk{_props}": stochrsi_k,
                    f"STOCHRSId{_props}": stochrsi_d,
                }
            ).to_struct("STOCHRSI")

        result_expr = close_expr.map_batches(
            compute_stochrsi_talib,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHRSIk{_props}", pl.Float64),
                    pl.Field(f"STOCHRSId{_props}", pl.Float64),
                ]
            ),
        )
    else:
        # Pure Numba path
        def compute_stochrsi_numba(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            stochrsi_raw = _stochrsi_raw_core(arr, _length, _rsi_length)
            stochrsi_k, stochrsi_d = _stochrsi_smooth(stochrsi_raw, _k, _d, _mamode, _use_talib)

            return pl.DataFrame(
                {
                    f"STOCHRSIk{_props}": stochrsi_k,
                    f"STOCHRSId{_props}": stochrsi_d,
                }
            ).to_struct("STOCHRSI")

        result_expr = close_expr.map_batches(
            compute_stochrsi_numba,
            return_dtype=pl.Struct(
                [
                    pl.Field(f"STOCHRSIk{_props}", pl.Float64),
                    pl.Field(f"STOCHRSId{_props}", pl.Float64),
                ]
            ),
        )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("STOCHRSI")
