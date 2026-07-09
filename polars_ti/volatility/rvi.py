# -*- coding: utf-8 -*-
# =============================================================================
# Polars RVI Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr
from polars_ti.utils._validate import v_expr


def _rolling_std(arr: np.ndarray, length: int, ddof: int) -> np.ndarray:
    """Rolling standard deviation (pandas ``Series.rolling(length).std()``)."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        window = arr[i - length + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = np.std(window, ddof=ddof)
    return out


def _dispatch_ma(arr: np.ndarray, mamode: str, length: int, talib: bool) -> np.ndarray:
    """Smooth *arr* with the requested MA via the shared ``ma()`` dispatcher.

    The MA is applied from the first valid (non-NaN) index onward, mirroring the
    stoch/trama native-mamode handling, so every MA type supported by ``ma()``
    works for RVI's averaging step.
    """
    from polars_ti.ma import ma

    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.flatnonzero(~np.isnan(arr))
    if valid.size == 0:
        return out
    start = int(valid[0])
    sub = arr[start:]
    smoothed = pl.DataFrame({"_c": sub}).select(ma(mamode, "_c", length=length, talib=talib)).to_series().to_numpy()
    out[start:] = smoothed
    return out


def _pl_rvi_single(
    arr: np.ndarray,
    length: int,
    scalar: float,
    mamode: str,
    drift: int,
    talib: bool = True,
) -> np.ndarray:
    """Core RVI computation on a NumPy array.

    Mirrors OLD pandas-ta ``_rvi``:
        std = stdev(source, length)
        pos, neg = unsigned_differences(source, drift)
        pos_avg = ma(mode, pos * std, length)
        neg_avg = ma(mode, neg * std, length)
        rvi = scalar * pos_avg / (pos_avg + neg_avg)

    Honours ``talib`` (Decision 4): with TA-Lib available and ``talib=True``,
    ``stdev`` is TA-Lib ``STDDEV`` (population, ddof=0) and the averaging MA is
    the TA-Lib EMA/SMA — matching the OLD library's talib-mode output. Native
    mode uses ddof=1 std and the pandas ``ewm`` EMA seed.
    """
    from polars_ti.overlap.ema import _ema_numba
    from polars_ti.maps import Imports

    n = len(arr)
    use_talib = bool(talib) and Imports["talib"]

    if use_talib:
        from talib import STDDEV

        std_arr = STDDEV(arr, length)  # population std (ddof=0)
    else:
        std_arr = _rolling_std(arr, length, ddof=1)

    # unsigned_differences: diff is NaN-filled to 0 at the first bar.
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    if drift < n:
        diff[drift:] = arr[drift:] - arr[:-drift]
        diff[:drift] = 0.0
    pos = (diff > 0).astype(np.float64)
    neg = (diff < 0).astype(np.float64)

    pos_std = pos * std_arr
    neg_std = neg * std_arr

    _mamode = (mamode or "ema").lower()
    if _mamode in ("sma", "ema"):
        # Fast paths preserved byte-for-byte so the default (ema) output is
        # unchanged versus the pre-dispatch implementation.
        if use_talib:
            from talib import EMA, SMA

            if _mamode == "sma":
                pos_avg, neg_avg = SMA(pos_std, length), SMA(neg_std, length)
            else:
                pos_avg, neg_avg = EMA(pos_std, length), EMA(neg_std, length)
        elif _mamode == "sma":
            from polars_ti.momentum.ppo import _sma_numba_ppo

            pos_avg = _sma_numba_ppo(pos_std, length)
            neg_avg = _sma_numba_ppo(neg_std, length)
        else:
            # Native EMA seeded from the first valid value (pandas ewm).
            pos_avg = _ema_numba(pos_std, length, False, False)
            neg_avg = _ema_numba(neg_std, length, False, False)
    else:
        # All other MA types route through the shared ``ma()`` dispatcher,
        # mirroring OLD pandas-ta ``ma(mode, pos_std, length=length)`` (and the
        # trama/stoch native-mamode fixes). Without this branch non-sma/ema
        # modes silently fell back to EMA.
        pos_avg = _dispatch_ma(pos_std, _mamode, length, use_talib)
        neg_avg = _dispatch_ma(neg_std, _mamode, length, use_talib)

    denom = pos_avg + neg_avg
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom != 0, scalar * pos_avg / denom, np.nan)
    return result


def rvi(
    close: IntoExpr,
    high: IntoExpr | None = None,
    low: IntoExpr | None = None,
    length: int = 14,
    scalar: float = 100.0,
    refined: bool = False,
    thirds: bool = False,
    mamode: str = "ema",
    drift: int = 1,
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Relative Volatility Index (RVI)

    RVI adds up standard deviations based on price direction (unlike RSI
    which adds up price changes).

    Sources:
        https://www.tradingview.com/support/solutions/43000594684-relative-volatility-index/

    Args:
        close: Column name or pl.Expr for 'close'
        high: Column name or pl.Expr for 'high' (for refined/thirds mode)
        low: Column name or pl.Expr for 'low' (for refined/thirds mode)
        length: The period. Default: 14
        scalar: Scale factor. Default: 100.0
        refined: Average of RVI(high) and RVI(low). Default: False
        thirds: Average of high, low, and close. Default: False
        mamode: MA type. Default: 'ema'
        drift: The diff period. Default: 1
        talib: Use TA-Lib STDDEV/EMA (matches OLD talib mode) when available.
            Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: RVI expression
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _scalar = scalar
    _drift = drift
    _mamode = mamode
    _refined = refined
    _thirds = thirds
    _talib = talib

    if refined or thirds:
        high_expr = v_expr(high)
        low_expr = v_expr(low)

        def compute_refined(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            c_arr = df["close"].to_numpy().astype(np.float64)
            h_arr = df["high"].to_numpy().astype(np.float64)
            l_arr = df["low"].to_numpy().astype(np.float64)

            high_rvi = _pl_rvi_single(h_arr, _length, _scalar, _mamode, _drift, _talib)
            low_rvi = _pl_rvi_single(l_arr, _length, _scalar, _mamode, _drift, _talib)

            if _thirds:
                close_rvi = _pl_rvi_single(c_arr, _length, _scalar, _mamode, _drift, _talib)
                result = (high_rvi + low_rvi + close_rvi) / 3.0
            else:
                result = 0.5 * (high_rvi + low_rvi)

            return pl.Series(result)

        struct_expr = pl.struct(close=close_expr, high=high_expr, low=low_expr)
        result = struct_expr.map_batches(compute_refined, return_dtype=pl.Float64)
        _mode = "r" if refined else "t"
    else:

        def compute_simple(s: pl.Series) -> pl.Series:
            arr = s.to_numpy().astype(np.float64)
            return pl.Series(_pl_rvi_single(arr, _length, _scalar, _mamode, _drift, _talib))

        result = close_expr.map_batches(compute_simple, return_dtype=pl.Float64)
        _mode = ""

    if offset != 0:
        result = result.shift(offset)

    return result.alias(f"RVI{_mode}_{length}")
