# -*- coding: utf-8 -*-
# =============================================================================
# Polars MACD Implementation
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def macd(
    close: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    as_mode: bool = False,
    talib: bool = True,
    offset: int = 0,
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
        talib: If True and TA-Lib is installed, use TA-Lib MACD. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with fields:
            - MACD_{fast}_{slow}_{signal}: MACD line
            - MACDs_{fast}_{slow}_{signal}: Signal line
            - MACDh_{fast}_{slow}_{signal}: Histogram
    """
    from polars_ti.maps import Imports
    from polars_ti.overlap.ema import _ema_numba
    from polars_ti.utils import v_talib

    def _signal_ema(values: np.ndarray, length: int) -> np.ndarray:
        """EMA of *values* seeded from the first valid value.

        Mirrors OLD pandas-ta which slices the MACD line to its
        ``first_valid_index`` before applying the EMA, so the presma SMA
        seed spans the first ``length`` *valid* values, not the first
        ``length`` raw positions (which would be NaN during warmup).
        """
        out = np.full(values.shape, np.nan, dtype=np.float64)
        valid = ~np.isnan(values)
        if not valid.any():
            return out
        first = int(np.argmax(valid))
        out[first:] = _ema_numba(values[first:], length, True, False)
        return out

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _fast = fast
    _slow = slow
    _signal = signal
    _as_mode = as_mode
    _use_talib = Imports["talib"] and v_talib(talib)

    _as = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"
    macd_name = f"MACD{_as}{_props}"
    signal_name = f"MACD{_as}s{_props}"
    hist_name = f"MACD{_as}h{_props}"

    def compute_macd(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib:
            from talib import MACD as TALIB_MACD

            macd_line, signal_line, histogram = TALIB_MACD(arr, _fast, _slow, _signal)
        else:
            fast_ema = _ema_numba(arr, _fast, True, False)
            slow_ema = _ema_numba(arr, _slow, True, False)
            macd_line = fast_ema - slow_ema
            # Signal = EMA of MACD seeded from the first valid MACD value
            # (matches OLD pandas-ta which slices to first_valid_index first).
            signal_line = _signal_ema(macd_line, _signal)
            histogram = macd_line - signal_line

        if _as_mode:
            macd_line = macd_line - signal_line
            signal_line = _signal_ema(macd_line, _signal)
            histogram = macd_line - signal_line

        return pl.DataFrame(
            {
                macd_name: macd_line,
                signal_name: signal_line,
                hist_name: histogram,
            }
        ).to_struct("MACD")

    result_expr = close_expr.map_batches(
        compute_macd,
        return_dtype=pl.Struct(
            [
                pl.Field(macd_name, pl.Float64),
                pl.Field(signal_name, pl.Float64),
                pl.Field(hist_name, pl.Float64),
            ]
        ),
    )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("MACD")
