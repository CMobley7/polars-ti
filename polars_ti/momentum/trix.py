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
    from polars_ti.overlap.ema import ema

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

    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    _use_talib = Imports["talib"] and v_talib(talib)
    _struct_dtype = pl.Struct(
        [
            pl.Field(f"TRIX{_props}", pl.Float64),
            pl.Field(f"TRIXs{_props}", pl.Float64),
        ]
    )

    if _use_talib and _scalar == 100.0 and _drift == 1:
        # Fast path: TA-Lib's dedicated ``TRIX`` is a single C call — much cheaper
        # than three ``talib.EMA`` calls plus a Python rate-of-change. It hardcodes
        # scalar=100/drift=1, so this path is only taken at those defaults (the
        # composition below handles non-default scalar/drift).
        def compute_trix(s: pl.Series) -> pl.Series:
            from talib import TRIX as _TRIX

            trix = _TRIX(s.to_numpy().astype(np.float64), timeperiod=_length)
            trix_signal = _sma_numba(trix, _signal)
            return pl.DataFrame({f"TRIX{_props}": trix, f"TRIXs{_props}": trix_signal}).to_struct("TRIX")

        result_expr = close_expr.map_batches(compute_trix, return_dtype=_struct_dtype)
    else:
        # The triple EMA is delegated to the library's own ``ema`` (native
        # NaN-tolerant EMA, which agrees with TA-Lib to float noise). The
        # scalar/drift rate-of-change and the SMA signal are then applied once, so
        # ``scalar``/``drift`` are always honoured.
        ema3_expr = ema(ema(ema(close_expr, _length, talib=talib), _length, talib=talib), _length, talib=talib)

        def compute_trix(s: pl.Series) -> pl.Series:
            ema3 = s.to_numpy().astype(np.float64)

            trix = np.full(ema3.shape, np.nan, dtype=np.float64)
            for i in range(_drift, len(ema3)):
                prev = ema3[i - _drift]
                if not np.isnan(ema3[i]) and not np.isnan(prev) and prev != 0:
                    trix[i] = _scalar * (ema3[i] / prev - 1.0)
            trix_signal = _sma_numba(trix, _signal)

            return pl.DataFrame({f"TRIX{_props}": trix, f"TRIXs{_props}": trix_signal}).to_struct("TRIX")

        result_expr = ema3_expr.map_batches(compute_trix, return_dtype=_struct_dtype)

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("TRIX")
