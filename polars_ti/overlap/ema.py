# -*- coding: utf-8 -*-
# =============================================================================
# Polars EMA Implementation
# =============================================================================
import polars as pl
from numba import njit
from numpy import empty, float64, nan, isnan


from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _ema_numba(close, length, presma=True, adjust=False):
    """Numba-optimized EMA matching Pandas ewm behavior with presma."""
    n = len(close)
    result = empty(n)
    result[:] = nan

    alpha = 2.0 / (length + 1)

    # First finite index (leading NaNs — e.g. a diff series or a cascaded EMA
    # whose warmup exceeds the window — must never poison the whole column).
    fv = -1
    for i in range(n):
        if not isnan(close[i]):
            fv = i
            break

    if presma and fv >= 0 and fv + length <= n:
        # SMA seed over the first ``length`` FINITE values (contiguous from the
        # first finite index), placed at ``fv+length-1`` — exactly matching
        # TA-Lib's EMA warmup, including on leading-NaN inputs. For a fully finite
        # input (fv=0) this is the mean of close[0:length] at index length-1,
        # identical to before; on a leading-NaN input it seeds one bar later than
        # the old NaN-skipping window (which jumped the gun by a bar).
        seeded = empty(n)
        for i in range(n):
            seeded[i] = close[i]
        sma_sum = 0.0
        for i in range(fv, fv + length):
            sma_sum += close[i]
        for i in range(fv + length - 1):
            seeded[i] = nan
        seeded[fv + length - 1] = sma_sum / length
        close = seeded

    # ewm(adjust=False): seed from the first finite value, recurse, carrying the
    # previous value forward across internal NaNs (pandas ewm semantics).
    first_valid = -1
    for i in range(n):
        if not isnan(close[i]):
            first_valid = i
            break

    if first_valid >= 0:
        result[first_valid] = close[first_valid]
        for i in range(first_valid + 1, n):
            if not isnan(close[i]):
                result[i] = alpha * close[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]

    return result


def ema(
    close: IntoExpr,
    length: int = 10,
    talib: bool = True,
    presma: bool = True,
    adjust: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Exponential Moving Average (EMA)

    The Exponential Moving Average is a more responsive moving average
    compared to the Simple Moving Average (SMA). The weights are determined
    by alpha which is proportional to its length.

    Sources:
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
        https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

    Args:
        close: Column name or pl.Expr for 'close' prices
        length: Span period for EMA calculation. Default: 10
        talib: If True and TA-Lib is installed, uses TA-Lib. Default: True
        presma: If True, uses SMA for initial value like TA-Lib. Default: True
        adjust: Adjust the decay (not used, for API compat). Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: EMA expression for lazy evaluation
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _presma = presma
    _adjust = adjust
    _use_talib = Imports["talib"] and v_talib(talib) and length > 1

    def compute_ema(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(float64)

        if _use_talib:
            from talib import EMA as TALIB_EMA

            result = TALIB_EMA(arr, timeperiod=_length)
        else:
            result = _ema_numba(arr, _length, _presma, _adjust)

        return pl.Series(result)

    ema_expr = close_expr.map_batches(compute_ema, return_dtype=pl.Float64)

    # Apply offset
    if offset != 0:
        ema_expr = ema_expr.shift(offset)

    return ema_expr.alias(f"EMA_{length}")
