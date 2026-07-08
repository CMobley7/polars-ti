# -*- coding: utf-8 -*-
# =============================================================================
# Polars PPO Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _sma_numba_ppo(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-accelerated SMA (rolling mean, min_periods=length)."""
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
        window_sum = window_sum - values[i - length] + values[i]
        result[i] = window_sum / length

    return result


def ppo(
    close: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Percentage Price Oscillator (PPO)

    The Percentage Price Oscillator is similar to MACD but measures the
    percentage difference between two moving averages instead of the
    absolute difference.

    Sources:
        https://www.investopedia.com/terms/p/ppo.asp

    Args:
        close: Column name or pl.Expr for 'close' prices
        fast: Fast MA period. Default: 12
        slow: Slow MA period. Default: 26
        signal: Signal EMA period. Default: 9
        scalar: Multiplication factor. Default: 100
        mamode: Moving average mode ('sma' or 'ema'). Default: 'sma'
        talib: If True and TA-Lib is installed, use TA-Lib PPO. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with fields:
            - PPO_{fast}_{slow}_{signal}: PPO line
            - PPOs_{fast}_{slow}_{signal}: Signal line
            - PPOh_{fast}_{slow}_{signal}: Histogram
    """
    from polars_ti.maps import Imports
    from polars_ti.overlap.ema import _ema_numba
    from polars_ti.utils import tal_ma, v_talib

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _fast = fast
    _slow = slow
    _signal = signal
    _scalar = scalar
    _mamode = (mamode or "sma").lower()
    _use_talib = Imports["talib"] and v_talib(talib)

    _props = f"_{fast}_{slow}_{signal}"
    ppo_name = f"PPO{_props}"
    signal_name = f"PPOs{_props}"
    hist_name = f"PPOh{_props}"

    def _signal_ema(values: np.ndarray, length: int) -> np.ndarray:
        """EMA of PPO line, matching OLD ``ma('ema', ppo, length=signal)``.

        - Native (talib=False): pandas ewm over the leading-NaN PPO series.
          The presma SMA seed spans the first ``length`` *raw* (NaN) positions,
          so ewm re-seeds from the first finite PPO value (no SMA-warmup delay).
        - talib=True: TA-Lib EMA, which uses an SMA seed over the first
          ``length`` valid values (slice to first_valid first).
        """
        if not _use_talib:
            return _ema_numba(values, length, False, False)
        out = np.full(values.shape, np.nan, dtype=np.float64)
        valid = ~np.isnan(values)
        if not valid.any():
            return out
        first = int(np.argmax(valid))
        out[first:] = _ema_numba(values[first:], length, True, False)
        return out

    def compute_ppo(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)

        if _use_talib:
            from talib import PPO as TALIB_PPO

            # TA-Lib PPO hardcodes scalar=100; rescale so user ``scalar`` is
            # honoured (signal/histogram derive from the scaled line). At
            # scalar=100 this is an exact *1.0 no-op.
            ppo_line = TALIB_PPO(arr, fastperiod=_fast, slowperiod=_slow, matype=tal_ma(_mamode)) * (_scalar / 100.0)
        else:
            if _mamode == "sma":
                fast_ma = _sma_numba_ppo(arr, _fast)
                slow_ma = _sma_numba_ppo(arr, _slow)
            else:
                fast_ma = _ema_numba(arr, _fast, True, False)
                slow_ma = _ema_numba(arr, _slow, True, False)
            with np.errstate(divide="ignore", invalid="ignore"):
                ppo_line = np.where(slow_ma != 0, _scalar * (fast_ma - slow_ma) / slow_ma, np.nan)

        signal_line = _signal_ema(ppo_line, _signal)
        histogram = ppo_line - signal_line

        return pl.DataFrame(
            {
                ppo_name: ppo_line,
                signal_name: signal_line,
                hist_name: histogram,
            }
        ).to_struct("PPO")

    result_expr = close_expr.map_batches(
        compute_ppo,
        return_dtype=pl.Struct(
            [
                pl.Field(ppo_name, pl.Float64),
                pl.Field(signal_name, pl.Float64),
                pl.Field(hist_name, pl.Float64),
            ]
        ),
    )

    if offset != 0:
        result_expr = result_expr.shift(offset)

    return result_expr.alias("PPO")
