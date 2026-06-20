# -*- coding: utf-8 -*-
# =============================================================================
# Polars PPO Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _ema_numba_ppo(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-accelerated EMA calculation with SMA initialization."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    # Find first valid index
    first_valid = 0
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break
    else:
        return result

    if n - first_valid < length:
        return result

    alpha = 2.0 / (length + 1.0)

    # Initialize with SMA
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


@njit(cache=True)
def _sma_numba_ppo(values: np.ndarray, length: int) -> np.ndarray:
    """Numba-accelerated SMA calculation."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < length:
        return result

    # Find first valid
    first_valid = 0
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break
    else:
        return result

    if n - first_valid < length:
        return result

    # First SMA
    window_sum = 0.0
    for i in range(first_valid, first_valid + length):
        window_sum += values[i]
    result[first_valid + length - 1] = window_sum / length

    # Rolling
    for i in range(first_valid + length, n):
        window_sum = window_sum - values[i - length] + values[i]
        result[i] = window_sum / length

    return result


def _ppo_calc(
    close: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
    scalar: float,
    mamode: str,
) -> np.ndarray:
    """Calculate PPO, Signal, and Histogram."""
    # Select MA function
    if mamode.lower() == "sma":
        ma_func = _sma_numba_ppo
    else:
        ma_func = _ema_numba_ppo

    # Calculate fast and slow MAs
    fast_ma = ma_func(close, fast)
    slow_ma = ma_func(close, slow)

    # PPO = scalar * (fast - slow) / slow
    ppo = np.where(slow_ma != 0, scalar * (fast_ma - slow_ma) / slow_ma, np.nan)

    # Signal = EMA of PPO (ewm-style, no SMA init - starts from first valid)
    signal_ma = _ema_ewm_style(ppo, signal)

    # Histogram = PPO - Signal
    histogram = ppo - signal_ma

    return np.column_stack([ppo, signal_ma, histogram])


def _ema_ewm_style(values: np.ndarray, length: int) -> np.ndarray:
    """EMA matching Pandas ewm behavior - first valid value as starting point."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    # Find first valid index
    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break

    if first_valid == -1:
        return result

    alpha = 2.0 / (length + 1.0)

    # Start with the first valid value
    result[first_valid] = values[first_valid]

    # Continue with EMA
    for i in range(first_valid + 1, n):
        if not np.isnan(values[i]):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result


def ppo(
    close: IntoExpr,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
    mamode: str = "sma",
    talib: bool = True,
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

    Returns:
        pl.Expr: Struct expression with columns:
            - PPO_{fast}_{slow}_{signal}: PPO line
            - PPOs_{fast}_{slow}_{signal}: Signal line
            - PPOh_{fast}_{slow}_{signal}: Histogram
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib, tal_ma

    close_expr = v_expr(close)
    if close_expr is None:
        return None

    if slow < fast:
        fast, slow = slow, fast

    _use_talib = Imports["talib"] and v_talib(talib)
    _props = f"_{fast}_{slow}_{signal}"

    if _use_talib:
        # Use TA-Lib PPO
        def compute_ppo_talib(s: pl.Series) -> pl.DataFrame:
            import talib

            arr = s.to_numpy().astype(np.float64)

            # TA-Lib PPO: returns just the PPO line
            ppo_val = talib.PPO(arr, fastperiod=fast, slowperiod=slow, matype=tal_ma(mamode))

            # Calculate signal and histogram using EMA on the PPO
            signal_ma = _ema_ewm_style(ppo_val, signal)
            histogram = ppo_val - signal_ma

            return pl.DataFrame(
                {
                    f"PPO{_props}": ppo_val,
                    f"PPOs{_props}": signal_ma,
                    f"PPOh{_props}": histogram,
                }
            )

        return close_expr.map_batches(
            lambda s: compute_ppo_talib(s).to_struct("PPO"),
            return_dtype=pl.Struct(
                [
                    pl.Field(f"PPO{_props}", pl.Float64),
                    pl.Field(f"PPOs{_props}", pl.Float64),
                    pl.Field(f"PPOh{_props}", pl.Float64),
                ]
            ),
        )
    else:
        # Use Numba implementation
        return close_expr.map_batches(
            lambda s: pl.DataFrame(
                _ppo_calc(s.to_numpy(), fast, slow, signal, scalar, mamode),
                schema=[f"PPO{_props}", f"PPOs{_props}", f"PPOh{_props}"],
            ).to_struct("PPO"),
            return_dtype=pl.Struct(
                [
                    pl.Field(f"PPO{_props}", pl.Float64),
                    pl.Field(f"PPOs{_props}", pl.Float64),
                    pl.Field(f"PPOh{_props}", pl.Float64),
                ]
            ),
        )
