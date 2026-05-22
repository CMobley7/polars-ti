# -*- coding: utf-8 -*-
# =============================================================================
# Polars HILO Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.ma import pl_ma


@njit(cache=True)
def nb_hilo(close: np.ndarray, high_ma: np.ndarray, low_ma: np.ndarray) -> tuple:
    """Numba-optimized HILO calculation.

    Returns: (hilo, long, short) arrays
    """
    m = len(close)
    hilo = np.empty(m, dtype=np.float64)
    long = np.empty(m, dtype=np.float64)
    short = np.empty(m, dtype=np.float64)

    hilo[0] = np.nan
    long[0] = np.nan
    short[0] = np.nan

    for i in range(1, m):
        if close[i] > high_ma[i - 1]:
            hilo[i] = low_ma[i]
            long[i] = low_ma[i]
            short[i] = np.nan
        elif close[i] < low_ma[i - 1]:
            hilo[i] = high_ma[i]
            long[i] = np.nan
            short[i] = high_ma[i]
        else:
            hilo[i] = hilo[i - 1]
            long[i] = hilo[i - 1]
            short[i] = hilo[i - 1]

    return hilo, long, short


def pl_hilo(
    df: pl.DataFrame,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> pl.DataFrame:
    """Polars: Gann HiLo Activator

    The Gann High Low Activator Indicator tracks both curves (of the highs
    and lows). The close of the bar defines which of the two gets plotted.

    Args:
        df: Polars DataFrame with OHLC columns
        high: Column name for 'high' prices. Default: "high"
        low: Column name for 'low' prices. Default: "low"
        close: Column name for 'close' prices. Default: "close"
        high_length: Period for high MA. Default: 13
        low_length: Period for low MA. Default: 21
        mamode: MA type (sma, ema, etc.). Default: "sma"
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.DataFrame: Original DataFrame with HILO, HILOl, HILOs columns
    """
    # Calculate MAs
    high_ma = df.select(pl_ma(mamode, high, length=high_length, talib=talib).alias("high_ma")).get_column("high_ma")
    low_ma = df.select(pl_ma(mamode, low, length=low_length, talib=talib).alias("low_ma")).get_column("low_ma")
    close_arr = df.get_column(close)

    # Convert to numpy for Numba kernel
    high_ma_np = high_ma.to_numpy().astype(np.float64)
    low_ma_np = low_ma.to_numpy().astype(np.float64)
    close_np = close_arr.to_numpy().astype(np.float64)

    # Call Numba kernel
    hilo, long, short = nb_hilo(close_np, high_ma_np, low_ma_np)

    # Apply offset
    if offset != 0:
        hilo = np.roll(hilo, offset)
        long = np.roll(long, offset)
        short = np.roll(short, offset)
        if offset > 0:
            hilo[:offset] = np.nan
            long[:offset] = np.nan
            short[:offset] = np.nan
        else:
            hilo[offset:] = np.nan
            long[offset:] = np.nan
            short[offset:] = np.nan

    # Create result columns
    _props = f"_{high_length}_{low_length}"
    return df.with_columns(
        [
            pl.Series(f"HILO{_props}", hilo),
            pl.Series(f"HILOl{_props}", long),
            pl.Series(f"HILOs{_props}", short),
        ]
    )
