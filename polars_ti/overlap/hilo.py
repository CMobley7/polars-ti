# -*- coding: utf-8 -*-
# =============================================================================
# Polars HILO Implementation
# =============================================================================
import polars as pl
import numpy as np
from numba import njit

from polars_ti._typing import IntoExpr
from polars_ti.ma import ma
from polars_ti.utils._validate import v_expr


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


def hilo(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = "sma",
    talib: bool = True,
    offset: int = 0,
) -> pl.Expr:
    """Polars: Gann HiLo Activator

    The Gann High Low Activator Indicator tracks both curves (of the highs
    and lows). The close of the bar defines which of the two gets plotted.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        high_length: Period for high MA. Default: 13
        low_length: Period for low MA. Default: 21
        mamode: MA type (sma, ema, etc.). Default: "sma"
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with HILO, HILOl, HILOs columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)

    _props = f"_{high_length}_{low_length}"
    hilo_name = f"HILO{_props}"
    long_name = f"HILOl{_props}"
    short_name = f"HILOs{_props}"
    _offset = offset

    high_ma_expr = ma(mamode, high_expr, length=high_length, talib=talib)
    low_ma_expr = ma(mamode, low_expr, length=low_length, talib=talib)

    def compute_hilo(struct: pl.Series) -> pl.Series:
        df = struct.struct.unnest()
        close_np = df["_close"].to_numpy().astype(np.float64)
        high_ma_np = df["_high_ma"].to_numpy().astype(np.float64)
        low_ma_np = df["_low_ma"].to_numpy().astype(np.float64)

        hilo, long, short = nb_hilo(close_np, high_ma_np, low_ma_np)

        if _offset != 0:
            for arr in (hilo, long, short):
                arr[:] = np.roll(arr, _offset)
            if _offset > 0:
                hilo[:_offset] = np.nan
                long[:_offset] = np.nan
                short[:_offset] = np.nan
            else:
                hilo[_offset:] = np.nan
                long[_offset:] = np.nan
                short[_offset:] = np.nan

        n = len(close_np)
        return pl.Series([{hilo_name: hilo[i], long_name: long[i], short_name: short[i]} for i in range(n)])

    fields = [
        pl.Field(hilo_name, pl.Float64),
        pl.Field(long_name, pl.Float64),
        pl.Field(short_name, pl.Float64),
    ]

    return (
        pl.struct(
            [
                close_expr.alias("_close"),
                high_ma_expr.alias("_high_ma"),
                low_ma_expr.alias("_low_ma"),
            ]
        )
        .map_batches(compute_hilo, return_dtype=pl.Struct(fields))
        .alias("HILO")
    )
