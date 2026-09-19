import numpy as np

# -*- coding: utf-8 -*-
# =============================================================================
# Polars MFI (Money Flow Index) Implementation
# =============================================================================
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._rolling import rolling_sum
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    length: int,
) -> np.ndarray:
    """Calculate MFI from compensated rolling positive/negative flows."""
    n = len(close)
    tp = (high + low + close) / 3.0
    positive = np.zeros(n)
    negative = np.zeros(n)
    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            positive[i] = tp[i] * volume[i]
        elif tp[i] < tp[i - 1]:
            negative[i] = tp[i] * volume[i]
    positive = rolling_sum(positive, length)
    negative = rolling_sum(negative, length)
    result = np.full(n, np.nan)
    for i in range(length, n):
        if positive[i] + negative[i] > 0:
            result[i] = 100.0 * positive[i] / (positive[i] + negative[i])
    return result


def mfi(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    length: int = 14,
    talib: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: Money Flow Index (MFI)

    Money Flow Index is an oscillator indicator that measures buying and selling
    pressure by utilizing both price and volume.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        close: Column name or pl.Expr for 'close' prices
        volume: Column name or pl.Expr for 'volume'
        length: Rolling period. Default: 14
        talib: If True and TA-Lib installed, use TA-Lib. Default: True
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: MFI expression
    """
    from polars_ti.maps import Imports
    from polars_ti.utils import v_talib

    high_expr = v_expr(high)
    low_expr = v_expr(low)
    close_expr = v_expr(close)
    volume_expr = v_expr(volume)

    if any(e is None for e in [high_expr, low_expr, close_expr, volume_expr]):
        return None

    _use_talib = Imports["talib"] and v_talib(talib)
    _length = length

    if _use_talib:

        def compute_mfi(df: pl.DataFrame) -> pl.Series:
            from talib import MFI as TALIB_MFI

            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            v = df["volume"].to_numpy().astype(np.float64)
            result = TALIB_MFI(h, l, c, v, timeperiod=_length)
            return pl.Series(f"MFI_{_length}", result)

        mfi_expr = pl.struct(
            [
                high_expr.alias("high"),
                low_expr.alias("low"),
                close_expr.alias("close"),
                volume_expr.alias("volume"),
            ]
        ).map_batches(lambda s: compute_mfi(s.struct.unnest()), return_dtype=pl.Float64)
    else:

        def compute_mfi_numba(df: pl.DataFrame) -> pl.Series:
            h = df["high"].to_numpy().astype(np.float64)
            l = df["low"].to_numpy().astype(np.float64)
            c = df["close"].to_numpy().astype(np.float64)
            v = df["volume"].to_numpy().astype(np.float64)
            result = _nb_mfi(h, l, c, v, _length)
            return pl.Series(f"MFI_{_length}", result)

        mfi_expr = pl.struct(
            [
                high_expr.alias("high"),
                low_expr.alias("low"),
                close_expr.alias("close"),
                volume_expr.alias("volume"),
            ]
        ).map_batches(lambda s: compute_mfi_numba(s.struct.unnest()), return_dtype=pl.Float64)

    if offset != 0:
        mfi_expr = mfi_expr.shift(offset)

    return mfi_expr.alias(f"MFI_{length}")
