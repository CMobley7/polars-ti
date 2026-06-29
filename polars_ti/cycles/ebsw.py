# -*- coding: utf-8 -*-
# =============================================================================
# Polars EBSW Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


@njit(cache=True)
def _nb_ebsw(close: np.ndarray, length: int, bars: int, initial: bool) -> np.ndarray:
    """Even Better SineWave recursion (ports pandas-ta ``ebsw``)."""
    m = close.size
    out = np.full(m, np.nan, dtype=np.float64)
    if m <= length:
        return out
    out[length - 1] = 0.0

    last_hp = 0.0
    last_close = 0.0
    # filter history [oldest, mid, newest]
    f0 = 0.0
    f1 = 0.0
    f2 = 0.0

    if initial:
        # OLD "initial version" uses the raw degree values as radians (as-is).
        alpha1 = (1.0 - np.sin(360.0 / length)) / np.cos(360.0 / length)
        a1 = np.exp(-np.sqrt(2.0) * np.pi / bars)
        c2 = 2.0 * a1 * np.cos(np.sqrt(2.0) * 180.0 / bars)
    else:
        angle = 2.0 * np.pi / length
        alpha1 = (1.0 - np.sin(angle)) / np.cos(angle)
        ang = np.sqrt(2.0) * np.pi / bars
        a1 = np.exp(-ang)
        c2 = 2.0 * a1 * np.cos(ang)
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3

    for i in range(length, m):
        hp = 0.5 * (1.0 + alpha1) * (close[i] - last_close) + alpha1 * last_hp

        # roll(filtHist, -1) then overwrite the last element:
        # new uses the (pre-roll) newest f2 and mid f1.
        new = 0.5 * c1 * (hp + last_hp) + c2 * f2 + c3 * f1
        f0, f1, f2 = f1, f2, new

        wave = (f0 + f1 + f2) / 3.0
        power = (f0 * f0 + f1 * f1 + f2 * f2) / 3.0
        out[i] = wave / np.sqrt(power)

        last_hp = hp
        last_close = close[i]

    return out


def ebsw(
    close: IntoExpr,
    length: int = 40,
    bars: int = 10,
    initial_version: bool = False,
    offset: int = 0,
) -> PlExpr:
    """Polars: Even Better SineWave (EBSW)

    Measures market cycles using a low pass filter to remove noise.
    Output is bounded between -1 and 1.

    Sources:
        - https://www.prorealcode.com/prorealtime-indicators/even-better-sinewave/
        - J.F.Ehlers 'Cycle Analytics for Traders', 2014

    Args:
        close: Column name or pl.Expr for 'close' prices.
        length: Max cycle/trend period. Default: 40
        bars: Period of low pass filtering. Default: 10
        initial_version: Use the more responsive initial version. Default: False
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: EBSW expression.
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _length = length
    _bars = bars
    _initial = bool(initial_version)
    _offset = offset

    def compute_ebsw(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        out = _nb_ebsw(arr, _length, _bars, _initial)
        if _offset != 0:
            out = np.roll(out, _offset)
            if _offset > 0:
                out[:_offset] = np.nan
            else:
                out[_offset:] = np.nan
        return pl.Series(out)

    return close_expr.map_batches(compute_ebsw, return_dtype=pl.Float64).alias(f"EBSW_{length}_{bars}")
