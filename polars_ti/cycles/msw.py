# -*- coding: utf-8 -*-
# =============================================================================
# Polars MSW (Mesa Sine Wave) Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr, v_pos_int


@njit(cache=True)
def nb_msw(arr: np.ndarray, period: int):
    """Numba-optimized Mesa Sine Wave — matches Tulip Indicators algorithm."""
    pi = np.pi
    tpi = 2.0 * pi
    size = len(arr)
    sine = np.full(size, np.nan)
    lead = np.full(size, np.nan)

    if period > size:
        # Window larger than the data -> no full cycle exists, so the output is
        # all NaN. Return before allocating the O(period) basis vectors, which on
        # an absurd period (e.g. 1e9) would exhaust memory / hang.
        return sine, lead

    # Precompute cos/sin basis vectors
    cos_arr = np.empty(period)
    sin_arr = np.empty(period)
    for j in range(period):
        cos_arr[j] = np.cos(tpi * j / period)
        sin_arr[j] = np.sin(tpi * j / period)

    for i in range(period, size):
        rp = 0.0
        ip = 0.0
        for j in range(period):
            v = arr[i - j]  # newest first: j=0 is arr[i]
            rp += v * cos_arr[j]
            ip += v * sin_arr[j]

        if abs(rp) > 0.001:
            phase = np.arctan(ip / rp)
        else:
            if ip < 0.0:
                phase = -(tpi / 2.0)
            else:
                phase = tpi / 2.0

        if rp < 0.0:
            phase += pi
        phase += pi / 2.0
        if phase < 0.0:
            phase += tpi
        if phase > tpi:
            phase -= tpi

        sine[i] = np.sin(phase)
        lead[i] = np.sin(phase + pi / 4.0)

    return sine, lead


def msw(
    close: IntoExpr,
    period: int = 5,
    offset: int = 0,
) -> PlExpr:
    """Polars: Mesa Sine Wave (MSW)

    Identifies cycles using a DFT-based approach from Ehlers (2001).
    Returns two oscillator series: sine and lead (sine + 45 degrees).

    Sources:
        Tulip Indicators: https://tulipindicators.org/msw
        Ehlers, John F. (2001) Rocket Science For Traders

    Args:
        close: Column name or pl.Expr for 'close' prices.
        period: Lookback period. Default: 5
        offset: Shift result by N periods. Default: 0

    Returns:
        pl.Expr: Struct expression with fields:
            - sine:     sin(phase)
            - lead:     sin(phase + 45 degrees)
    """
    close_expr = v_expr(close)
    if close_expr is None:
        return None

    _period = v_pos_int(period, "period")
    _offset = offset

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy().astype(np.float64)
        sine_arr, lead_arr = nb_msw(arr, _period)
        return pl.DataFrame(
            {
                "sine": sine_arr,
                "lead": lead_arr,
            }
        ).to_struct(f"MSW_{_period}")

    result_expr = close_expr.map_batches(
        _compute,
        return_dtype=pl.Struct(
            [
                pl.Field("sine", pl.Float64),
                pl.Field("lead", pl.Float64),
            ]
        ),
    )

    if _offset != 0:
        result_expr = result_expr.shift(_offset)

    return result_expr.alias(f"MSW_{period}")
