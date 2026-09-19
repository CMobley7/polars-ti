# -*- coding: utf-8 -*-
# =============================================================================
# Polars MSW (Mesa Sine Wave) Implementation
# =============================================================================
import numpy as np
import polars as pl
from numba import njit

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._rolling import rolling_extreme, rolling_sum
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

    real = imaginary = 0.0
    rotation_real = np.cos(tpi / period)
    rotation_imaginary = np.sin(tpi / period)
    finite = np.isfinite(arr)
    clean = np.where(finite, arr, 0.0)
    invalid = rolling_sum((~finite).astype(np.float64), period)
    magnitude = rolling_sum(np.abs(clean), period)
    maximum = rolling_extreme(np.abs(clean), period, True)[0]
    state_magnitude = 0.0
    for i in range(period, size):
        if i % period == 0 or state_magnitude > maximum[i] * 1e6:
            real = imaginary = 0.0
            state_magnitude = 0.0
            for j in range(period):
                real += clean[i - j] * cos_arr[j]
                imaginary += clean[i - j] * sin_arr[j]
                state_magnitude = max(state_magnitude, abs(clean[i - j]))
        else:
            previous_real = real
            real = clean[i] - clean[i - period] + rotation_real * real - rotation_imaginary * imaginary
            imaginary = rotation_imaginary * previous_real + rotation_real * imaginary
            state_magnitude = max(state_magnitude, abs(clean[i]))
        if invalid[i]:
            continue
        rp = real
        ip = imaginary
        # Preserve phase branch decisions when accumulated roundoff could change
        # the real-component sign or cross the algorithm's 0.001 threshold.
        uncertainty = 64 * np.finfo(np.float64).eps * period * max(magnitude[i], period * state_magnitude)
        if abs(rp) <= uncertainty or abs(abs(rp) - 0.001) <= uncertainty:
            rp = ip = 0.0
            for j in range(period):
                rp += arr[i - j] * cos_arr[j]
                ip += arr[i - j] * sin_arr[j]

        if np.isnan(rp) or np.isnan(ip):
            continue

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
