# -*- coding: utf-8 -*-
from numpy import floor, isnan, nan, roll, zeros, zeros_like
from numba import njit


@njit(cache=True)
def nb_rolling_hl(np_high, np_low, window_size):
    m = np_high.size
    # Each iteration can append TWO entries (a bar that is simultaneously the
    # window local-low and local-high, e.g. on flat runs where high == low), so
    # the buffers must hold the worst case of 2 per bar to avoid OOB writes.
    capacity = 2 * m
    idx = zeros(capacity)
    swing = zeros(capacity)  # where a high = 1 and low = -1
    value = zeros(capacity)

    extremums = 0
    left = int(floor(window_size / 2))
    right = left + 1
    # sample_array = [*[left-window], *[center], *[right-window]]
    for i in range(left, m - right):
        low_center = np_low[i]
        high_center = np_high[i]
        low_window = np_low[i - left : i + right]
        high_window = np_high[i - left : i + right]

        if (low_center <= low_window).all():
            idx[extremums] = i
            swing[extremums] = -1
            value[extremums] = low_center
            extremums += 1

        if (high_center >= high_window).all():
            idx[extremums] = i
            swing[extremums] = 1
            value[extremums] = high_center
            extremums += 1

    return idx[:extremums], swing[:extremums], value[:extremums]


@njit(cache=True)
def nb_find_zigzags_backward(idx, swing, value, deviation):
    zz_idx = zeros_like(idx)
    zz_swing = zeros_like(swing)
    zz_value = zeros_like(value)
    zz_dev = zeros_like(idx)

    zigzags = 0
    zz_idx[zigzags] = idx[-1]
    zz_swing[zigzags] = swing[-1]
    zz_value[zigzags] = value[-1]
    zz_dev[zigzags] = 0

    m = idx.size
    for i in range(m - 2, -1, -1):
        # last point in zigzag is bottom
        if zz_swing[zigzags] == -1:
            if swing[i] == -1:
                if zz_value[zigzags] > value[i] and zigzags > 1:
                    current_dev = (zz_value[zigzags - 1] - value[i]) / value[i]
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev
            else:
                current_dev = (value[i] - zz_value[zigzags]) / value[i]
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev

        # last point in zigzag is peak
        else:
            if swing[i] == 1:
                if zz_value[zigzags] < value[i] and zigzags > 1:
                    current_dev = (value[i] - zz_value[zigzags - 1]) / value[i]
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev
            else:
                current_dev = (zz_value[zigzags] - value[i]) / value[i]
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags - 1] = 100 * current_dev

    _n = zigzags + 1
    return zz_idx[:_n], zz_swing[:_n], zz_value[:_n], zz_dev[:_n]


@njit(cache=True)
def nb_find_zigzags_forward(idx, swing, value, deviation):
    """Calculate zigzag points forward in time for backtest-safe results.

    Unlike nb_find_zigzags which processes backwards (using future data),
    this function processes forward in time, ensuring swing points are
    placed at the candle where they would have been detected in real-time.
    This eliminates look-ahead bias for realistic backtesting.

    Args:
        idx (1d np array): Pivot indices
        swing (1d np array): Pivot swing direction (-1 low, 1 high)
        value (1d np array): Pivot values
        deviation (float): Deviation percentage for reversal detection

    Returns:
        tuple: (indices, swings, values, deviations) arrays
    """
    zz_idx = zeros_like(idx)
    zz_swing = zeros_like(swing)
    zz_value = zeros_like(value)
    zz_dev = zeros_like(idx)

    zigzags = 0
    changes = 0
    zz_idx[zigzags] = idx[0]
    zz_swing[zigzags] = swing[0]
    zz_value[zigzags] = value[0]
    zz_dev[zigzags] = 0

    m = idx.size
    for i in range(1, m):
        last_zz_value = zz_value[zigzags]
        current_dev = (value[i] - last_zz_value) / last_zz_value

        # Last point in zigzag is bottom
        if zz_swing[zigzags - changes] == -1:
            if swing[i] == -1:
                # If the current pivot is lower than the last ZZ bottom:
                # create a new point and log it as a change
                if value[i] < zz_value[zigzags]:
                    if zz_idx[zigzags - changes] == idx[i]:
                        continue
                    zigzags += 1
                    changes += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
            else:
                # If deviation is great enough, create new ZZ point
                if current_dev > 0.01 * deviation:
                    if zz_idx[zigzags - changes] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
                    changes = 0

        # Last point in zigzag is top
        else:
            if swing[i] == 1:
                # If the current pivot is higher than the last ZZ top:
                # create a new point and log it as a change
                if value[i] > zz_value[zigzags]:
                    if zz_idx[zigzags - changes] == idx[i]:
                        continue
                    zigzags += 1
                    changes += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
            else:
                # If deviation is great enough, create new ZZ point
                if current_dev < -0.01 * deviation:
                    if zz_idx[zigzags - changes] == idx[i]:
                        continue
                    zigzags += 1
                    zz_idx[zigzags] = idx[i]
                    zz_swing[zigzags] = swing[i]
                    zz_value[zigzags] = value[i]
                    zz_dev[zigzags] = 100 * current_dev
                    changes = 0

    _n = zigzags + 1
    return zz_idx[:_n], zz_swing[:_n], zz_value[:_n], zz_dev[:_n]


@njit(cache=True)
def nb_map_zigzag(idx, swing, value, deviation, n):
    swing_map = zeros(n)
    value_map = zeros(n)
    dev_map = zeros(n)

    for j, i in enumerate(idx):
        i = int(i)
        swing_map[i] = swing[j]
        value_map[i] = value[j]
        dev_map[i] = deviation[j]

    for i in range(n):
        if swing_map[i] == 0:
            swing_map[i] = nan
            value_map[i] = nan
            dev_map[i] = nan

    return swing_map, value_map, dev_map


# =============================================================================
# Polars ZigZag Implementation (reuses existing Numba kernels)
# =============================================================================
import numpy as np
import polars as pl

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


def zigzag(
    high: IntoExpr,
    low: IntoExpr,
    legs: int = 10,
    deviation: float = 5.0,
    lookahead: bool = True,
    offset: int = 0,
) -> PlExpr:
    """Polars: ZigZag

    Filters out small price movements while highlighting trend direction.
    Identifies swing highs and lows.

    Args:
        high: Column name or pl.Expr for 'high' prices
        low: Column name or pl.Expr for 'low' prices
        legs: Number of legs. Default: 10
        deviation: Price deviation % for reversal. Default: 5.0
        lookahead: Use future data for precise placement. Default: True
        offset: Shift result. Default: 0

    Returns:
        pl.Expr: Struct with ZIGZAGs, ZIGZAGv, ZIGZAGd columns
    """
    high_expr = v_expr(high)
    low_expr = v_expr(low)

    def _compute(s: pl.Series) -> pl.Series:
        data = s.struct.unnest()
        h = data["_h"].to_numpy().astype(np.float64)
        l_ = data["_l"].to_numpy().astype(np.float64)

        hli, hls, hlv = nb_rolling_hl(h, l_, legs)

        if lookahead:
            zzi, zzs, zzv, zzd = nb_find_zigzags_backward(hli, hls, hlv, deviation)
        else:
            zzi, zzs, zzv, zzd = nb_find_zigzags_forward(hli, hls, hlv, deviation)

        zz_swing, zz_value, zz_dev = nb_map_zigzag(zzi, zzs, zzv, zzd, len(h))

        if offset != 0:
            zz_swing = np.roll(zz_swing, offset)
            zz_value = np.roll(zz_value, offset)
            zz_dev = np.roll(zz_dev, offset)
            if offset > 0:
                zz_swing[:offset] = np.nan
                zz_value[:offset] = np.nan
                zz_dev[:offset] = np.nan
            else:
                zz_swing[offset:] = np.nan
                zz_value[offset:] = np.nan
                zz_dev[offset:] = np.nan

        _props = f"_{deviation}%_{legs}"
        n = len(h)
        return pl.Series(
            values=[
                {
                    f"ZIGZAGs{_props}": zz_swing[i],
                    f"ZIGZAGv{_props}": zz_value[i],
                    f"ZIGZAGd{_props}": zz_dev[i],
                }
                for i in range(n)
            ]
        )

    _props = f"_{deviation}%_{legs}"
    fields = [
        pl.Field(f"ZIGZAGs{_props}", pl.Float64),
        pl.Field(f"ZIGZAGv{_props}", pl.Float64),
        pl.Field(f"ZIGZAGd{_props}", pl.Float64),
    ]
    return (
        pl.struct(
            high_expr.alias("_h"),
            low_expr.alias("_l"),
        )
        .map_batches(_compute, return_dtype=pl.Struct(fields))
        .alias(f"ZIGZAG{_props}")
    )
