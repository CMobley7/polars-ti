# -*- coding: utf-8 -*-
from numba import njit

from polars_ti.utils._core import nb_non_zero_range


@njit(cache=True)
def pivot_camarilla(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = close - 11 / 120 * hl_range
    s2 = close - 11 / 60 * hl_range
    s3 = close - 0.275 * hl_range
    s4 = close - 0.55 * hl_range

    r1 = close + 11 / 120 * hl_range
    r2 = close + 11 / 60 * hl_range
    r3 = close + 0.275 * hl_range
    r4 = close + 0.55 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_classic(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2 * hl_range
    s4 = tp - 3 * hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2 * hl_range
    r4 = tp + 3 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_demark(open_, high, low, close):
    if (open_ == close).all():
        tp = 0.25 * (high + low + 2 * close)
    elif (close > open_).all():
        tp = 0.25 * (2 * high + low + close)
    else:
        tp = 0.25 * (high + 2 * low + close)

    s1 = 2 * tp - high
    r1 = 2 * tp - low

    return tp, s1, r1


@njit(cache=True)
def pivot_fibonacci(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = tp - 0.382 * hl_range
    s2 = tp - 0.618 * hl_range
    s3 = tp - hl_range

    r1 = tp + 0.382 * hl_range
    r2 = tp + 0.618 * hl_range
    r3 = tp + hl_range

    return tp, s1, s2, s3, r1, r2, r3


@njit(cache=True)
def pivot_traditional(high, low, close):
    tp = (high + low + close) / 3
    hl_range = nb_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = tp - 2 * hl_range
    s4 = tp - 2 * hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = tp + 2 * hl_range
    r4 = tp + 2 * hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


@njit(cache=True)
def pivot_woodie(open_, high, low):
    tp = (2 * open_ + high + low) / 4
    hl_range = nb_non_zero_range(high, low)

    s1 = 2 * tp - high
    s2 = tp - hl_range
    s3 = low - 2 * (high - tp)
    s4 = s3 - hl_range

    r1 = 2 * tp - low
    r2 = tp + hl_range
    r3 = high + 2 * (tp - low)
    r4 = r3 + hl_range

    return tp, s1, s2, s3, s4, r1, r2, r3, r4


# =============================================================================
# Polars PIVOTS Implementation (Pure Polars + Numba)
# =============================================================================
import polars as pl
import numpy as np

from polars_ti._typing import IntoExpr, PlExpr
from polars_ti.utils._validate import v_expr


# Anchor frequency mapping to Polars duration
_anchor_to_duration = {
    "D": "1d",
    "W": "1w",
    "M": "1mo",
    "ME": "1mo",
    "Y": "1y",
    "YE": "1y",
}


def pivots(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    method: str = "traditional",
    anchor: str = "D",
) -> pl.DataFrame:
    """Polars: Pivot Points (Pure Polars + Numba)

    Calculates support and resistance levels from previous price action.
    Uses Polars group_by_dynamic for resampling and Numba kernels for calculation.

    Args:
        df: Polars DataFrame with OHLC and datetime column
        open_col: Column name for 'open'. Default: "open"
        high_col: Column name for 'high'. Default: "high"
        low_col: Column name for 'low'. Default: "low"
        close_col: Column name for 'close'. Default: "close"
        date_col: Column name for datetime. Default: "date"
        method: Pivot calculation method. Default: "traditional"
            Options: traditional, fibonacci, woodie, classic, demark, camarilla
        anchor: Anchor frequency (D, W, M, Y). Default: "D"

    Returns:
        pl.DataFrame: Pivot point columns (P, S1-S4, R1-R4)
    """
    method = method.lower()
    anchor = anchor.upper()
    _props = f"PIVOTS_{method[:4].upper()}_{anchor}"

    duration = _anchor_to_duration.get(anchor, "1d")

    # Resample OHLC to anchor period
    resampled = (
        df.sort(date_col)
        .group_by_dynamic(date_col, every=duration)
        .agg(
            [
                pl.col(open_col).first().alias("_open"),
                pl.col(high_col).max().alias("_high"),
                pl.col(low_col).min().alias("_low"),
                pl.col(close_col).last().alias("_close"),
            ]
        )
        .drop_nulls()
    )

    # Extract arrays for Numba
    np_open = resampled["_open"].to_numpy().astype(np.float64)
    np_high = resampled["_high"].to_numpy().astype(np.float64)
    np_low = resampled["_low"].to_numpy().astype(np.float64)
    np_close = resampled["_close"].to_numpy().astype(np.float64)

    # Calculate pivots using existing Numba kernels
    n = len(np_close)
    nan_array = np.full(n, np.nan)

    if method == "camarilla":
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_camarilla(np_high, np_low, np_close)
    elif method == "classic":
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_classic(np_high, np_low, np_close)
    elif method == "demark":
        tp, s1, r1 = pivot_demark(np_open, np_high, np_low, np_close)
        s2, s3, s4, r2, r3, r4 = (
            nan_array,
            nan_array,
            nan_array,
            nan_array,
            nan_array,
            nan_array,
        )
    elif method == "fibonacci":
        tp, s1, s2, s3, r1, r2, r3 = pivot_fibonacci(np_high, np_low, np_close)
        s4, r4 = nan_array, nan_array
    elif method == "woodie":
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_woodie(np_open, np_high, np_low)
    else:  # Traditional
        tp, s1, s2, s3, s4, r1, r2, r3, r4 = pivot_traditional(np_high, np_low, np_close)

    # Build result DataFrame with pivot values
    pivot_df = pl.DataFrame(
        {
            date_col: resampled[date_col],
            f"{_props}_P": tp,
            f"{_props}_S1": s1,
            f"{_props}_S2": s2,
            f"{_props}_S3": s3,
            f"{_props}_S4": s4,
            f"{_props}_R1": r1,
            f"{_props}_R2": r2,
            f"{_props}_R3": r3,
            f"{_props}_R4": r4,
        }
    )

    # Shift pivot dates forward by one period (pivots apply to next period)
    offset_map = {"D": "1d", "W": "1w", "M": "1mo", "ME": "1mo", "Y": "1y", "YE": "1y"}
    if anchor in offset_map:
        pivot_df = pivot_df.with_columns(pl.col(date_col).dt.offset_by(offset_map[anchor]))

    # Join back to original DataFrame and forward-fill
    result = (
        df.select(date_col)
        .join(pivot_df, on=date_col, how="left")
        .select(pl.exclude(date_col))
        .fill_null(strategy="forward")
    )

    # Drop all-NaN columns for demark and fibonacci
    if method in ["demark", "fibonacci"]:
        cols_to_keep = [c for c in result.columns if not (result[c].is_null().all() or result[c].is_nan().all())]
        result = result.select(cols_to_keep)

    return result
